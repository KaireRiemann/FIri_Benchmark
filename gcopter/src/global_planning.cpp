#include "misc/visualizer.hpp"
#include "gcopter/trajectory.hpp"
#include "gcopter/gcopter.hpp"
#include "gcopter/firi.hpp"
#include "gcopter/flatness.hpp"
#include "gcopter/voxel_map.hpp"
#include "gcopter/sfc_gen.hpp"
#include "gcopter/sfc_gen_benchmark.hpp"

#include <ros/ros.h>
#include <ros/console.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <random>

struct Config
{
    std::string mapTopic;
    std::string targetTopic;
    double dilateRadius;
    double voxelWidth;
    std::vector<double> mapBound;
    double timeoutRRT;
    double maxVelMag;
    double maxBdrMag;
    double maxTiltAngle;
    double minThrust;
    double maxThrust;
    double vehicleMass;
    double gravAcc;
    double horizDrag;
    double vertDrag;
    double parasDrag;
    double speedEps;
    double weightT;
    std::vector<double> chiVec;
    double smoothingEps;
    int integralIntervs;
    double relCostTol;

    Config(const ros::NodeHandle &nh_priv)
    {
        nh_priv.getParam("MapTopic", mapTopic);
        nh_priv.getParam("TargetTopic", targetTopic);
        nh_priv.getParam("DilateRadius", dilateRadius);
        nh_priv.getParam("VoxelWidth", voxelWidth);
        nh_priv.getParam("MapBound", mapBound);
        nh_priv.getParam("TimeoutRRT", timeoutRRT);
        nh_priv.getParam("MaxVelMag", maxVelMag);
        nh_priv.getParam("MaxBdrMag", maxBdrMag);
        nh_priv.getParam("MaxTiltAngle", maxTiltAngle);
        nh_priv.getParam("MinThrust", minThrust);
        nh_priv.getParam("MaxThrust", maxThrust);
        nh_priv.getParam("VehicleMass", vehicleMass);
        nh_priv.getParam("GravAcc", gravAcc);
        nh_priv.getParam("HorizDrag", horizDrag);
        nh_priv.getParam("VertDrag", vertDrag);
        nh_priv.getParam("ParasDrag", parasDrag);
        nh_priv.getParam("SpeedEps", speedEps);
        nh_priv.getParam("WeightT", weightT);
        nh_priv.getParam("ChiVec", chiVec);
        nh_priv.getParam("SmoothingEps", smoothingEps);
        nh_priv.getParam("IntegralIntervs", integralIntervs);
        nh_priv.getParam("RelCostTol", relCostTol);
    }
};

class GlobalPlanner
{
private:
    Config config;

    ros::NodeHandle nh;
    ros::Subscriber mapSub;
    ros::Subscriber targetSub;

    bool mapInitialized;
    voxel_map::VoxelMap voxelMap;
    Visualizer visualizer;
    std::vector<Eigen::Vector3d> startGoal;

    Trajectory<5> traj;
    double trajStamp;

public:
    GlobalPlanner(const Config &conf,
                  ros::NodeHandle &nh_)
        : config(conf),
          nh(nh_),
          mapInitialized(false),
          visualizer(nh)
    {
        const Eigen::Vector3i xyz((config.mapBound[1] - config.mapBound[0]) / config.voxelWidth,
                                  (config.mapBound[3] - config.mapBound[2]) / config.voxelWidth,
                                  (config.mapBound[5] - config.mapBound[4]) / config.voxelWidth);

        const Eigen::Vector3d offset(config.mapBound[0], config.mapBound[2], config.mapBound[4]);

        voxelMap = voxel_map::VoxelMap(xyz, offset, config.voxelWidth);

        mapSub = nh.subscribe(config.mapTopic, 1, &GlobalPlanner::mapCallBack, this,
                              ros::TransportHints().tcpNoDelay());

        targetSub = nh.subscribe(config.targetTopic, 1, &GlobalPlanner::targetCallBack, this,
                                 ros::TransportHints().tcpNoDelay());
    }

    inline void mapCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        if (!mapInitialized)
        {
            size_t cur = 0;
            const size_t total = msg->data.size() / msg->point_step;
            float *fdata = (float *)(&msg->data[0]);
            for (size_t i = 0; i < total; i++)
            {
                cur = msg->point_step / sizeof(float) * i;

                if (std::isnan(fdata[cur + 0]) || std::isinf(fdata[cur + 0]) ||
                    std::isnan(fdata[cur + 1]) || std::isinf(fdata[cur + 1]) ||
                    std::isnan(fdata[cur + 2]) || std::isinf(fdata[cur + 2]))
                {
                    continue;
                }
                voxelMap.setOccupied(Eigen::Vector3d(fdata[cur + 0],
                                                     fdata[cur + 1],
                                                     fdata[cur + 2]));
            }

            voxelMap.dilate(std::ceil(config.dilateRadius / voxelMap.getScale()));

            mapInitialized = true;
        }
    }

    inline void plan()
    {
        if (startGoal.size() == 2)
        {
            std::vector<Eigen::Vector3d> route;
            sfc_gen::planPath<voxel_map::VoxelMap>(startGoal[0],
                                                   startGoal[1],
                                                   voxelMap.getOrigin(),
                                                   voxelMap.getCorner(),
                                                   &voxelMap, 0.01,
                                                   route);
            std::vector<Eigen::Vector3d> pc;
            voxelMap.getSurf(pc);

            if (route.size() > 1)
            {
                const sfc_gen_benchmark::BenchmarkResult benchmark =
                    sfc_gen_benchmark::convexCover(route,
                                                   pc,
                                                   voxelMap.getOrigin(),
                                                   voxelMap.getCorner(),
                                                   7.0,
                                                   3.0);
                benchmark.printBenchmarkResult();

                std::vector<Eigen::MatrixX4d> hPolys = benchmark.firi.hpolys;
                std::vector<Eigen::MatrixX4d> hPolys_opt = benchmark.firi_opt.hpolys;
                std::vector<Eigen::MatrixX4d> hPolys_nd = benchmark.firi_nd.hpolys;

                sfc_gen_benchmark::shortCut(hPolys);
                sfc_gen_benchmark::shortCut(hPolys_opt);
                sfc_gen_benchmark::shortCut(hPolys_nd);

                std::vector<double> firi_mesh_color = {0.0, 0.0, 1.0, 0.85};  
                std::vector<double> firi_edge_color = {0.0, 1.0, 1.0, 1.0}; 
                visualizer.visualizePolytope(hPolys,"firi",firi_mesh_color,firi_edge_color);
                double firi_volume = 0.0;
                for(size_t i = 0; i < hPolys.size(); i++)
                {
                    const double v1 = sfc_gen_benchmark::calculateExactPolyhedronVolume(hPolys[i]);
                    firi_volume += v1;
                }
                

                std::vector<double> opt_mesh_color = {1.0, 0.45, 0.0, 0.18};  
                std::vector<double> opt_edge_color = {1.0, 0.45, 0.0, 1.0};
                visualizer.visualizePolytope(hPolys_opt, "firi_opt", opt_mesh_color, opt_edge_color);
                double firi_opt_volume = 0.0;
                for(size_t i = 0; i < hPolys_opt.size(); i++)
                {
                    const double v2 = sfc_gen_benchmark::calculateExactPolyhedronVolume(hPolys_opt[i]);
                    firi_opt_volume += v2;
                }

                std::vector<double> nd_mesh_color = {0.0, 0.8, 0.2, 0.15};  
                std::vector<double> nd_edge_color = {0.0, 0.8, 0.2, 1.0};
                visualizer.visualizePolytope(hPolys_nd, "firi_nd", nd_mesh_color, nd_edge_color);
                double firi_nd_volume = 0.0;
                for(size_t i = 0; i < hPolys_nd.size(); i++)
                {
                    const double v3 = sfc_gen_benchmark::calculateExactPolyhedronVolume(hPolys_nd[i]);
                    firi_nd_volume += v3;
                }
                
                if (firi_volume > 0.0)
                {
                    std::cout<<"firi_opt/firi volume ratio : "<<firi_opt_volume / firi_volume<<std::endl;
                    std::cout<<"firi_nd/firi volume ratio : "<<firi_nd_volume / firi_volume<<std::endl;
                }

                Eigen::Matrix3d iniState;
                Eigen::Matrix3d finState;
                iniState << route.front(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
                finState << route.back(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

                gcopter::GCOPTER_PolytopeSFC gcopter;
                //gcopter_bs::GCOPTER_PolytopeSFC gcopter_bs;

                // magnitudeBounds = [v_max, omg_max, theta_max, thrust_min, thrust_max]^T
                // penaltyWeights = [pos_weight, vel_weight, omg_weight, theta_weight, thrust_weight]^T
                // physicalParams = [vehicle_mass, gravitational_acceleration, horitonral_drag_coeff,
                //                   vertical_drag_coeff, parasitic_drag_coeff, speed_smooth_factor]^T
                // initialize some constraint parameters
                Eigen::VectorXd magnitudeBounds(5);
                Eigen::VectorXd penaltyWeights(5);
                Eigen::VectorXd physicalParams(6);
                magnitudeBounds(0) = config.maxVelMag;
                magnitudeBounds(1) = config.maxBdrMag;
                magnitudeBounds(2) = config.maxTiltAngle;
                magnitudeBounds(3) = config.minThrust;
                magnitudeBounds(4) = config.maxThrust;
                penaltyWeights(0) = (config.chiVec)[0];
                penaltyWeights(1) = (config.chiVec)[1];
                penaltyWeights(2) = (config.chiVec)[2];
                penaltyWeights(3) = (config.chiVec)[3];
                penaltyWeights(4) = (config.chiVec)[4];
                physicalParams(0) = config.vehicleMass;
                physicalParams(1) = config.gravAcc;
                physicalParams(2) = config.horizDrag;
                physicalParams(3) = config.vertDrag;
                physicalParams(4) = config.parasDrag;
                physicalParams(5) = config.speedEps;
                const int quadratureRes = config.integralIntervs;

                traj.clear();

                if (hPolys_nd.empty())
                {
                    return;
                }

                if (!gcopter.setup(config.weightT,
                                   iniState, finState,
                                   hPolys_nd, INFINITY,
                                   config.smoothingEps,
                                   quadratureRes,
                                   magnitudeBounds,
                                   penaltyWeights,
                                   physicalParams))
                {
                    return;
                }

                if (std::isinf(gcopter.optimize(traj, config.relCostTol)))
                {
                    return;
                }

                if (traj.getPieceNum() > 0)
                {
                    trajStamp = ros::Time::now().toSec();
                    visualizer.visualize(traj, route);
                }
            }
        }
    }

    inline void targetCallBack(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        if (mapInitialized)
        {
            if (startGoal.size() >= 2)
            {
                startGoal.clear();
            }
            const double zGoal = config.mapBound[4] + config.dilateRadius +
                                 fabs(msg->pose.orientation.z) *
                                     (config.mapBound[5] - config.mapBound[4] - 2 * config.dilateRadius);
            const Eigen::Vector3d goal(msg->pose.position.x, msg->pose.position.y, zGoal);
            if (voxelMap.query(goal) == 0)
            {
                visualizer.visualizeStartGoal(goal, 0.5, startGoal.size());
                startGoal.emplace_back(goal);
            }
            else
            {
                ROS_WARN("Infeasible Position Selected !!!\n");
            }

            plan();
        }
        return;
    }

    inline void process()
    {
        Eigen::VectorXd physicalParams(6);
        physicalParams(0) = config.vehicleMass;
        physicalParams(1) = config.gravAcc;
        physicalParams(2) = config.horizDrag;
        physicalParams(3) = config.vertDrag;
        physicalParams(4) = config.parasDrag;
        physicalParams(5) = config.speedEps;

        flatness::FlatnessMap flatmap;
        flatmap.reset(physicalParams(0), physicalParams(1), physicalParams(2),
                      physicalParams(3), physicalParams(4), physicalParams(5));

        if (traj.getPieceNum() > 0)
        {
            const double delta = ros::Time::now().toSec() - trajStamp;
            if (delta > 0.0 && delta < traj.getTotalDuration())
            {
                double thr;
                Eigen::Vector4d quat;
                Eigen::Vector3d omg;

                flatmap.forward(traj.getVel(delta),
                                traj.getAcc(delta),
                                traj.getJer(delta),
                                0.0, 0.0,
                                thr, quat, omg);
                double speed = traj.getVel(delta).norm();
                double bodyratemag = omg.norm();
                double tiltangle = acos(1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2)));
                std_msgs::Float64 speedMsg, thrMsg, tiltMsg, bdrMsg;
                speedMsg.data = speed;
                thrMsg.data = thr;
                tiltMsg.data = tiltangle;
                bdrMsg.data = bodyratemag;
                visualizer.speedPub.publish(speedMsg);
                visualizer.thrPub.publish(thrMsg);
                visualizer.tiltPub.publish(tiltMsg);
                visualizer.bdrPub.publish(bdrMsg);

                visualizer.visualizeSphere(traj.getPos(delta),
                                           config.dilateRadius);
            }
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "global_planning_node");
    ros::NodeHandle nh_;

    GlobalPlanner global_planner(Config(ros::NodeHandle("~")), nh_);

    ros::Rate lr(1000);
    while (ros::ok())
    {
        global_planner.process();
        ros::spinOnce();
        lr.sleep();
    }

    return 0;
}

// #include "misc/visualizer.hpp"
// #include "gcopter/flatness.hpp"
// #include "gcopter/voxel_map.hpp"
// #include "gcopter/sfc_gen.hpp"
// #include "gcopter/sfc_gen_opt.hpp"

// // 引入我们的 NUBS 终极优化器
// #include "gcopter/nubs_optimizer.hpp" 

// #include <ros/ros.h>
// #include <ros/console.h>
// #include <geometry_msgs/PoseStamped.h>
// #include <sensor_msgs/PointCloud2.h>
// #include <nav_msgs/Path.h> 

// #include <cmath>
// #include <iostream>
// #include <string>
// #include <vector>
// #include <chrono>

// struct Config
// {
//     std::string mapTopic;
//     std::string targetTopic;
//     double dilateRadius;
//     double voxelWidth;
//     std::vector<double> mapBound;
//     double maxVelMag;
//     double maxBdrMag;
//     double maxTiltAngle;
//     double minThrust;
//     double maxThrust;
//     double vehicleMass;
//     double gravAcc;
//     double horizDrag;
//     double vertDrag;
//     double parasDrag;
//     double speedEps;
//     double smoothingEps;
//     double relCostTol;

//     Config(const ros::NodeHandle &nh_priv)
//     {
//         nh_priv.getParam("MapTopic", mapTopic);
//         nh_priv.getParam("TargetTopic", targetTopic);
//         nh_priv.getParam("DilateRadius", dilateRadius);
//         nh_priv.getParam("VoxelWidth", voxelWidth);
//         nh_priv.getParam("MapBound", mapBound);
//         nh_priv.getParam("MaxVelMag", maxVelMag);
//         nh_priv.getParam("MaxBdrMag", maxBdrMag);
//         nh_priv.getParam("MaxTiltAngle", maxTiltAngle);
//         nh_priv.getParam("MinThrust", minThrust);
//         nh_priv.getParam("MaxThrust", maxThrust);
//         nh_priv.getParam("VehicleMass", vehicleMass);
//         nh_priv.getParam("GravAcc", gravAcc);
//         nh_priv.getParam("HorizDrag", horizDrag);
//         nh_priv.getParam("VertDrag", vertDrag);
//         nh_priv.getParam("ParasDrag", parasDrag);
//         nh_priv.getParam("SpeedEps", speedEps);
//         nh_priv.getParam("SmoothingEps", smoothingEps);
//         nh_priv.getParam("RelCostTol", relCostTol);
//     }
// };

// class GlobalPlanner
// {
// private:
//     Config config;
//     ros::NodeHandle nh;
//     ros::Subscriber mapSub;
//     ros::Subscriber targetSub;
//     ros::Publisher bsplinePub; // 专门发布 B 样条可视化路径

//     bool mapInitialized;
//     voxel_map::VoxelMap voxelMap;
//     Visualizer visualizer;
//     std::vector<Eigen::Vector3d> startGoal;

//     nubs::NUBSTrajectory<3> nubs_traj;
//     bool has_traj;
//     double trajStamp;

//     // 根据 A* 路径计算合理的时间分配（替代原版时间放入 L-BFGS）
//     inline void getInitialWaypointsAndTimes(const Eigen::Matrix3Xd &path,
//                                             const double &speed,
//                                             const Eigen::VectorXi &intervalNs,
//                                             Eigen::VectorXd &timeAlloc,
//                                             Eigen::MatrixXd &waypoints)
//     {
//         const int sizeM = intervalNs.size();
//         const int sizeN = intervalNs.sum();
//         waypoints.resize(sizeN + 1, 3);
//         timeAlloc.resize(sizeN);

//         waypoints.row(0) = path.col(0).transpose();
//         Eigen::Vector3d a, b, c;
//         for (int i = 0, j = 0, k = 1, l; i < sizeM; i++)
//         {
//             l = intervalNs(i);
//             a = path.col(i);
//             b = path.col(i + 1);
//             c = (b - a) / l;
//             // 为了安全，分配稍多一点时间，留给空间去优化拉平
//             timeAlloc.segment(j, l).setConstant(c.norm() / (speed * 0.8)); 
//             j += l;
//             for (int m = 1; m <= l; m++)
//             {
//                 waypoints.row(k++) = (a + c * m).transpose();
//             }
//         }
//     }

// public:
//     GlobalPlanner(const Config &conf, ros::NodeHandle &nh_)
//         : config(conf), nh(nh_), mapInitialized(false), visualizer(nh), has_traj(false)
//     {
//         const Eigen::Vector3i xyz((config.mapBound[1] - config.mapBound[0]) / config.voxelWidth,
//                                   (config.mapBound[3] - config.mapBound[2]) / config.voxelWidth,
//                                   (config.mapBound[5] - config.mapBound[4]) / config.voxelWidth);

//         const Eigen::Vector3d offset(config.mapBound[0], config.mapBound[2], config.mapBound[4]);
//         voxelMap = voxel_map::VoxelMap(xyz, offset, config.voxelWidth);

//         mapSub = nh.subscribe(config.mapTopic, 1, &GlobalPlanner::mapCallBack, this, ros::TransportHints().tcpNoDelay());
//         targetSub = nh.subscribe(config.targetTopic, 1, &GlobalPlanner::targetCallBack, this, ros::TransportHints().tcpNoDelay());
        
//         bsplinePub = nh.advertise<nav_msgs::Path>("/nubs_trajectory", 1);
//     }

//     inline void mapCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg)
//     {
//         if (!mapInitialized)
//         {
//             size_t cur = 0;
//             const size_t total = msg->data.size() / msg->point_step;
//             float *fdata = (float *)(&msg->data[0]);
//             for (size_t i = 0; i < total; i++)
//             {
//                 cur = msg->point_step / sizeof(float) * i;
//                 if (std::isnan(fdata[cur]) || std::isnan(fdata[cur + 1]) || std::isnan(fdata[cur + 2])) continue;
//                 voxelMap.setOccupied(Eigen::Vector3d(fdata[cur], fdata[cur + 1], fdata[cur + 2]));
//             }
//             voxelMap.dilate(std::ceil(config.dilateRadius / voxelMap.getScale()));
//             mapInitialized = true;
//         }
//     }

//     inline void plan()
//     {
//         if (startGoal.size() == 2)
//         {
//             std::vector<Eigen::Vector3d> route;
//             sfc_gen::planPath<voxel_map::VoxelMap>(startGoal[0], startGoal[1],
//                                                    voxelMap.getOrigin(), voxelMap.getCorner(),
//                                                    &voxelMap, 0.01, route);
            
//             std::vector<Eigen::MatrixX4d> hPolys, hPolys_new;
//             std::vector<Eigen::Vector3d> pc;
//             voxelMap.getSurf(pc);
            
//             sfc_gen_opt::convexCover(route, pc, voxelMap.getOrigin(), voxelMap.getCorner(), 7.0, 3.0, hPolys, hPolys_new);
//             sfc_gen_opt::shortCut(hPolys_new);

//             if (route.size() > 1)
//             {
//                 std::vector<double> opt_mesh_color = {1.0, 0.0, 0.0, 0.15};  
//                 std::vector<double> opt_edge_color = {1.0, 0.0, 0.0, 1.0};
//                 visualizer.visualizePolytope(hPolys_new, "new_firi_opt", opt_mesh_color, opt_edge_color);

//                 Eigen::Matrix3d iniState, finState;
//                 iniState << route.front(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
//                 finState << route.back(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

//                 Eigen::Matrix3Xd shortPath(3, route.size());
//                 for (size_t i = 0; i < route.size(); i++) shortPath.col(i) = route[i];
                
//                 const Eigen::Matrix3Xd deltas = shortPath.rightCols(route.size()-1) - shortPath.leftCols(route.size()-1);
//                 double lengthPerPiece = 1.0; 
//                 Eigen::VectorXi pieceIdx = (deltas.colwise().norm() / lengthPerPiece).cast<int>().transpose();
//                 pieceIdx.array() += 1;

//                 Eigen::VectorXd times_init;
//                 Eigen::MatrixXd waypoints_init;
//                 getInitialWaypointsAndTimes(shortPath, config.maxVelMag, pieceIdx, times_init, waypoints_init);

//                 double max_a = config.maxThrust - config.gravAcc;
//                 if (max_a < 1.0) max_a = 5.0;

//                 auto t1 = std::chrono::high_resolution_clock::now();
                
//                 // ★ 调用新版的 NUBS 极速纯控制点优化器
//                 nubs::NUBSOptimizer optimizer(3);
//                 optimizer.setup(iniState, finState, hPolys_new, config.maxVelMag, max_a);

//                 if (optimizer.optimize(times_init, waypoints_init, nubs_traj))
//                 {
//                     auto t2 = std::chrono::high_resolution_clock::now();
//                     double dt = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
//                     ROS_INFO("\033[1;32m NUBS Trajectory Optimization Succeeded in %.2f ms! \033[0m", dt);

//                     has_traj = true;
//                     trajStamp = ros::Time::now().toSec();

//                     // 发布红色 B-Spline 轨迹给 Rviz
//                     nav_msgs::Path path_msg;
//                     path_msg.header.stamp = ros::Time::now();
//                     path_msg.header.frame_id = "world"; 

//                     double total_time = nubs_traj.getTotalDuration();
//                     for (double t = 0.0; t <= total_time; t += 0.05) {
//                         Eigen::Vector3d pt = nubs_traj.evaluate(t, 0); 
//                         geometry_msgs::PoseStamped pose;
//                         pose.pose.position.x = pt.x();
//                         pose.pose.position.y = pt.y();
//                         pose.pose.position.z = pt.z();
//                         path_msg.poses.push_back(pose);
//                     }
//                     bsplinePub.publish(path_msg);
//                 }
//                 else
//                 {
//                     has_traj = false;
//                     ROS_ERROR("\033[1;31m NUBS Trajectory Optimization Failed. \033[0m");
//                 }
//             }
//         }
//     }

//     inline void targetCallBack(const geometry_msgs::PoseStamped::ConstPtr &msg)
//     {
//         if (mapInitialized)
//         {
//             if (startGoal.size() >= 2) startGoal.clear();
//             const double zGoal = config.mapBound[4] + config.dilateRadius +
//                                  fabs(msg->pose.orientation.z) * (config.mapBound[5] - config.mapBound[4] - 2 * config.dilateRadius);
//             const Eigen::Vector3d goal(msg->pose.position.x, msg->pose.position.y, zGoal);
            
//             if (voxelMap.query(goal) == 0) {
//                 visualizer.visualizeStartGoal(goal, 0.5, startGoal.size());
//                 startGoal.emplace_back(goal);
//             } else {
//                 ROS_WARN("Infeasible Position Selected !!!");
//             }
//             plan();
//         }
//     }

//     inline void process()
//     {
//         Eigen::VectorXd physicalParams(6);
//         physicalParams(0) = config.vehicleMass;
//         physicalParams(1) = config.gravAcc;
//         physicalParams(2) = config.horizDrag;
//         physicalParams(3) = config.vertDrag;
//         physicalParams(4) = config.parasDrag;
//         physicalParams(5) = config.speedEps;

//         flatness::FlatnessMap flatmap;
//         flatmap.reset(physicalParams(0), physicalParams(1), physicalParams(2),
//                       physicalParams(3), physicalParams(4), physicalParams(5));

//         // 核心循环：极速采样 NUBS 发送无人机指令
//         if (has_traj)
//         {
//             const double delta = ros::Time::now().toSec() - trajStamp;
//             if (delta > 0.0 && delta < nubs_traj.getTotalDuration())
//             {
//                 double thr; Eigen::Vector4d quat; Eigen::Vector3d omg;

//                 // 利用局部支撑特性，极速获取微积分状态
//                 Eigen::Vector3d pos = nubs_traj.evaluate(delta, 0);
//                 Eigen::Vector3d vel = nubs_traj.evaluate(delta, 1);
//                 Eigen::Vector3d acc = nubs_traj.evaluate(delta, 2);
//                 Eigen::Vector3d jer = nubs_traj.evaluate(delta, 3);

//                 flatmap.forward(vel, acc, jer, 0.0, 0.0, thr, quat, omg);

//                 double tiltangle = acos(1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2)));
                
//                 std_msgs::Float64 speedMsg, thrMsg, tiltMsg, bdrMsg;
//                 speedMsg.data = vel.norm();
//                 thrMsg.data = thr;
//                 tiltMsg.data = tiltangle;
//                 bdrMsg.data = omg.norm();
                
//                 visualizer.speedPub.publish(speedMsg);
//                 visualizer.thrPub.publish(thrMsg);
//                 visualizer.tiltPub.publish(tiltMsg);
//                 visualizer.bdrPub.publish(bdrMsg);

//                 visualizer.visualizeSphere(pos, config.dilateRadius);
//             }
//         }
//     }
// };

// int main(int argc, char **argv)
// {
//     ros::init(argc, argv, "global_planning_node");
//     ros::NodeHandle nh_;
//     GlobalPlanner global_planner(Config(ros::NodeHandle("~")), nh_);

//     ros::Rate lr(1000);
//     while (ros::ok())
//     {
//         global_planner.process();
//         ros::spinOnce();
//         lr.sleep();
//     }
//     return 0;
// }

// #include "misc/visualizer.hpp"
// #include "gcopter/flatness.hpp"
// #include "gcopter/voxel_map.hpp"
// #include "gcopter/sfc_gen.hpp"
// #include "gcopter/sfc_gen_opt.hpp"
// #include "gcopter/nubs_optimizer.hpp" 

// #include <ros/ros.h>
// #include <ros/console.h>
// #include <geometry_msgs/PoseStamped.h>
// #include <sensor_msgs/PointCloud2.h>
// #include <nav_msgs/Path.h> 

// #include <cmath>
// #include <iostream>
// #include <string>
// #include <vector>
// #include <chrono>

// struct Config
// {
//     std::string mapTopic;
//     std::string targetTopic;
//     double dilateRadius;
//     double voxelWidth;
//     std::vector<double> mapBound;
//     double maxVelMag;
//     double maxBdrMag;
//     double maxTiltAngle;
//     double minThrust;
//     double maxThrust;
//     double vehicleMass;
//     double gravAcc;
//     double horizDrag;
//     double vertDrag;
//     double parasDrag;
//     double speedEps;
//     double smoothingEps;
//     double relCostTol;

//     Config(const ros::NodeHandle &nh_priv)
//     {
//         nh_priv.getParam("MapTopic", mapTopic);
//         nh_priv.getParam("TargetTopic", targetTopic);
//         nh_priv.getParam("DilateRadius", dilateRadius);
//         nh_priv.getParam("VoxelWidth", voxelWidth);
//         nh_priv.getParam("MapBound", mapBound);
//         nh_priv.getParam("MaxVelMag", maxVelMag);
//         nh_priv.getParam("MaxBdrMag", maxBdrMag);
//         nh_priv.getParam("MaxTiltAngle", maxTiltAngle);
//         nh_priv.getParam("MinThrust", minThrust);
//         nh_priv.getParam("MaxThrust", maxThrust);
//         nh_priv.getParam("VehicleMass", vehicleMass);
//         nh_priv.getParam("GravAcc", gravAcc);
//         nh_priv.getParam("HorizDrag", horizDrag);
//         nh_priv.getParam("VertDrag", vertDrag);
//         nh_priv.getParam("ParasDrag", parasDrag);
//         nh_priv.getParam("SpeedEps", speedEps);
//         nh_priv.getParam("SmoothingEps", smoothingEps);
//         nh_priv.getParam("RelCostTol", relCostTol);
//     }
// };

// class GlobalPlanner
// {
// private:
//     Config config;
//     ros::NodeHandle nh;
//     ros::Subscriber mapSub;
//     ros::Subscriber targetSub;
//     ros::Publisher bsplinePub;

//     bool mapInitialized;
//     voxel_map::VoxelMap voxelMap;
//     Visualizer visualizer;
//     std::vector<Eigen::Vector3d> startGoal;

//     nubs::NUBSTrajectory<3> nubs_traj;
//     bool has_traj;
//     double trajStamp;

//     inline void getInitialWaypointsAndTimes(const Eigen::Matrix3Xd &path,
//                                             const double &speed,
//                                             const Eigen::VectorXi &intervalNs,
//                                             Eigen::VectorXd &timeAlloc,
//                                             Eigen::MatrixXd &waypoints)
//     {
//         const int sizeM = intervalNs.size();
//         const int sizeN = intervalNs.sum();
//         waypoints.resize(sizeN + 1, 3);
//         timeAlloc.resize(sizeN);

//         waypoints.row(0) = path.col(0).transpose();
//         Eigen::Vector3d a, b, c;
//         for (int i = 0, j = 0, k = 1, l; i < sizeM; i++)
//         {
//             l = intervalNs(i);
//             a = path.col(i);
//             b = path.col(i + 1);
//             c = (b - a) / l;
//             timeAlloc.segment(j, l).setConstant(c.norm() / (speed * 0.8)); 
//             j += l;
//             for (int m = 1; m <= l; m++)
//             {
//                 waypoints.row(k++) = (a + c * m).transpose();
//             }
//         }
//     }

// public:
//     GlobalPlanner(const Config &conf, ros::NodeHandle &nh_)
//         : config(conf), nh(nh_), mapInitialized(false), visualizer(nh), has_traj(false)
//     {
//         const Eigen::Vector3i xyz((config.mapBound[1] - config.mapBound[0]) / config.voxelWidth,
//                                   (config.mapBound[3] - config.mapBound[2]) / config.voxelWidth,
//                                   (config.mapBound[5] - config.mapBound[4]) / config.voxelWidth);

//         const Eigen::Vector3d offset(config.mapBound[0], config.mapBound[2], config.mapBound[4]);
//         voxelMap = voxel_map::VoxelMap(xyz, offset, config.voxelWidth);

//         mapSub = nh.subscribe(config.mapTopic, 1, &GlobalPlanner::mapCallBack, this, ros::TransportHints().tcpNoDelay());
//         targetSub = nh.subscribe(config.targetTopic, 1, &GlobalPlanner::targetCallBack, this, ros::TransportHints().tcpNoDelay());
        
//         bsplinePub = nh.advertise<nav_msgs::Path>("/nubs_trajectory", 1);
//     }

//     inline void mapCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg)
//     {
//         if (!mapInitialized)
//         {
//             size_t cur = 0;
//             const size_t total = msg->data.size() / msg->point_step;
//             float *fdata = (float *)(&msg->data[0]);
//             for (size_t i = 0; i < total; i++)
//             {
//                 cur = msg->point_step / sizeof(float) * i;
//                 if (std::isnan(fdata[cur]) || std::isnan(fdata[cur + 1]) || std::isnan(fdata[cur + 2])) continue;
//                 voxelMap.setOccupied(Eigen::Vector3d(fdata[cur], fdata[cur + 1], fdata[cur + 2]));
//             }
//             voxelMap.dilate(std::ceil(config.dilateRadius / voxelMap.getScale()));
//             mapInitialized = true;
//         }
//     }

//     inline void plan()
//     {
//         if (startGoal.size() == 2)
//         {
//             std::vector<Eigen::Vector3d> route;
//             sfc_gen::planPath<voxel_map::VoxelMap>(startGoal[0], startGoal[1],
//                                                    voxelMap.getOrigin(), voxelMap.getCorner(),
//                                                    &voxelMap, 0.01, route);
            
//             std::vector<Eigen::MatrixX4d> hPolys, hPolys_new;
//             std::vector<Eigen::Vector3d> pc;
//             voxelMap.getSurf(pc);
            
//             sfc_gen_opt::convexCover(route, pc, voxelMap.getOrigin(), voxelMap.getCorner(), 7.0, 3.0, hPolys, hPolys_new);
//             sfc_gen_opt::shortCut(hPolys_new);

//             if (route.size() > 1)
//             {
//                 std::vector<double> opt_mesh_color = {1.0, 0.0, 0.0, 0.15};  
//                 std::vector<double> opt_edge_color = {1.0, 0.0, 0.0, 1.0};
//                 visualizer.visualizePolytope(hPolys_new, "new_firi_opt", opt_mesh_color, opt_edge_color);

//                 Eigen::Matrix3d iniState, finState;
//                 iniState << route.front(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
//                 finState << route.back(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

//                 Eigen::Matrix3Xd shortPath(3, route.size());
//                 for (size_t i = 0; i < route.size(); i++) shortPath.col(i) = route[i];
                
//                 const Eigen::Matrix3Xd deltas = shortPath.rightCols(route.size()-1) - shortPath.leftCols(route.size()-1);
//                 double lengthPerPiece = 1.0; 
//                 Eigen::VectorXi pieceIdx = (deltas.colwise().norm() / lengthPerPiece).cast<int>().transpose();
//                 pieceIdx.array() += 1;

//                 Eigen::VectorXd times_init;
//                 Eigen::MatrixXd waypoints_init;
//                 getInitialWaypointsAndTimes(shortPath, config.maxVelMag, pieceIdx, times_init, waypoints_init);

//                 double max_a = config.maxThrust - config.gravAcc;
//                 if (max_a < 1.0) max_a = 5.0;

//                 auto t1 = std::chrono::high_resolution_clock::now();
                
//                 nubs::NUBSOptimizer optimizer(3);
//                 optimizer.setup(iniState, finState, hPolys_new, config.maxVelMag, max_a);

//                 if (optimizer.optimize(times_init, waypoints_init, nubs_traj))
//                 {
//                     auto t2 = std::chrono::high_resolution_clock::now();
//                     double dt = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
//                     ROS_INFO("\033[1;32m NUBS Trajectory Optimization Succeeded in %.2f ms! \033[0m", dt);

//                     has_traj = true;
//                     trajStamp = ros::Time::now().toSec();

//                     nav_msgs::Path path_msg;
//                     path_msg.header.stamp = ros::Time::now();
//                     path_msg.header.frame_id = "world"; 

//                     double total_time = nubs_traj.getTotalDuration();
//                     for (double t = 0.0; t <= total_time; t += 0.05) {
//                         Eigen::Vector3d pt = nubs_traj.evaluate(t, 0); 
//                         geometry_msgs::PoseStamped pose;
//                         pose.pose.position.x = pt.x();
//                         pose.pose.position.y = pt.y();
//                         pose.pose.position.z = pt.z();
//                         path_msg.poses.push_back(pose);
//                     }
//                     bsplinePub.publish(path_msg);
//                 }
//                 else
//                 {
//                     has_traj = false;
//                     ROS_ERROR("\033[1;31m NUBS Trajectory Optimization Failed. \033[0m");
//                 }
//             }
//         }
//     }

//     inline void targetCallBack(const geometry_msgs::PoseStamped::ConstPtr &msg)
//     {
//         if (mapInitialized)
//         {
//             if (startGoal.size() >= 2) startGoal.clear();
//             const double zGoal = config.mapBound[4] + config.dilateRadius +
//                                  fabs(msg->pose.orientation.z) * (config.mapBound[5] - config.mapBound[4] - 2 * config.dilateRadius);
//             const Eigen::Vector3d goal(msg->pose.position.x, msg->pose.position.y, zGoal);
            
//             if (voxelMap.query(goal) == 0) {
//                 visualizer.visualizeStartGoal(goal, 0.5, startGoal.size());
//                 startGoal.emplace_back(goal);
//             } else {
//                 ROS_WARN("Infeasible Position Selected !!!");
//             }
//             plan();
//         }
//     }

//     inline void process()
//     {
//         Eigen::VectorXd physicalParams(6);
//         physicalParams(0) = config.vehicleMass;
//         physicalParams(1) = config.gravAcc;
//         physicalParams(2) = config.horizDrag;
//         physicalParams(3) = config.vertDrag;
//         physicalParams(4) = config.parasDrag;
//         physicalParams(5) = config.speedEps;

//         flatness::FlatnessMap flatmap;
//         flatmap.reset(physicalParams(0), physicalParams(1), physicalParams(2),
//                       physicalParams(3), physicalParams(4), physicalParams(5));

//         if (has_traj)
//         {
//             const double delta = ros::Time::now().toSec() - trajStamp;
//             if (delta > 0.0 && delta < nubs_traj.getTotalDuration())
//             {
//                 double thr; Eigen::Vector4d quat; Eigen::Vector3d omg;

//                 Eigen::Vector3d pos = nubs_traj.evaluate(delta, 0);
//                 Eigen::Vector3d vel = nubs_traj.evaluate(delta, 1);
//                 Eigen::Vector3d acc = nubs_traj.evaluate(delta, 2);
//                 Eigen::Vector3d jer = nubs_traj.evaluate(delta, 3);

//                 flatmap.forward(vel, acc, jer, 0.0, 0.0, thr, quat, omg);

//                 double tiltangle = acos(1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2)));
                
//                 std_msgs::Float64 speedMsg, thrMsg, tiltMsg, bdrMsg;
//                 speedMsg.data = vel.norm();
//                 thrMsg.data = thr;
//                 tiltMsg.data = tiltangle;
//                 bdrMsg.data = omg.norm();
                
//                 visualizer.speedPub.publish(speedMsg);
//                 visualizer.thrPub.publish(thrMsg);
//                 visualizer.tiltPub.publish(tiltMsg);
//                 visualizer.bdrPub.publish(bdrMsg);

//                 visualizer.visualizeSphere(pos, config.dilateRadius);
//             }
//         }
//     }
// };

// int main(int argc, char **argv)
// {
//     ros::init(argc, argv, "global_planning_node");
//     ros::NodeHandle nh_;
//     GlobalPlanner global_planner(Config(ros::NodeHandle("~")), nh_);

//     ros::Rate lr(1000);
//     while (ros::ok())
//     {
//         global_planner.process();
//         ros::spinOnce();
//         lr.sleep();
//     }
//     return 0;
// }

// #include "misc/visualizer.hpp"
// #include "gcopter/flatness.hpp"
// #include "gcopter/voxel_map.hpp"
// #include "gcopter/sfc_gen.hpp"
// #include "gcopter/sfc_gen_opt.hpp"
// #include "gcopter/nubs_optimizer.hpp" 

// #include <ros/ros.h>
// #include <ros/console.h>
// #include <geometry_msgs/PoseStamped.h>
// #include <sensor_msgs/PointCloud2.h>
// #include <nav_msgs/Path.h> 

// #include <cmath>
// #include <iostream>
// #include <string>
// #include <vector>
// #include <chrono>

// struct Config
// {
//     std::string mapTopic, targetTopic;
//     double dilateRadius, voxelWidth, timeoutRRT;
//     std::vector<double> mapBound;
//     double maxVelMag, maxBdrMag, maxTiltAngle, minThrust, maxThrust;
//     double vehicleMass, gravAcc, horizDrag, vertDrag, parasDrag;
//     double speedEps, smoothingEps;

//     Config(const ros::NodeHandle &nh_priv)
//     {
//         nh_priv.getParam("MapTopic", mapTopic);
//         nh_priv.getParam("TargetTopic", targetTopic);
//         nh_priv.getParam("DilateRadius", dilateRadius);
//         nh_priv.getParam("VoxelWidth", voxelWidth);
//         nh_priv.getParam("MapBound", mapBound);
//         nh_priv.getParam("MaxVelMag", maxVelMag);
//         nh_priv.getParam("MaxBdrMag", maxBdrMag);
//         nh_priv.getParam("MaxTiltAngle", maxTiltAngle);
//         nh_priv.getParam("MinThrust", minThrust);
//         nh_priv.getParam("MaxThrust", maxThrust);
//         nh_priv.getParam("VehicleMass", vehicleMass);
//         nh_priv.getParam("GravAcc", gravAcc);
//         nh_priv.getParam("HorizDrag", horizDrag);
//         nh_priv.getParam("VertDrag", vertDrag);
//         nh_priv.getParam("ParasDrag", parasDrag);
//         nh_priv.getParam("SpeedEps", speedEps);
//         nh_priv.getParam("SmoothingEps", smoothingEps);
//     }
// };

// class GlobalPlanner
// {
// private:
//     Config config;
//     ros::NodeHandle nh;
//     ros::Subscriber mapSub, targetSub;
//     ros::Publisher bsplinePub;

//     bool mapInitialized;
//     voxel_map::VoxelMap voxelMap;
//     Visualizer visualizer;
//     std::vector<Eigen::Vector3d> startGoal;

//     nubs::NUBSTrajectory<3> nubs_traj;
//     bool has_traj;
//     double trajStamp;

//     inline bool processCorridor(const std::vector<Eigen::MatrixX4d> &hPs, std::vector<Eigen::Matrix3Xd> &vPs)
//     {
//         int sizeCorridor = hPs.size() - 1;
//         vPs.clear();
//         vPs.reserve(2 * sizeCorridor + 1);
//         int nv;
//         Eigen::MatrixX4d curIH;
//         Eigen::Matrix3Xd curIV, curIOB;
//         for (int i = 0; i < sizeCorridor; i++) {
//             if (!geo_utils::enumerateVs(hPs[i], curIV)) return false;
//             nv = curIV.cols(); curIOB.resize(3, nv);
//             curIOB.col(0) = curIV.col(0); curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
//             vPs.push_back(curIOB);

//             curIH.resize(hPs[i].rows() + hPs[i + 1].rows(), 4);
//             curIH.topRows(hPs[i].rows()) = hPs[i]; curIH.bottomRows(hPs[i + 1].rows()) = hPs[i + 1];
//             if (!geo_utils::enumerateVs(curIH, curIV)) return false;
//             nv = curIV.cols(); curIOB.resize(3, nv);
//             curIOB.col(0) = curIV.col(0); curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
//             vPs.push_back(curIOB);
//         }
//         if (!geo_utils::enumerateVs(hPs.back(), curIV)) return false;
//         nv = curIV.cols(); curIOB.resize(3, nv);
//         curIOB.col(0) = curIV.col(0); curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
//         vPs.push_back(curIOB);
//         return true;
//     }

// public:
//     GlobalPlanner(const Config &conf, ros::NodeHandle &nh_)
//         : config(conf), nh(nh_), mapInitialized(false), visualizer(nh), has_traj(false)
//     {
//         const Eigen::Vector3i xyz((config.mapBound[1] - config.mapBound[0]) / config.voxelWidth,
//                                   (config.mapBound[3] - config.mapBound[2]) / config.voxelWidth,
//                                   (config.mapBound[5] - config.mapBound[4]) / config.voxelWidth);
//         const Eigen::Vector3d offset(config.mapBound[0], config.mapBound[2], config.mapBound[4]);
//         voxelMap = voxel_map::VoxelMap(xyz, offset, config.voxelWidth);

//         mapSub = nh.subscribe(config.mapTopic, 1, &GlobalPlanner::mapCallBack, this, ros::TransportHints().tcpNoDelay());
//         targetSub = nh.subscribe(config.targetTopic, 1, &GlobalPlanner::targetCallBack, this, ros::TransportHints().tcpNoDelay());
//         bsplinePub = nh.advertise<nav_msgs::Path>("/nubs_trajectory", 1);
//     }

//     inline void mapCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg)
//     {
//         if (!mapInitialized) {
//             size_t cur = 0; const size_t total = msg->data.size() / msg->point_step; float *fdata = (float *)(&msg->data[0]);
//             for (size_t i = 0; i < total; i++) {
//                 cur = msg->point_step / sizeof(float) * i;
//                 if (std::isnan(fdata[cur]) || std::isnan(fdata[cur + 1]) || std::isnan(fdata[cur + 2])) continue;
//                 voxelMap.setOccupied(Eigen::Vector3d(fdata[cur], fdata[cur + 1], fdata[cur + 2]));
//             }
//             voxelMap.dilate(std::ceil(config.dilateRadius / voxelMap.getScale()));
//             mapInitialized = true;
//         }
//     }

//     inline void plan()
//     {
//         if (startGoal.size() == 2)
//         {
//             std::vector<Eigen::Vector3d> route;
//             sfc_gen::planPath<voxel_map::VoxelMap>(startGoal[0], startGoal[1], voxelMap.getOrigin(), voxelMap.getCorner(), &voxelMap, 0.01, route);
            
//             std::vector<Eigen::MatrixX4d> hPolys, hPolys_new;
//             std::vector<Eigen::Matrix3Xd> vPolys;
//             std::vector<Eigen::Vector3d> pc;
//             voxelMap.getSurf(pc);
            
//             sfc_gen_opt::convexCover(route, pc, voxelMap.getOrigin(), voxelMap.getCorner(), 7.0, 3.0, hPolys, hPolys_new);
//             sfc_gen_opt::shortCut(hPolys_new);

//             if (route.size() > 1 && processCorridor(hPolys_new, vPolys))
//             {
//                 std::vector<double> opt_mesh_color = {1.0, 0.0, 0.0, 0.15};  
//                 std::vector<double> opt_edge_color = {1.0, 0.0, 0.0, 1.0};
//                 visualizer.visualizePolytope(hPolys_new, "new_firi_opt", opt_mesh_color, opt_edge_color);

//                 // ★ 显式赋予确定的 3x3 结构，彻底杜绝逗号赋值引起的隐患
//                 Eigen::Matrix3d iniState = Eigen::Matrix3d::Zero();
//                 Eigen::Matrix3d finState = Eigen::Matrix3d::Zero();
//                 iniState.col(0) = route.front();
//                 finState.col(0) = route.back();

//                 int polyN = hPolys_new.size();
//                 int pieceN = polyN;
//                 Eigen::VectorXi vPolyIdx(pieceN - 1), hPolyIdx(pieceN);
//                 for (int i = 0; i < polyN; i++) {
//                     if (i < polyN - 1) vPolyIdx(i) = 2 * i + 1; 
//                     hPolyIdx(i) = i;
//                 }

//                 Eigen::VectorXd times_init(pieceN);
//                 Eigen::MatrixXd waypoints_init(pieceN - 1, 3);
//                 for(int i = 0; i < pieceN; i++) {
//                     times_init(i) = (route[i+1] - route[i]).norm() / config.maxVelMag;
//                     if(i < pieceN - 1) waypoints_init.row(i) = route[i+1].transpose();
//                 }

//                 double max_a = config.maxThrust - config.gravAcc; if (max_a < 1.0) max_a = 5.0;

//                 auto t1 = std::chrono::high_resolution_clock::now();
                
//                 nubs::NUBSOptimizer optimizer(3);
//                 optimizer.setup(iniState, finState, vPolys, hPolys_new, vPolyIdx, hPolyIdx, config.maxVelMag, max_a);

//                 if (optimizer.optimize(times_init, waypoints_init, nubs_traj))
//                 {
//                     auto t2 = std::chrono::high_resolution_clock::now();
//                     double dt = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
//                     ROS_INFO("\033[1;32m NUBS-MINCO Optimization Succeeded in %.2f ms! \033[0m", dt);

//                     has_traj = true;
//                     trajStamp = ros::Time::now().toSec();

//                     nav_msgs::Path path_msg;
//                     path_msg.header.stamp = ros::Time::now();
//                     path_msg.header.frame_id = "world"; 
//                     double total_time = nubs_traj.getTotalDuration();
//                     for (double t = 0.0; t <= total_time; t += 0.05) {
//                         Eigen::Vector3d pt = nubs_traj.evaluate(t, 0); 
//                         geometry_msgs::PoseStamped pose;
//                         pose.pose.position.x = pt.x(); pose.pose.position.y = pt.y(); pose.pose.position.z = pt.z();
//                         path_msg.poses.push_back(pose);
//                     }
//                     bsplinePub.publish(path_msg);
//                 }
//                 else { has_traj = false; ROS_ERROR("\033[1;31m NUBS Optimization Failed. \033[0m"); }
//             }
//         }
//     }

//     inline void targetCallBack(const geometry_msgs::PoseStamped::ConstPtr &msg)
//     {
//         if (mapInitialized) {
//             if (startGoal.size() >= 2) startGoal.clear();
//             const double zGoal = config.mapBound[4] + config.dilateRadius + fabs(msg->pose.orientation.z) * (config.mapBound[5] - config.mapBound[4] - 2 * config.dilateRadius);
//             const Eigen::Vector3d goal(msg->pose.position.x, msg->pose.position.y, zGoal);
//             if (voxelMap.query(goal) == 0) {
//                 visualizer.visualizeStartGoal(goal, 0.5, startGoal.size());
//                 startGoal.emplace_back(goal);
//             }
//             plan();
//         }
//     }

//     inline void process()
//     {
//         Eigen::VectorXd physicalParams(6);
//         physicalParams << config.vehicleMass, config.gravAcc, config.horizDrag, config.vertDrag, config.parasDrag, config.speedEps;
//         flatness::FlatnessMap flatmap;
//         flatmap.reset(physicalParams(0), physicalParams(1), physicalParams(2), physicalParams(3), physicalParams(4), physicalParams(5));

//         if (has_traj) {
//             const double delta = ros::Time::now().toSec() - trajStamp;
//             if (delta > 0.0 && delta < nubs_traj.getTotalDuration()) {
//                 double thr; Eigen::Vector4d quat; Eigen::Vector3d omg;
//                 Eigen::Vector3d pos = nubs_traj.evaluate(delta, 0), vel = nubs_traj.evaluate(delta, 1);
//                 Eigen::Vector3d acc = nubs_traj.evaluate(delta, 2), jer = nubs_traj.evaluate(delta, 3);
//                 flatmap.forward(vel, acc, jer, 0.0, 0.0, thr, quat, omg);

//                 double tiltangle = acos(1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2)));
//                 std_msgs::Float64 speedMsg, thrMsg, tiltMsg, bdrMsg;
//                 speedMsg.data = vel.norm(); thrMsg.data = thr; tiltMsg.data = tiltangle; bdrMsg.data = omg.norm();
                
//                 visualizer.speedPub.publish(speedMsg); visualizer.thrPub.publish(thrMsg);
//                 visualizer.tiltPub.publish(tiltMsg); visualizer.bdrPub.publish(bdrMsg);
//                 visualizer.visualizeSphere(pos, config.dilateRadius);
//             }
//         }
//     }
// };

// int main(int argc, char **argv) {
//     ros::init(argc, argv, "global_planning_node"); ros::NodeHandle nh_;
//     GlobalPlanner global_planner(Config(ros::NodeHandle("~")), nh_);
//     ros::Rate lr(1000);
//     while (ros::ok()) { global_planner.process(); ros::spinOnce(); lr.sleep(); }
//     return 0;
// }
