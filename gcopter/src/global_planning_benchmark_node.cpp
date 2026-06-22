#include "gcopter/benchmark/benchmark_timer.hpp"
#include "gcopter/benchmark/benchmark_types.hpp"
#include "gcopter/benchmark/csv_writer.hpp"
#include "gcopter/benchmark/planning_benchmark.hpp"
#include "gcopter/benchmark/seed_sampler.hpp"
#include "gcopter/gcopter.hpp"
#include "gcopter/sfc_gen.hpp"
#include "gcopter/sfc_gen_benchmark.hpp"
#include "gcopter/trajectory.hpp"
#include "gcopter/voxel_map.hpp"

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <Eigen/Eigen>
#include <ompl/util/RandomNumbers.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace
{
std::string shellQuote(const std::string &value)
{
    std::string out = "'";
    for (const char c : value)
    {
        if (c == '\'')
        {
            out += "'\\''";
        }
        else
        {
            out += c;
        }
    }
    out += "'";
    return out;
}

void ensureDirectory(const std::string &path)
{
    const int rc = std::system(("mkdir -p " + shellQuote(path)).c_str());
    if (rc != 0)
    {
        ROS_WARN_STREAM("mkdir failed for " << path << ", rc=" << rc);
    }
}

std::vector<double> getDoubleVectorParam(const ros::NodeHandle &nh,
                                         const std::string &name,
                                         const std::vector<double> &fallback)
{
    std::vector<double> values;
    return nh.getParam(name, values) ? values : fallback;
}

std::string defaultRunId(const std::string &density,
                         const int map_seed,
                         const std::uint64_t master_seed)
{
    std::ostringstream oss;
    oss << "global_planning_" << density << "_map" << map_seed << "_master" << master_seed;
    return oss.str();
}

struct BenchmarkConfig
{
    std::string map_topic = "/voxel_map";
    std::string output_dir = "/tmp/firi_global_planning_benchmark";
    std::string run_id;
    std::string density = "medium";
    int map_seed = 1024;
    std::uint64_t master_seed = 20260622ULL;
    double dilate_radius = 0.5;
    double voxel_width = 0.25;
    std::vector<double> map_bound = {-25.0, 25.0, -25.0, 25.0, 0.0, 5.0};
    double timeout_rrt = 0.02;
    double local_range = 3.0;
    double corridor_progress = 7.0;
    int planning_case_count = 30;
    int max_sampling_attempts = 10000;
    double planning_min_distance = 8.0;
    double planning_max_distance = 45.0;
    double planning_sample_margin = 0.5;
    firi_benchmark::TrajectoryOptimizerConfig trajectory_config;

    explicit BenchmarkConfig(const ros::NodeHandle &nh)
    {
        nh.param("MapTopic", map_topic, map_topic);
        nh.param("OutputDir", output_dir, output_dir);
        nh.param("RunId", run_id, run_id);
        nh.param("DensityLabel", density, density);
        nh.param("MapSeed", map_seed, map_seed);
        int master_seed_i = static_cast<int>(master_seed);
        nh.param("MasterSeed", master_seed_i, master_seed_i);
        master_seed = static_cast<std::uint64_t>(master_seed_i);
        nh.param("DilateRadius", dilate_radius, dilate_radius);
        nh.param("VoxelWidth", voxel_width, voxel_width);
        nh.getParam("MapBound", map_bound);

        double map_size_x = map_bound[1] - map_bound[0];
        double map_size_y = map_bound[3] - map_bound[2];
        double map_size_z = map_bound[5] - map_bound[4];
        double map_x_origin = map_bound[0];
        double map_y_origin = map_bound[2];
        double map_z_origin = map_bound[4];
        const bool has_explicit_map_box =
            nh.getParam("MapSizeX", map_size_x) |
            nh.getParam("MapSizeY", map_size_y) |
            nh.getParam("MapSizeZ", map_size_z) |
            nh.getParam("MapOriginX", map_x_origin) |
            nh.getParam("MapOriginY", map_y_origin) |
            nh.getParam("MapOriginZ", map_z_origin);
        if (has_explicit_map_box)
        {
            map_bound = {map_x_origin,
                         map_x_origin + map_size_x,
                         map_y_origin,
                         map_y_origin + map_size_y,
                         map_z_origin,
                         map_z_origin + map_size_z};
        }

        nh.param("TimeoutRRT", timeout_rrt, timeout_rrt);
        nh.param("LocalRange", local_range, local_range);
        nh.param("CorridorProgress", corridor_progress, corridor_progress);
        nh.param("PlanningCaseCount", planning_case_count, planning_case_count);
        nh.param("MaxSamplingAttempts", max_sampling_attempts, max_sampling_attempts);
        nh.param("PlanningMinDistance", planning_min_distance, planning_min_distance);
        nh.param("PlanningMaxDistance", planning_max_distance, planning_max_distance);
        nh.param("PlanningSampleMargin", planning_sample_margin, planning_sample_margin);

        nh.param("MaxVelMag", trajectory_config.max_vel_mag, trajectory_config.max_vel_mag);
        nh.param("MaxBdrMag", trajectory_config.max_bdr_mag, trajectory_config.max_bdr_mag);
        nh.param("MaxTiltAngle", trajectory_config.max_tilt_angle, trajectory_config.max_tilt_angle);
        nh.param("MinThrust", trajectory_config.min_thrust, trajectory_config.min_thrust);
        nh.param("MaxThrust", trajectory_config.max_thrust, trajectory_config.max_thrust);
        nh.param("VehicleMass", trajectory_config.vehicle_mass, trajectory_config.vehicle_mass);
        nh.param("GravAcc", trajectory_config.grav_acc, trajectory_config.grav_acc);
        nh.param("HorizDrag", trajectory_config.horiz_drag, trajectory_config.horiz_drag);
        nh.param("VertDrag", trajectory_config.vert_drag, trajectory_config.vert_drag);
        nh.param("ParasDrag", trajectory_config.paras_drag, trajectory_config.paras_drag);
        nh.param("SpeedEps", trajectory_config.speed_eps, trajectory_config.speed_eps);
        nh.param("WeightT", trajectory_config.weight_t, trajectory_config.weight_t);
        const std::vector<double> chi =
            getDoubleVectorParam(nh, "ChiVec", {1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e5});
        trajectory_config.chi_vec.resize(5);
        for (int i = 0; i < 5; ++i)
        {
            trajectory_config.chi_vec(i) = i < static_cast<int>(chi.size()) ? chi[i] : 1.0e4;
        }
        nh.param("SmoothingEps", trajectory_config.smoothing_eps, trajectory_config.smoothing_eps);
        nh.param("IntegralIntervs", trajectory_config.integral_intervals, trajectory_config.integral_intervals);
        nh.param("RelCostTol", trajectory_config.rel_cost_tol, trajectory_config.rel_cost_tol);
        nh.param("TrajectoryValidationDt", trajectory_config.validation_dt, trajectory_config.validation_dt);

        if (run_id.empty())
        {
            run_id = defaultRunId(density, map_seed, master_seed);
        }
    }

    Eigen::Vector3d mapLow() const
    {
        return Eigen::Vector3d(map_bound[0], map_bound[2], map_bound[4]);
    }

    Eigen::Vector3d mapHigh() const
    {
        return Eigen::Vector3d(map_bound[1], map_bound[3], map_bound[5]);
    }

    std::string mapId() const
    {
        std::ostringstream oss;
        oss << density << "_seed_" << map_seed;
        return oss.str();
    }
};

struct MethodCorridor
{
    std::string method;
    std::string solver;
    std::vector<Eigen::MatrixX4d> hpolys;
    double solver_ms = 0.0;
    double shortcut_ms = 0.0;
    double shared_ms = 0.0;
    double volume_sum = 0.0;
    int faces_total = 0;

    double corridorMs() const
    {
        return shared_ms + solver_ms + shortcut_ms;
    }
};

struct TrajectoryResult
{
    bool setup_success = false;
    bool optimize_success = false;
    bool collision_free = false;
    int sampled_collision_count = 0;
    double setup_ms = 0.0;
    double optimize_ms = 0.0;
    double cost = INFINITY;
    double duration = 0.0;
    int pieces = 0;
    std::string failure_reason;
};

const std::vector<std::string> &trialHeader()
{
    static const std::vector<std::string> header = {
        "run_id", "map_id", "density", "planning_case_id", "method", "solver",
        "start_x", "start_y", "start_z", "goal_x", "goal_y", "goal_z",
        "path_success", "path_search_ms", "surface_extract_ms", "route_hash",
        "route_points", "route_length", "corridor_success", "regions",
        "corridor_shared_ms", "corridor_solver_ms", "corridor_shortcut_ms",
        "corridor_total_ms", "corridor_volume_sum", "corridor_faces_total",
        "trajectory_setup_success", "trajectory_setup_ms",
        "trajectory_optimize_success", "trajectory_optimize_ms", "trajectory_cost",
        "trajectory_duration", "trajectory_pieces", "sampled_collision_count",
        "trajectory_collision_free", "end_to_end_ms", "global_planning_success",
        "success", "failure_reason"};
    return header;
}

double totalSolverMs(const sfc_gen_benchmark::BenchmarkResult &result)
{
    return result.firi.total_time_ms +
           result.firi_socp.total_time_ms +
           result.firi_opt.total_time_ms +
           result.firi_nd.total_time_ms;
}

void fillCorridorMetrics(MethodCorridor &corridor)
{
    corridor.volume_sum = 0.0;
    corridor.faces_total = 0;
    for (const Eigen::MatrixX4d &hpoly : corridor.hpolys)
    {
        const firi_benchmark::PolytopeMetrics metrics = firi_benchmark::measurePolytope(hpoly);
        corridor.volume_sum += metrics.volume;
        corridor.faces_total += static_cast<int>(metrics.face_count);
    }
}

firi_benchmark::TrajectoryTrialPod toGlobalPlanningPod(const TrajectoryResult &result)
{
    firi_benchmark::TrajectoryTrialPod pod;
    pod.setup_success = result.setup_success ? 1 : 0;
    pod.optimize_success = result.optimize_success ? 1 : 0;
    pod.collision_free = result.collision_free ? 1 : 0;
    pod.sampled_collision_count = result.sampled_collision_count;
    pod.setup_ms = result.setup_ms;
    pod.optimize_ms = result.optimize_ms;
    pod.cost = result.cost;
    pod.duration = result.duration;
    pod.pieces = result.pieces;
    firi_benchmark::copyFailureReason(result.failure_reason, pod.failure_reason, sizeof(pod.failure_reason));
    return pod;
}

TrajectoryResult fromGlobalPlanningPod(const firi_benchmark::TrajectoryTrialPod &pod)
{
    TrajectoryResult result;
    result.setup_success = pod.setup_success != 0;
    result.optimize_success = pod.optimize_success != 0;
    result.collision_free = pod.collision_free != 0;
    result.sampled_collision_count = pod.sampled_collision_count;
    result.setup_ms = pod.setup_ms;
    result.optimize_ms = pod.optimize_ms;
    result.cost = pod.cost;
    result.duration = pod.duration;
    result.pieces = pod.pieces;
    result.failure_reason = std::string(pod.failure_reason);
    return result;
}

TrajectoryResult optimizeGlobalPlanningCorridorUnsafe(const Eigen::Vector3d &start,
                                                      const Eigen::Vector3d &goal,
                                                      const std::vector<Eigen::MatrixX4d> &hpolys,
                                                      const firi_benchmark::TrajectoryOptimizerConfig &config,
                                                      const voxel_map::VoxelMap &map)
{
    TrajectoryResult result;
    if (hpolys.empty())
    {
        result.failure_reason = "empty_corridor";
        return result;
    }

    Eigen::Matrix3d ini_state;
    Eigen::Matrix3d fin_state;
    ini_state << start, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
    fin_state << goal, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

    Eigen::VectorXd magnitude_bounds(5);
    Eigen::VectorXd penalty_weights(5);
    Eigen::VectorXd physical_params(6);
    magnitude_bounds(0) = config.max_vel_mag;
    magnitude_bounds(1) = config.max_bdr_mag;
    magnitude_bounds(2) = config.max_tilt_angle;
    magnitude_bounds(3) = config.min_thrust;
    magnitude_bounds(4) = config.max_thrust;
    penalty_weights = config.chi_vec;
    physical_params(0) = config.vehicle_mass;
    physical_params(1) = config.grav_acc;
    physical_params(2) = config.horiz_drag;
    physical_params(3) = config.vert_drag;
    physical_params(4) = config.paras_drag;
    physical_params(5) = config.speed_eps;

    gcopter::GCOPTER_PolytopeSFC optimizer;
    Trajectory<5> trajectory;

    firi_benchmark::SteadyTimer setup_timer;
    result.setup_success = optimizer.setup(config.weight_t,
                                           ini_state,
                                           fin_state,
                                           hpolys,
                                           INFINITY,
                                           config.smoothing_eps,
                                           config.integral_intervals,
                                           magnitude_bounds,
                                           penalty_weights,
                                           physical_params);
    result.setup_ms = setup_timer.elapsedMs();
    if (!result.setup_success)
    {
        result.failure_reason = "trajectory_setup_failed";
        return result;
    }

    firi_benchmark::SteadyTimer optimize_timer;
    result.cost = optimizer.optimize(trajectory, config.rel_cost_tol);
    result.optimize_ms = optimize_timer.elapsedMs();
    result.optimize_success = std::isfinite(result.cost);
    if (!result.optimize_success)
    {
        result.failure_reason = "trajectory_optimize_failed";
        return result;
    }

    result.pieces = trajectory.getPieceNum();
    result.duration = result.pieces > 0 ? trajectory.getTotalDuration() : 0.0;
    if (result.pieces <= 0)
    {
        result.failure_reason = "trajectory_empty";
        return result;
    }

    result.sampled_collision_count =
        firi_benchmark::countTrajectoryCollisions(trajectory, map, config.validation_dt);
    result.collision_free = result.sampled_collision_count == 0;
    if (!result.collision_free)
    {
        result.failure_reason = "trajectory_collision";
    }
    return result;
}

TrajectoryResult optimizeGlobalPlanningCorridor(const Eigen::Vector3d &start,
                                                const Eigen::Vector3d &goal,
                                                const std::vector<Eigen::MatrixX4d> &hpolys,
                                                const firi_benchmark::TrajectoryOptimizerConfig &config,
                                                const voxel_map::VoxelMap &map)
{
    int pipe_fd[2];
    if (pipe(pipe_fd) != 0)
    {
        TrajectoryResult result;
        result.failure_reason = "trajectory_pipe_failed";
        return result;
    }

    const pid_t pid = fork();
    if (pid < 0)
    {
        close(pipe_fd[0]);
        close(pipe_fd[1]);
        TrajectoryResult result;
        result.failure_reason = "trajectory_fork_failed";
        return result;
    }

    if (pid == 0)
    {
        close(pipe_fd[0]);
        const firi_benchmark::TrajectoryTrialPod pod =
            toGlobalPlanningPod(optimizeGlobalPlanningCorridorUnsafe(start, goal, hpolys, config, map));
        const char *bytes = reinterpret_cast<const char *>(&pod);
        std::size_t remaining = sizeof(pod);
        while (remaining > 0)
        {
            const ssize_t written = write(pipe_fd[1], bytes, remaining);
            if (written <= 0)
            {
                break;
            }
            bytes += written;
            remaining -= static_cast<std::size_t>(written);
        }
        close(pipe_fd[1]);
        _exit(0);
    }

    close(pipe_fd[1]);
    firi_benchmark::TrajectoryTrialPod pod;
    char *bytes = reinterpret_cast<char *>(&pod);
    std::size_t remaining = sizeof(pod);
    bool complete_read = true;
    while (remaining > 0)
    {
        const ssize_t n = read(pipe_fd[0], bytes, remaining);
        if (n == 0)
        {
            complete_read = false;
            break;
        }
        if (n < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            complete_read = false;
            break;
        }
        bytes += n;
        remaining -= static_cast<std::size_t>(n);
    }
    close(pipe_fd[0]);

    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR)
    {
    }

    if (!complete_read || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
    {
        TrajectoryResult result;
        if (WIFSIGNALED(status))
        {
            result.failure_reason =
                "trajectory_optimizer_crashed_signal_" + std::to_string(WTERMSIG(status));
        }
        else
        {
            result.failure_reason = "trajectory_optimizer_failed_without_result";
        }
        return result;
    }

    return fromGlobalPlanningPod(pod);
}
}

class GlobalPlanningBenchmarkNode
{
public:
    GlobalPlanningBenchmarkNode(const BenchmarkConfig &config, ros::NodeHandle &nh)
        : config_(config), nh_(nh)
    {
        const Eigen::Vector3i xyz((config_.map_bound[1] - config_.map_bound[0]) / config_.voxel_width,
                                  (config_.map_bound[3] - config_.map_bound[2]) / config_.voxel_width,
                                  (config_.map_bound[5] - config_.map_bound[4]) / config_.voxel_width);
        voxel_map_ = voxel_map::VoxelMap(xyz, config_.mapLow(), config_.voxel_width);
        map_sub_ = nh_.subscribe(config_.map_topic, 1, &GlobalPlanningBenchmarkNode::mapCallback, this,
                                 ros::TransportHints().tcpNoDelay());
    }

    bool ready() const
    {
        return map_initialized_;
    }

    bool run()
    {
        ensureDirectory(config_.output_dir);
        firi_benchmark::CsvWriter writer(config_.output_dir + "/global_planning_trials.csv",
                                         trialHeader());
        firi_benchmark::CsvWriter manifest_writer(config_.output_dir + "/global_planning_cases.csv",
                                                  {"run_id", "map_id", "density", "case_id",
                                                   "start_x", "start_y", "start_z",
                                                   "goal_x", "goal_y", "goal_z",
                                                   "deterministic_seed"});

        std::vector<firi_benchmark::PlanningCase> cases;
        generatePlanningCases(cases, manifest_writer);
        ROS_INFO_STREAM("Running global planning benchmark: density=" << config_.density
                        << ", cases=" << cases.size()
                        << ", output_dir=" << config_.output_dir);

        for (const firi_benchmark::PlanningCase &planning_case : cases)
        {
            runPlanningCase(planning_case, writer);
        }
        writeMetadata(cases.size());
        return true;
    }

private:
    BenchmarkConfig config_;
    ros::NodeHandle nh_;
    ros::Subscriber map_sub_;
    voxel_map::VoxelMap voxel_map_;
    bool map_initialized_ = false;
    double map_preprocess_ms_ = 0.0;

    void mapCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        if (map_initialized_)
        {
            return;
        }
        firi_benchmark::SteadyTimer timer;
        const size_t total = msg->data.size() / msg->point_step;
        const float *fdata = reinterpret_cast<const float *>(&msg->data[0]);
        for (size_t i = 0; i < total; ++i)
        {
            const size_t cur = msg->point_step / sizeof(float) * i;
            if (std::isnan(fdata[cur + 0]) || std::isinf(fdata[cur + 0]) ||
                std::isnan(fdata[cur + 1]) || std::isinf(fdata[cur + 1]) ||
                std::isnan(fdata[cur + 2]) || std::isinf(fdata[cur + 2]))
            {
                continue;
            }
            voxel_map_.setOccupied(Eigen::Vector3d(fdata[cur + 0],
                                                   fdata[cur + 1],
                                                   fdata[cur + 2]));
        }
        voxel_map_.dilate(std::ceil(config_.dilate_radius / voxel_map_.getScale()));
        map_preprocess_ms_ = timer.elapsedMs();
        map_initialized_ = true;
        ROS_INFO_STREAM("Map initialized: preprocess_ms=" << map_preprocess_ms_);
    }

    void generatePlanningCases(std::vector<firi_benchmark::PlanningCase> &cases,
                               firi_benchmark::CsvWriter &manifest_writer) const
    {
        const Eigen::Vector3d map_low = config_.mapLow();
        const Eigen::Vector3d map_high = config_.mapHigh();
        for (int i = 0; i < config_.planning_case_count; ++i)
        {
            firi_benchmark::PlanningCase planning_case;
            const std::uint64_t seed = firi_benchmark::mixSeed(config_.master_seed, 300000ULL + i);
            if (!firi_benchmark::samplePlanningCase(voxel_map_,
                                                    map_low,
                                                    map_high,
                                                    config_.planning_sample_margin,
                                                    config_.planning_min_distance,
                                                    config_.planning_max_distance,
                                                    config_.max_sampling_attempts,
                                                    firi_benchmark::caseId("planning", i),
                                                    seed,
                                                    planning_case))
            {
                ROS_ERROR_STREAM("Failed to sample planning case " << i);
                continue;
            }
            cases.push_back(planning_case);
            manifest_writer.writeRow(config_.run_id,
                                     config_.mapId(),
                                     config_.density,
                                     planning_case.case_id,
                                     planning_case.start.x(),
                                     planning_case.start.y(),
                                     planning_case.start.z(),
                                     planning_case.goal.x(),
                                     planning_case.goal.y(),
                                     planning_case.goal.z(),
                                     planning_case.deterministic_seed);
        }
    }

    std::vector<MethodCorridor> makeCorridors(const std::vector<Eigen::Vector3d> &route,
                                              const std::vector<Eigen::Vector3d> &surface_points,
                                              double &corridor_wall_ms) const
    {
        firi_benchmark::SteadyTimer corridor_timer;
        const sfc_gen_benchmark::BenchmarkResult benchmark =
            sfc_gen_benchmark::convexCover(route,
                                           surface_points,
                                           voxel_map_.getOrigin(),
                                           voxel_map_.getCorner(),
                                           config_.corridor_progress,
                                           config_.local_range);
        corridor_wall_ms = corridor_timer.elapsedMs();

        const double shared_ms =
            std::max(0.0, corridor_wall_ms - totalSolverMs(benchmark));

        MethodCorridor baseline;
        baseline.method = "baseline_firi";
        baseline.solver = benchmark.firi.name;
        baseline.hpolys = benchmark.firi.hpolys;
        baseline.solver_ms = benchmark.firi.total_time_ms;
        baseline.shared_ms = shared_ms;
        firi_benchmark::SteadyTimer baseline_shortcut_timer;
        sfc_gen_benchmark::shortCut(baseline.hpolys);
        baseline.shortcut_ms = baseline_shortcut_timer.elapsedMs();
        fillCorridorMetrics(baseline);

        MethodCorridor hom;
        hom.method = "hom_mvie";
        hom.solver = benchmark.firi_nd.name;
        hom.hpolys = benchmark.firi_nd.hpolys;
        hom.solver_ms = benchmark.firi_nd.total_time_ms;
        hom.shared_ms = shared_ms;
        firi_benchmark::SteadyTimer hom_shortcut_timer;
        sfc_gen_benchmark::shortCut(hom.hpolys);
        hom.shortcut_ms = hom_shortcut_timer.elapsedMs();
        fillCorridorMetrics(hom);

        return {baseline, hom};
    }

    void runPlanningCase(const firi_benchmark::PlanningCase &planning_case,
                         firi_benchmark::CsvWriter &writer) const
    {
        firi_benchmark::SteadyTimer path_timer;
        std::vector<Eigen::Vector3d> route;
        ompl::RNG::setSeed(static_cast<std::uint_fast32_t>(planning_case.deterministic_seed));
        const double path_cost = sfc_gen::planPath<voxel_map::VoxelMap>(planning_case.start,
                                                                        planning_case.goal,
                                                                        voxel_map_.getOrigin(),
                                                                        voxel_map_.getCorner(),
                                                                        &voxel_map_,
                                                                        config_.timeout_rrt,
                                                                        route);
        const double path_search_ms = path_timer.elapsedMs();
        const bool path_success = std::isfinite(path_cost) && route.size() > 1;
        const std::uint64_t route_hash = route.empty() ? 0 :
            region_inflation::hashEigenDense(firi_benchmark::makePointMatrix(route));
        const double route_length = firi_benchmark::routeLength(route);

        firi_benchmark::SteadyTimer surface_timer;
        std::vector<Eigen::Vector3d> surface_points;
        if (path_success)
        {
            voxel_map_.getSurf(surface_points);
        }
        const double surface_extract_ms = surface_timer.elapsedMs();

        if (!path_success)
        {
            writeFailureRows(writer,
                             planning_case,
                             path_success,
                             path_search_ms,
                             surface_extract_ms,
                             route_hash,
                             route.size(),
                             route_length,
                             "path_search_failed");
            return;
        }

        double corridor_wall_ms = 0.0;
        const std::vector<MethodCorridor> corridors =
            makeCorridors(route, surface_points, corridor_wall_ms);

        ROS_INFO_STREAM("case=" << planning_case.case_id
                        << ", route_points=" << route.size()
                        << ", route_length=" << route_length
                        << ", corridor_wall_ms=" << corridor_wall_ms);

        for (const MethodCorridor &corridor : corridors)
        {
            const bool corridor_success = !corridor.hpolys.empty();
            TrajectoryResult traj;
            if (corridor_success)
            {
                traj = optimizeGlobalPlanningCorridor(planning_case.start,
                                                      planning_case.goal,
                                                      corridor.hpolys,
                                                      config_.trajectory_config,
                                                      voxel_map_);
            }
            else
            {
                traj.failure_reason = "empty_corridor";
            }

            const bool global_planning_success =
                path_success && corridor_success && traj.setup_success &&
                traj.optimize_success && traj.pieces > 0;
            const bool success = global_planning_success && traj.collision_free;
            const double end_to_end_ms = path_search_ms +
                                         surface_extract_ms +
                                         corridor.corridorMs() +
                                         traj.setup_ms +
                                         traj.optimize_ms;
            writer.writeRow(config_.run_id,
                            config_.mapId(),
                            config_.density,
                            planning_case.case_id,
                            corridor.method,
                            corridor.solver,
                            planning_case.start.x(),
                            planning_case.start.y(),
                            planning_case.start.z(),
                            planning_case.goal.x(),
                            planning_case.goal.y(),
                            planning_case.goal.z(),
                            path_success,
                            path_search_ms,
                            surface_extract_ms,
                            route_hash,
                            route.size(),
                            route_length,
                            corridor_success,
                            corridor.hpolys.size(),
                            corridor.shared_ms,
                            corridor.solver_ms,
                            corridor.shortcut_ms,
                            corridor.corridorMs(),
                            corridor.volume_sum,
                            corridor.faces_total,
                            traj.setup_success,
                            traj.setup_ms,
                            traj.optimize_success,
                            traj.optimize_ms,
                            traj.cost,
                            traj.duration,
                            traj.pieces,
                            traj.sampled_collision_count,
                            traj.collision_free,
                            end_to_end_ms,
                            global_planning_success,
                            success,
                            traj.failure_reason);
        }
    }

    void writeFailureRows(firi_benchmark::CsvWriter &writer,
                          const firi_benchmark::PlanningCase &planning_case,
                          const bool path_success,
                          const double path_search_ms,
                          const double surface_extract_ms,
                          const std::uint64_t route_hash,
                          const std::size_t route_points,
                          const double route_length,
                          const std::string &failure_reason) const
    {
        for (const char *method : {"baseline_firi", "hom_mvie"})
        {
            writer.writeRow(config_.run_id,
                            config_.mapId(),
                            config_.density,
                            planning_case.case_id,
                            method,
                            "",
                            planning_case.start.x(),
                            planning_case.start.y(),
                            planning_case.start.z(),
                            planning_case.goal.x(),
                            planning_case.goal.y(),
                            planning_case.goal.z(),
                            path_success,
                            path_search_ms,
                            surface_extract_ms,
                            route_hash,
                            route_points,
                            route_length,
                            false,
                            0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0,
                            false,
                            0.0,
                            false,
                            0.0,
                            INFINITY,
                            0.0,
                            0,
                            0,
                            false,
                            path_search_ms + surface_extract_ms,
                            false,
                            false,
                            failure_reason);
        }
    }

    void writeMetadata(const std::size_t cases) const
    {
        std::ofstream os(config_.output_dir + "/global_planning_metadata.json");
        os << std::setprecision(12);
        os << "{\n";
        os << "  \"run_id\": \"" << config_.run_id << "\",\n";
        os << "  \"map_id\": \"" << config_.mapId() << "\",\n";
        os << "  \"density\": \"" << config_.density << "\",\n";
        os << "  \"map_seed\": " << config_.map_seed << ",\n";
        os << "  \"master_seed\": " << config_.master_seed << ",\n";
        os << "  \"planning_case_count_requested\": " << config_.planning_case_count << ",\n";
        os << "  \"planning_case_count_sampled\": " << cases << ",\n";
        os << "  \"planning_sample_margin\": " << config_.planning_sample_margin << ",\n";
        os << "  \"timeout_rrt\": " << config_.timeout_rrt << ",\n";
        os << "  \"corridor_progress\": " << config_.corridor_progress << ",\n";
        os << "  \"corridor_range\": " << config_.local_range << ",\n";
        os << "  \"map_preprocess_ms\": " << map_preprocess_ms_ << "\n";
        os << "}\n";
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "global_planning_benchmark_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    GlobalPlanningBenchmarkNode node(BenchmarkConfig(pnh), nh);
    ros::Rate rate(100.0);
    while (ros::ok() && !node.ready())
    {
        ros::spinOnce();
        rate.sleep();
    }
    if (!ros::ok())
    {
        return 1;
    }

    const bool ok = node.run();
    ros::shutdown();
    return ok ? 0 : 1;
}
