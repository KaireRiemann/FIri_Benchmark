#ifndef GCOPTER_BENCHMARK_PLANNING_BENCHMARK_HPP
#define GCOPTER_BENCHMARK_PLANNING_BENCHMARK_HPP

#include "gcopter/benchmark/benchmark_timer.hpp"
#include "gcopter/benchmark/benchmark_types.hpp"
#include "gcopter/geo_utils.hpp"
#include "gcopter/gcopter.hpp"
#include "gcopter/trajectory.hpp"
#include "gcopter/voxel_map.hpp"

#include <Eigen/Eigen>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace firi_benchmark
{
    struct TrajectoryOptimizerConfig
    {
        double max_vel_mag = 4.0;
        double max_bdr_mag = 2.1;
        double max_tilt_angle = 1.05;
        double min_thrust = 2.0;
        double max_thrust = 12.0;
        double vehicle_mass = 0.61;
        double grav_acc = 9.8;
        double horiz_drag = 0.70;
        double vert_drag = 0.80;
        double paras_drag = 0.01;
        double speed_eps = 1.0e-4;
        double weight_t = 20.0;
        Eigen::VectorXd chi_vec = Eigen::VectorXd::Ones(5);
        double smoothing_eps = 1.0e-2;
        int integral_intervals = 16;
        double rel_cost_tol = 1.0e-5;
        double validation_dt = 0.05;
    };

    struct TrajectoryTrialResult
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

    inline int countTrajectoryCollisions(const Trajectory<5> &traj,
                                         const voxel_map::VoxelMap &map,
                                         const double dt)
    {
        int collisions = 0;
        const double duration = traj.getTotalDuration();
        if (!(duration > 0.0) || !(dt > 0.0))
        {
            return 1;
        }
        const int samples = std::max(1, static_cast<int>(std::ceil(duration / dt)));
        for (int i = 0; i <= samples; ++i)
        {
            const double t = std::min(duration, static_cast<double>(i) * dt);
            if (map.query(traj.getPos(t)) != 0)
            {
                ++collisions;
            }
        }
        return collisions;
    }

    inline bool validatePolytopeForTrajectory(const Eigen::MatrixX4d &hpoly,
                                              std::string &reason)
    {
        if (hpoly.rows() < 4 || hpoly.cols() != 4 || !hpoly.array().isFinite().all())
        {
            reason = "invalid_polytope_matrix";
            return false;
        }
        const Eigen::ArrayXd norms = hpoly.leftCols<3>().rowwise().norm();
        if (!(norms > 0.0).all())
        {
            reason = "zero_polytope_normal";
            return false;
        }
        const PolytopeMetrics metrics = measurePolytope(hpoly);
        if (metrics.vertex_count < 4)
        {
            reason = "degenerate_polytope_vertices";
            return false;
        }
        return true;
    }

    inline bool validateCorridorForTrajectory(const Eigen::Vector3d &start,
                                              const Eigen::Vector3d &goal,
                                              const std::vector<Eigen::MatrixX4d> &hpolys,
                                              std::string &reason)
    {
        if (hpolys.empty())
        {
            reason = "empty_corridor";
            return false;
        }
        if (!pointInsideHpoly(hpolys.front(), start, 1.0e-6))
        {
            reason = "start_outside_corridor";
            return false;
        }
        if (!pointInsideHpoly(hpolys.back(), goal, 1.0e-6))
        {
            reason = "goal_outside_corridor";
            return false;
        }
        for (std::size_t i = 0; i < hpolys.size(); ++i)
        {
            if (!validatePolytopeForTrajectory(hpolys[i], reason))
            {
                reason = "polytope_" + std::to_string(i) + "_" + reason;
                return false;
            }
        }
        for (std::size_t i = 1; i < hpolys.size(); ++i)
        {
            Eigen::MatrixX4d intersection(hpolys[i - 1].rows() + hpolys[i].rows(), 4);
            intersection.topRows(hpolys[i - 1].rows()) = hpolys[i - 1];
            intersection.bottomRows(hpolys[i].rows()) = hpolys[i];
            const PolytopeMetrics metrics = measurePolytope(intersection);
            if (metrics.vertex_count < 4)
            {
                reason = "degenerate_corridor_overlap_" + std::to_string(i - 1) + "_" + std::to_string(i);
                return false;
            }
        }
        return true;
    }

    inline TrajectoryTrialResult optimizeTrajectoryInCorridor(const Eigen::Vector3d &start,
                                                              const Eigen::Vector3d &goal,
                                                              const std::vector<Eigen::MatrixX4d> &hpolys,
                                                              const TrajectoryOptimizerConfig &config,
                                                              const voxel_map::VoxelMap &map)
    {
        TrajectoryTrialResult result;
        if (hpolys.empty())
        {
            result.failure_reason = "empty_corridor";
            return result;
        }
        std::string validation_reason;
        if (!validateCorridorForTrajectory(start, goal, hpolys, validation_reason))
        {
            result.failure_reason = validation_reason;
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

        SteadyTimer setup_timer;
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

        SteadyTimer optimize_timer;
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

        result.sampled_collision_count = countTrajectoryCollisions(trajectory, map, config.validation_dt);
        result.collision_free = result.sampled_collision_count == 0;
        if (!result.collision_free)
        {
            result.failure_reason = "trajectory_collision";
        }
        return result;
    }
}

#endif
