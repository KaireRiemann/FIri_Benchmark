#ifndef GCOPTER_REGION_INFLATION_ITERATIVE_REGION_GENERATOR_HPP
#define GCOPTER_REGION_INFLATION_ITERATIVE_REGION_GENERATOR_HPP

#include "gcopter/region_inflation/mvie_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace region_inflation
{
    inline double elapsedMsSince(const std::chrono::steady_clock::time_point &start)
    {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }

    inline bool seedInsideBoundary(const Eigen::MatrixX4d &bd,
                                   const Eigen::Vector3d &a,
                                   const Eigen::Vector3d &b,
                                   const double tol)
    {
        const Eigen::Vector4d ah(a(0), a(1), a(2), 1.0);
        const Eigen::Vector4d bh(b(0), b(1), b(2), 1.0);
        return (bd * ah).maxCoeff() <= tol && (bd * bh).maxCoeff() <= tol;
    }

    inline double ellipsoidMaxResidual(const Eigen::MatrixX4d &hpoly,
                                       const Ellipsoid3D &ellipsoid)
    {
        if (hpoly.rows() == 0)
        {
            return 0.0;
        }
        const Eigen::Matrix3d L = ellipsoid.R * ellipsoid.radii.asDiagonal();
        const Eigen::VectorXd residual =
            (hpoly.leftCols<3>() * L).rowwise().norm() +
            hpoly.leftCols<3>() * ellipsoid.center +
            hpoly.col(3);
        return residual.maxCoeff();
    }

    template <class ConstraintBuilder, class MvieSolver>
    bool generateRegion(const RegionInput &input,
                        const FiriOptions &options,
                        ConstraintBuilder &builder,
                        const MvieSolver &mvie_solver,
                        RegionOutput &output,
                        RegionStats &stats,
                        std::vector<MvieReplayCase> *replay_cases = nullptr)
    {
        stats = RegionStats();
        output = RegionOutput();
        const auto total_start = std::chrono::steady_clock::now();

        if (!seedInsideBoundary(input.boundary, input.a, input.b, options.geometric_epsilon))
        {
            stats.failure_stage = "input";
            stats.failure_reason = "seed_outside_boundary";
            stats.total_region_ms = elapsedMsSince(total_start);
            return false;
        }

        Ellipsoid3D current;
        current.R.setIdentity();
        current.center = 0.5 * (input.a + input.b);
        current.radii.setOnes();

        if (!builder.prepare(input, stats.constraint_stats))
        {
            stats.failure_stage = "constraint_prepare";
            stats.failure_reason = "builder_prepare_failed";
            stats.total_region_ms = elapsedMsSince(total_start);
            return false;
        }

        const int max_outer = std::max(1, options.max_outer_iterations);
        double previous_volume = current.volumeProxy();

        for (int loop = 0; loop < max_outer; ++loop)
        {
            if (!builder.build(input, current, output.hpoly, stats.constraint_stats))
            {
                stats.failure_stage = "constraint_build";
                stats.failure_reason = "builder_build_failed";
                stats.total_region_ms = elapsedMsSince(total_start);
                return false;
            }
            ++stats.outer_build_count;

            const bool last_fixed_iteration =
                options.outer_stop_mode == OuterStopMode::FixedIterations &&
                loop == max_outer - 1;
            if (last_fixed_iteration)
            {
                break;
            }

            if (replay_cases != nullptr)
            {
                MvieReplayCase replay;
                replay.hpoly = output.hpoly;
                replay.warm_start = current;
                replay.map_id = input.map_id;
                replay.density = input.density;
                replay.region_case_id = input.region_case_id;
                replay.outer_iteration = loop;
                replay.seed_type = input.seed_type;
                replay.seed_length = input.seed_length;
                replay.local_point_count = static_cast<std::size_t>(input.point_cloud.cols());
                replay.input_hash = input.input_hash;
                replay.warm_start_hash = hashEllipsoidWarmStart(current);
                replay_cases->push_back(replay);
            }

            Ellipsoid3D candidate = current;
            MvieStats mvie_stats;
            const bool mvie_ok = mvie_solver.solve(output.hpoly, candidate, options.mvie_options, mvie_stats);
            stats.mvie_calls.push_back(mvie_stats);
            ++stats.mvie_call_count;
            stats.total_mvie_iterations += mvie_stats.iterations;
            stats.total_objective_evaluations += mvie_stats.objective_evaluations;
            stats.total_mvie_ms += mvie_stats.solve_ms;

            if (!mvie_ok)
            {
                stats.failure_stage = "mvie";
                stats.failure_reason = mvie_stats.failure_reason;
                stats.total_region_ms = elapsedMsSince(total_start);
                return false;
            }

            bool accept = true;
            if (options.enable_monotonic_acceptance)
            {
                const double candidate_residual = ellipsoidMaxResidual(output.hpoly, candidate);
                const double previous_logdet = current.logDetL();
                const double candidate_logdet = candidate.logDetL();
                accept = candidate_residual <= options.mvie_options.feasibility_tolerance &&
                         candidate_logdet + options.acceptance_tolerance >= previous_logdet;
            }

            if (accept)
            {
                current = candidate;
            }
            else
            {
                ++stats.rejected_mvie_updates;
            }

            if (options.outer_stop_mode == OuterStopMode::RelativeEllipsoidVolume)
            {
                const double current_volume = current.volumeProxy();
                const double relative_growth =
                    previous_volume > 0.0 ? (current_volume - previous_volume) / previous_volume : INFINITY;
                previous_volume = current_volume;
                if (loop + 1 >= max_outer || std::abs(relative_growth) < options.relative_volume_tolerance)
                {
                    break;
                }
            }
        }

        if (!builder.finalize(input, current, output.hpoly, stats.constraint_stats))
        {
            stats.failure_stage = "constraint_finalize";
            stats.failure_reason = "builder_finalize_failed";
            stats.total_region_ms = elapsedMsSince(total_start);
            return false;
        }

        output.final_ellipsoid = current;
        stats.total_constraint_build_ms = stats.constraint_stats.prepare_ms +
                                          stats.constraint_stats.build_ms +
                                          stats.constraint_stats.certification_ms +
                                          stats.constraint_stats.repair_ms;
        stats.final_logdet = current.logDetL();
        stats.total_region_ms = elapsedMsSince(total_start);
        stats.success = true;
        return true;
    }
}

#endif
