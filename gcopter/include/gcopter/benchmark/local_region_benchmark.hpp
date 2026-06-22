#ifndef GCOPTER_BENCHMARK_LOCAL_REGION_BENCHMARK_HPP
#define GCOPTER_BENCHMARK_LOCAL_REGION_BENCHMARK_HPP

#include "gcopter/benchmark/benchmark_types.hpp"
#include "gcopter/benchmark/csv_writer.hpp"
#include "gcopter/region_inflation/firi_constraint_builder.hpp"
#include "gcopter/region_inflation/iterative_region_generator.hpp"
#include "gcopter/region_inflation/mvie_hom.hpp"
#include "gcopter/region_inflation/mvie_legacy.hpp"

#include <algorithm>
#include <memory>

namespace firi_benchmark
{
    inline region_inflation::RegionInput makeRegionInput(const RegionCase &region_case)
    {
        region_inflation::RegionInput input;
        input.boundary = region_case.boundary;
        input.point_cloud = region_case.local_points;
        input.a = region_case.a;
        input.b = region_case.b;
        input.map_id = region_case.map_id;
        input.density = region_case.density;
        input.region_case_id = region_case.case_id;
        input.seed_type = region_case.seed_type;
        input.seed_length = region_case.seed_length;
        input.input_hash = region_case.input_hash;
        input.global_surface_points = region_case.global_surface_points;
        return input;
    }

    inline bool runRegionMethod(const RegionCase &region_case,
                                const region_inflation::MethodConfig &method,
                                const region_inflation::FiriOptions &options,
                                region_inflation::RegionOutput &output,
                                region_inflation::RegionStats &stats,
                                std::vector<region_inflation::MvieReplayCase> *replay_cases = nullptr)
    {
        region_inflation::FullFiriConstraintBuilder builder(options.geometric_epsilon);
        const region_inflation::RegionInput input = makeRegionInput(region_case);

        if (method.mvie_solver == region_inflation::MvieSolverKind::LegacyPenalty)
        {
            region_inflation::LegacyPenaltyMvieSolver3D solver;
            return region_inflation::generateRegion(input, options, builder, solver, output, stats, replay_cases);
        }
        else
        {
            region_inflation::HomGaugeMvieSolver3D solver;
            return region_inflation::generateRegion(input, options, builder, solver, output, stats, replay_cases);
        }
    }

    inline double maxMvieResidual(const region_inflation::RegionStats &stats)
    {
        double worst = 0.0;
        for (const auto &call : stats.mvie_calls)
        {
            worst = std::max(worst, call.max_constraint_residual);
        }
        return worst;
    }

    inline void writeRegionTrialRow(CsvWriter &writer,
                                    const std::string &run_id,
                                    const RegionCase &region_case,
                                    const region_inflation::MethodConfig &method,
                                    const int repeat_id,
                                    const int order_index,
                                    const region_inflation::RegionOutput &output,
                                    const region_inflation::RegionStats &stats,
                                    const double tolerance)
    {
        const PolytopeMetrics metrics =
            stats.success ? measureBoundedPolytope(output.hpoly, region_case.boundary) : PolytopeMetrics();
        const bool seed_ok = stats.success && seedIncluded(output.hpoly, region_case.a, region_case.b, tolerance);
        const int interior_obstacles = stats.success ?
            countInteriorObstacles(output.hpoly, region_case.local_points, tolerance) : 0;
        const double region_online_ms = region_case.crop_ms + stats.total_region_ms;

        writer.writeRow(run_id,
                        region_case.map_id,
                        region_case.density,
                        region_case.case_id,
                        region_case.seed_type,
                        region_case.seed_length,
                        region_case.a.x(),
                        region_case.a.y(),
                        region_case.a.z(),
                        region_case.b.x(),
                        region_case.b.y(),
                        region_case.b.z(),
                        method.name,
                        repeat_id,
                        order_index,
                        region_case.global_surface_points,
                        region_case.local_points.cols(),
                        region_case.input_hash,
                        stats.success,
                        stats.failure_stage,
                        stats.outer_build_count,
                        stats.mvie_call_count,
                        stats.total_mvie_iterations,
                        stats.total_objective_evaluations,
                        region_case.crop_ms,
                        stats.constraint_stats.prepare_ms,
                        stats.constraint_stats.build_ms,
                        stats.total_mvie_ms,
                        stats.constraint_stats.certification_ms,
                        stats.constraint_stats.repair_ms,
                        stats.total_region_ms,
                        region_online_ms,
                        metrics.face_count,
                        metrics.vertex_count,
                        metrics.volume,
                        seed_ok,
                        interior_obstacles,
                        maxMvieResidual(stats),
                        stats.final_logdet);
    }

    inline void runRegionTrials(const std::string &run_id,
                                const std::vector<RegionCase> &cases,
                                const std::vector<region_inflation::MethodConfig> &methods,
                                const region_inflation::FiriOptions &options,
                                const int repeats,
                                CsvWriter &writer)
    {
        for (const RegionCase &region_case : cases)
        {
            for (int repeat = 0; repeat < repeats; ++repeat)
            {
                std::vector<int> order(methods.size());
                for (std::size_t i = 0; i < methods.size(); ++i)
                {
                    order[i] = static_cast<int>(i);
                }
                if (((region_case.input_hash + static_cast<std::uint64_t>(repeat)) & 1ULL) != 0ULL)
                {
                    std::reverse(order.begin(), order.end());
                }

                for (std::size_t order_index = 0; order_index < order.size(); ++order_index)
                {
                    const region_inflation::MethodConfig &method = methods[order[order_index]];
                    region_inflation::RegionOutput output;
                    region_inflation::RegionStats stats;
                    runRegionMethod(region_case, method, options, output, stats, nullptr);
                    writeRegionTrialRow(writer,
                                        run_id,
                                        region_case,
                                        method,
                                        repeat,
                                        static_cast<int>(order_index),
                                        output,
                                        stats,
                                        options.mvie_options.feasibility_tolerance);
                }
            }
        }
    }

    inline void collectReplayCases(const std::vector<RegionCase> &cases,
                                   const region_inflation::FiriOptions &options,
                                   std::vector<region_inflation::MvieReplayCase> &replay_cases)
    {
        const region_inflation::MethodConfig legacy{
            "firi_legacy",
            region_inflation::ConstraintBuilderKind::FullFiri,
            region_inflation::MvieSolverKind::LegacyPenalty};
        for (const RegionCase &region_case : cases)
        {
            region_inflation::RegionOutput output;
            region_inflation::RegionStats stats;
            runRegionMethod(region_case, legacy, options, output, stats, &replay_cases);
        }
    }

    inline bool solveReplayCase(const region_inflation::MvieReplayCase &replay_case,
                                const region_inflation::MethodConfig &method,
                                const region_inflation::MvieOptions &options,
                                region_inflation::MvieStats &stats)
    {
        region_inflation::Ellipsoid3D warm_start = replay_case.warm_start;
        if (method.mvie_solver == region_inflation::MvieSolverKind::LegacyPenalty)
        {
            region_inflation::LegacyPenaltyMvieSolver3D solver;
            return solver.solve(replay_case.hpoly, warm_start, options, stats);
        }
        region_inflation::HomGaugeMvieSolver3D solver;
        return solver.solve(replay_case.hpoly, warm_start, options, stats);
    }

    inline void writeMvieReplayRow(CsvWriter &writer,
                                   const std::string &run_id,
                                   const region_inflation::MvieReplayCase &replay_case,
                                   const region_inflation::MethodConfig &method,
                                   const int repeat_id,
                                   const int order_index,
                                   const region_inflation::MvieStats &stats)
    {
        writer.writeRow(run_id,
                        replay_case.map_id,
                        replay_case.density,
                        replay_case.region_case_id,
                        replay_case.outer_iteration,
                        replay_case.seed_type,
                        replay_case.seed_length,
                        replay_case.hpoly.rows(),
                        replay_case.local_point_count,
                        replay_case.input_hash,
                        replay_case.warm_start_hash,
                        method.name,
                        repeat_id,
                        order_index,
                        stats.success,
                        stats.status,
                        stats.iterations,
                        stats.objective_evaluations,
                        stats.solve_ms,
                        stats.logdet_l,
                        stats.max_mu,
                        stats.max_constraint_residual,
                        stats.active_constraint_count,
                        stats.finite_output,
                        stats.positive_diagonal,
                        stats.failure_reason);
    }

    inline void runMvieReplayBenchmark(const std::string &run_id,
                                       const std::vector<region_inflation::MvieReplayCase> &replay_cases,
                                       const std::vector<region_inflation::MethodConfig> &methods,
                                       const region_inflation::MvieOptions &options,
                                       const int repeats,
                                       const int warmup_repeats,
                                       CsvWriter &writer)
    {
        for (const region_inflation::MvieReplayCase &replay_case : replay_cases)
        {
            for (int repeat = -warmup_repeats; repeat < repeats; ++repeat)
            {
                std::vector<int> order(methods.size());
                for (std::size_t i = 0; i < methods.size(); ++i)
                {
                    order[i] = static_cast<int>(i);
                }
                if (((replay_case.input_hash ^ replay_case.warm_start_hash ^
                      static_cast<std::uint64_t>(std::max(0, repeat))) & 1ULL) != 0ULL)
                {
                    std::reverse(order.begin(), order.end());
                }

                for (std::size_t order_index = 0; order_index < order.size(); ++order_index)
                {
                    const region_inflation::MethodConfig &method = methods[order[order_index]];
                    region_inflation::MvieStats stats;
                    solveReplayCase(replay_case, method, options, stats);
                    if (repeat >= 0)
                    {
                        writeMvieReplayRow(writer,
                                           run_id,
                                           replay_case,
                                           method,
                                           repeat,
                                           static_cast<int>(order_index),
                                           stats);
                    }
                }
            }
        }
    }
}

#endif
