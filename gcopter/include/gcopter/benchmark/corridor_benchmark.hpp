#ifndef GCOPTER_BENCHMARK_CORRIDOR_BENCHMARK_HPP
#define GCOPTER_BENCHMARK_CORRIDOR_BENCHMARK_HPP

#include "gcopter/benchmark/local_region_benchmark.hpp"
#include "gcopter/benchmark/seed_sampler.hpp"
#include "gcopter/geo_utils.hpp"

#include <deque>

namespace firi_benchmark
{
    inline bool needGapPolytope(const std::vector<Eigen::MatrixX4d> &hpolys,
                                const Eigen::MatrixX4d &current,
                                const Eigen::Vector3d &anchor,
                                const double eps)
    {
        if (hpolys.empty() || current.rows() == 0)
        {
            return false;
        }
        const Eigen::Vector4d ah(anchor(0), anchor(1), anchor(2), 1.0);
        return 3 <= ((current * ah).array() > -eps).cast<int>().sum() +
                        ((hpolys.back() * ah).array() > -eps).cast<int>().sum();
    }

    inline void shortCut(std::vector<Eigen::MatrixX4d> &hpolys)
    {
        std::vector<Eigen::MatrixX4d> htemp = hpolys;
        if (htemp.empty())
        {
            hpolys.clear();
            return;
        }
        if (htemp.size() == 1)
        {
            htemp.insert(htemp.begin(), htemp.front());
        }
        hpolys.clear();

        const int M = htemp.size();
        std::deque<int> indices;
        indices.push_front(M - 1);
        for (int i = M - 1; i >= 0; --i)
        {
            for (int j = 0; j < i; ++j)
            {
                const bool overlap = j < i - 1 ? geo_utils::overlap(htemp[i], htemp[j], 0.01) : true;
                if (overlap)
                {
                    indices.push_front(j);
                    i = j + 1;
                    break;
                }
            }
        }
        for (const int idx : indices)
        {
            hpolys.push_back(htemp[idx]);
        }
    }

    inline std::vector<PathSeedCase> makePathSeedCases(const std::vector<Eigen::Vector3d> &route,
                                                       const std::vector<Eigen::Vector3d> &surface_points,
                                                       const Eigen::Vector3d &map_low,
                                                       const Eigen::Vector3d &map_high,
                                                       const double progress,
                                                       const double range)
    {
        std::vector<PathSeedCase> seeds;
        if (route.size() < 2)
        {
            return seeds;
        }

        Eigen::Vector3d a;
        Eigen::Vector3d b = route.front();
        for (std::size_t i = 1; i < route.size();)
        {
            a = b;
            if ((a - route[i]).norm() > progress)
            {
                b = (route[i] - a).normalized() * progress + a;
            }
            else
            {
                b = route[i];
                ++i;
            }

            PathSeedCase seed;
            seed.a = a;
            seed.b = b;
            seed.boundary = pathSegmentBoundary(a, b, map_low, map_high, range);
            SteadyTimer timer;
            seed.local_points = cropLocalPoints(surface_points, seed.boundary);
            seed.crop_ms = timer.elapsedMs();
            seed.input_hash = stableInputHash(seed.boundary, seed.local_points, seed.a, seed.b);
            seeds.push_back(seed);
        }
        return seeds;
    }

    inline RegionCase pathSeedToRegionCase(const PathSeedCase &seed,
                                           const std::string &map_id,
                                           const std::string &density,
                                           const std::string &case_id,
                                           const std::size_t global_surface_points)
    {
        RegionCase region_case;
        region_case.case_id = case_id;
        region_case.map_id = map_id;
        region_case.density = density;
        region_case.seed_type = "path_segment";
        region_case.a = seed.a;
        region_case.b = seed.b;
        region_case.boundary = seed.boundary;
        region_case.local_points = seed.local_points;
        region_case.seed_length = (seed.b - seed.a).norm();
        region_case.crop_ms = seed.crop_ms;
        region_case.input_hash = seed.input_hash;
        region_case.global_surface_points = global_surface_points;
        return region_case;
    }

    inline void fillCorridorMetrics(CorridorTrialStats &stats,
                                    const Eigen::Matrix3Xd *local_points_for_obstacles,
                                    const double tolerance)
    {
        stats.regions_after_shortcut = static_cast<int>(stats.hpolys.size());
        stats.faces_total = 0;
        stats.volume_sum = 0.0;
        stats.interior_obstacle_count = 0;
        for (const Eigen::MatrixX4d &hpoly : stats.hpolys)
        {
            const PolytopeMetrics metrics = measurePolytope(hpoly);
            stats.faces_total += static_cast<int>(metrics.face_count);
            stats.volume_sum += metrics.volume;
            if (local_points_for_obstacles != nullptr)
            {
                stats.interior_obstacle_count += countInteriorObstacles(hpoly, *local_points_for_obstacles, tolerance);
            }
        }
        const int n = std::max(1, stats.regions_after_shortcut);
        stats.faces_mean = static_cast<double>(stats.faces_total) / static_cast<double>(n);
        stats.volume_mean = stats.volume_sum / static_cast<double>(n);

        SteadyTimer overlap_timer;
        stats.overlap_failures = 0;
        for (std::size_t i = 1; i < stats.hpolys.size(); ++i)
        {
            if (!geo_utils::overlap(stats.hpolys[i - 1], stats.hpolys[i], tolerance))
            {
                ++stats.overlap_failures;
            }
        }
        stats.overlap_check_ms += overlap_timer.elapsedMs();
    }

    inline CorridorTrialStats runFixedSeedCorridor(const std::vector<PathSeedCase> &seeds,
                                                   const std::string &map_id,
                                                   const std::string &density,
                                                   const region_inflation::MethodConfig &method,
                                                   const region_inflation::FiriOptions &options,
                                                   const std::size_t global_surface_points)
    {
        CorridorTrialStats stats;
        stats.candidate_seed_count = static_cast<int>(seeds.size());
        for (std::size_t i = 0; i < seeds.size(); ++i)
        {
            const RegionCase region_case =
                pathSeedToRegionCase(seeds[i], map_id, density, caseId("path_seed", static_cast<int>(i)), global_surface_points);
            region_inflation::RegionOutput output;
            region_inflation::RegionStats region_stats;
            runRegionMethod(region_case, method, options, output, region_stats, nullptr);
            ++stats.region_call_count;
            stats.local_points_total += static_cast<std::size_t>(region_case.local_points.cols());
            stats.crop_ms += region_case.crop_ms;
            stats.region_core_ms += region_stats.total_region_ms;
            if (!region_stats.success)
            {
                stats.failure_reason = region_stats.failure_reason;
                continue;
            }
            const Eigen::MatrixX4d bounded_hpoly = stackHpolys(output.hpoly, region_case.boundary);
            if (!seedIncluded(bounded_hpoly, region_case.a, region_case.b, options.mvie_options.feasibility_tolerance))
            {
                ++stats.seed_inclusion_failures;
            }
            stats.hpolys.push_back(bounded_hpoly);
        }
        stats.regions_before_shortcut = static_cast<int>(stats.hpolys.size());
        fillCorridorMetrics(stats, nullptr, options.mvie_options.feasibility_tolerance);
        stats.corridor_total_ms = stats.crop_ms + stats.region_core_ms + stats.overlap_check_ms;
        stats.success = stats.region_call_count == static_cast<int>(seeds.size()) &&
                        stats.regions_after_shortcut > 0 &&
                        stats.seed_inclusion_failures == 0 &&
                        stats.overlap_failures == 0;
        if (!stats.success && stats.failure_reason.empty())
        {
            stats.failure_reason = "fixed_seed_corridor_quality_check_failed";
        }
        return stats;
    }

    inline CorridorTrialStats runCurrentCoverCorridor(const std::vector<PathSeedCase> &seeds,
                                                      const std::string &map_id,
                                                      const std::string &density,
                                                      const region_inflation::MethodConfig &method,
                                                      const region_inflation::FiriOptions &options,
                                                      const std::size_t global_surface_points)
    {
        CorridorTrialStats stats;
        stats.candidate_seed_count = static_cast<int>(seeds.size());
        for (std::size_t i = 0; i < seeds.size(); ++i)
        {
            const RegionCase region_case =
                pathSeedToRegionCase(seeds[i], map_id, density, caseId("path_seed", static_cast<int>(i)), global_surface_points);
            region_inflation::RegionOutput output;
            region_inflation::RegionStats region_stats;
            runRegionMethod(region_case, method, options, output, region_stats, nullptr);
            ++stats.region_call_count;
            stats.local_points_total += static_cast<std::size_t>(region_case.local_points.cols());
            stats.crop_ms += region_case.crop_ms;
            stats.region_core_ms += region_stats.total_region_ms;
            if (!region_stats.success)
            {
                stats.failure_reason = region_stats.failure_reason;
                continue;
            }
            const Eigen::MatrixX4d bounded_hpoly = stackHpolys(output.hpoly, region_case.boundary);
            if (needGapPolytope(stats.hpolys, bounded_hpoly, region_case.a, options.geometric_epsilon))
            {
                RegionCase gap_case = region_case;
                gap_case.b = gap_case.a;
                gap_case.seed_length = 0.0;
                gap_case.input_hash = stableInputHash(gap_case.boundary, gap_case.local_points, gap_case.a, gap_case.b);
                region_inflation::FiriOptions gap_options = options;
                gap_options.outer_stop_mode = region_inflation::OuterStopMode::FixedIterations;
                gap_options.max_outer_iterations = 1;
                region_inflation::RegionOutput gap_output;
                region_inflation::RegionStats gap_stats;
                runRegionMethod(gap_case, method, gap_options, gap_output, gap_stats, nullptr);
                ++stats.region_call_count;
                ++stats.gap_region_count;
                stats.gap_region_ms += gap_stats.total_region_ms;
                if (gap_stats.success)
                {
                    stats.hpolys.push_back(stackHpolys(gap_output.hpoly, gap_case.boundary));
                }
            }
            if (!seedIncluded(bounded_hpoly, region_case.a, region_case.b, options.mvie_options.feasibility_tolerance))
            {
                ++stats.seed_inclusion_failures;
            }
            stats.hpolys.push_back(bounded_hpoly);
        }

        stats.regions_before_shortcut = static_cast<int>(stats.hpolys.size());
        SteadyTimer shortcut_timer;
        shortCut(stats.hpolys);
        stats.shortcut_ms = shortcut_timer.elapsedMs();
        fillCorridorMetrics(stats, nullptr, options.mvie_options.feasibility_tolerance);
        stats.corridor_total_ms = stats.crop_ms +
                                  stats.region_core_ms +
                                  stats.gap_region_ms +
                                  stats.shortcut_ms +
                                  stats.overlap_check_ms;
        stats.success = stats.regions_after_shortcut > 0 &&
                        stats.seed_inclusion_failures == 0 &&
                        stats.overlap_failures == 0;
        if (!stats.success && stats.failure_reason.empty())
        {
            stats.failure_reason = "current_cover_corridor_quality_check_failed";
        }
        return stats;
    }

    inline void writeCorridorTrialRow(CsvWriter &writer,
                                      const std::string &run_id,
                                      const std::string &map_id,
                                      const std::string &density,
                                      const std::string &planning_case_id,
                                      const std::string &corridor_mode,
                                      const region_inflation::MethodConfig &method,
                                      const int repeat_id,
                                      const std::uint64_t route_hash,
                                      const std::size_t route_point_count,
                                      const double route_length,
                                      const CorridorTrialStats &stats)
    {
        writer.writeRow(run_id,
                        map_id,
                        density,
                        planning_case_id,
                        corridor_mode,
                        method.name,
                        repeat_id,
                        route_hash,
                        route_point_count,
                        route_length,
                        stats.candidate_seed_count,
                        stats.region_call_count,
                        stats.gap_region_count,
                        stats.regions_before_shortcut,
                        stats.regions_after_shortcut,
                        stats.local_points_total,
                        stats.crop_ms,
                        stats.region_core_ms,
                        stats.gap_region_ms,
                        stats.shortcut_ms,
                        stats.overlap_check_ms,
                        stats.corridor_total_ms,
                        stats.faces_total,
                        stats.faces_mean,
                        stats.volume_sum,
                        stats.volume_mean,
                        stats.seed_inclusion_failures,
                        stats.overlap_failures,
                        stats.interior_obstacle_count,
                        stats.success,
                        stats.failure_reason);
    }
}

#endif
