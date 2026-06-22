#ifndef GCOPTER_BENCHMARK_SEED_SAMPLER_HPP
#define GCOPTER_BENCHMARK_SEED_SAMPLER_HPP

#include "gcopter/benchmark/benchmark_timer.hpp"
#include "gcopter/benchmark/benchmark_types.hpp"
#include "gcopter/region_inflation/mvie_solver.hpp"
#include "gcopter/voxel_map.hpp"

#include <Eigen/Eigen>

#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>

namespace firi_benchmark
{
    inline std::uint64_t mixSeed(std::uint64_t seed, const std::uint64_t salt)
    {
        seed ^= salt + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        seed ^= seed >> 30;
        seed *= 0xbf58476d1ce4e5b9ULL;
        seed ^= seed >> 27;
        seed *= 0x94d049bb133111ebULL;
        seed ^= seed >> 31;
        return seed;
    }

    inline std::string caseId(const std::string &prefix,
                              const int index)
    {
        std::ostringstream oss;
        oss << prefix << "_" << index;
        return oss.str();
    }

    inline bool hasFreeNeighborhood(const voxel_map::VoxelMap &map,
                                    const Eigen::Vector3d &q)
    {
        const Eigen::Vector3i id = map.posD2I(q);
        for (int dx = -1; dx <= 1; ++dx)
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dz = -1; dz <= 1; ++dz)
                {
                    const Eigen::Vector3i neighbor = id + Eigen::Vector3i(dx, dy, dz);
                    if (map.query(neighbor) != 0)
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    inline bool sampleFreePoint(const voxel_map::VoxelMap &map,
                                const Eigen::Vector3d &map_low,
                                const Eigen::Vector3d &map_high,
                                const double boundary_margin,
                                const int max_attempts,
                                std::mt19937_64 &rng,
                                Eigen::Vector3d &point)
    {
        const Eigen::Vector3d low = map_low + Eigen::Vector3d::Constant(boundary_margin);
        const Eigen::Vector3d high = map_high - Eigen::Vector3d::Constant(boundary_margin);
        if ((high.array() <= low.array()).any())
        {
            return false;
        }

        std::uniform_real_distribution<double> dx(low.x(), high.x());
        std::uniform_real_distribution<double> dy(low.y(), high.y());
        std::uniform_real_distribution<double> dz(low.z(), high.z());
        for (int attempt = 0; attempt < max_attempts; ++attempt)
        {
            const Eigen::Vector3d q(dx(rng), dy(rng), dz(rng));
            if (map.query(q) == 0 && hasFreeNeighborhood(map, q))
            {
                point = q;
                return true;
            }
        }
        return false;
    }

    inline bool lineSegmentFree(const voxel_map::VoxelMap &map,
                                const Eigen::Vector3d &a,
                                const Eigen::Vector3d &b,
                                const double step)
    {
        const double length = (b - a).norm();
        const int samples = std::max(1, static_cast<int>(std::ceil(length / step)));
        for (int i = 0; i <= samples; ++i)
        {
            const double t = static_cast<double>(i) / static_cast<double>(samples);
            const Eigen::Vector3d q = ((1.0 - t) * a + t * b).eval();
            if (map.query(q) != 0)
            {
                return false;
            }
        }
        return true;
    }

    inline Eigen::MatrixX4d localBoundary(const Eigen::Vector3d &a,
                                          const Eigen::Vector3d &b,
                                          const Eigen::Vector3d &map_low,
                                          const Eigen::Vector3d &map_high,
                                          const double local_range)
    {
        const Eigen::Vector3d center = 0.5 * (a + b);
        const Eigen::Vector3d low = center.cwiseMax(map_low + Eigen::Vector3d::Constant(local_range)) -
                                    Eigen::Vector3d::Constant(local_range);
        const Eigen::Vector3d high = center.cwiseMin(map_high - Eigen::Vector3d::Constant(local_range)) +
                                     Eigen::Vector3d::Constant(local_range);
        return makeBoxBoundary(low.cwiseMax(map_low), high.cwiseMin(map_high));
    }

    inline Eigen::MatrixX4d pathSegmentBoundary(const Eigen::Vector3d &a,
                                                const Eigen::Vector3d &b,
                                                const Eigen::Vector3d &map_low,
                                                const Eigen::Vector3d &map_high,
                                                const double range)
    {
        const Eigen::Vector3d low = a.cwiseMin(b) - Eigen::Vector3d::Constant(range);
        const Eigen::Vector3d high = a.cwiseMax(b) + Eigen::Vector3d::Constant(range);
        return makeBoxBoundary(low.cwiseMax(map_low), high.cwiseMin(map_high));
    }

    inline Eigen::Matrix3Xd cropLocalPoints(const std::vector<Eigen::Vector3d> &surface_points,
                                            const Eigen::MatrixX4d &boundary)
    {
        std::vector<Eigen::Vector3d> cropped;
        cropped.reserve(surface_points.size());
        for (const Eigen::Vector3d &point : surface_points)
        {
            const Eigen::Vector4d ph(point(0), point(1), point(2), 1.0);
            if ((boundary * ph).maxCoeff() < 0.0)
            {
                cropped.push_back(point);
            }
        }
        return makePointMatrix(cropped);
    }

    inline std::uint64_t stableInputHash(const Eigen::MatrixX4d &boundary,
                                         const Eigen::Matrix3Xd &points,
                                         const Eigen::Vector3d &a,
                                         const Eigen::Vector3d &b)
    {
        std::uint64_t h = region_inflation::hashEigenDense(boundary);
        h = region_inflation::hashEigenDense(points, h);
        h = region_inflation::hashEigenDense(a, h);
        h = region_inflation::hashEigenDense(b, h);
        return h;
    }

    inline bool makePointRegionCase(const voxel_map::VoxelMap &map,
                                    const std::vector<Eigen::Vector3d> &surface_points,
                                    const Eigen::Vector3d &map_low,
                                    const Eigen::Vector3d &map_high,
                                    const std::string &map_id,
                                    const std::string &density,
                                    const std::string &case_id,
                                    const double local_range,
                                    const int max_attempts,
                                    std::uint64_t seed,
                                    RegionCase &region_case)
    {
        std::mt19937_64 rng(seed);
        Eigen::Vector3d q;
        if (!sampleFreePoint(map, map_low, map_high, local_range, max_attempts, rng, q))
        {
            return false;
        }

        region_case = RegionCase();
        region_case.case_id = case_id;
        region_case.map_id = map_id;
        region_case.density = density;
        region_case.seed_type = "point";
        region_case.a = q;
        region_case.b = q;
        region_case.seed_length = 0.0;
        region_case.boundary = localBoundary(q, q, map_low, map_high, local_range);
        region_case.deterministic_seed = seed;
        region_case.global_surface_points = surface_points.size();

        SteadyTimer timer;
        region_case.local_points = cropLocalPoints(surface_points, region_case.boundary);
        region_case.crop_ms = timer.elapsedMs();
        region_case.input_hash = stableInputHash(region_case.boundary,
                                                 region_case.local_points,
                                                 region_case.a,
                                                 region_case.b);
        return true;
    }

    inline bool makeLineRegionCase(const voxel_map::VoxelMap &map,
                                   const std::vector<Eigen::Vector3d> &surface_points,
                                   const Eigen::Vector3d &map_low,
                                   const Eigen::Vector3d &map_high,
                                   const std::string &map_id,
                                   const std::string &density,
                                   const std::string &case_id,
                                   const double local_range,
                                   const double line_length,
                                   const int max_attempts,
                                   std::uint64_t seed,
                                   RegionCase &region_case)
    {
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> normal(0.0, 1.0);
        const double step = std::max(1.0e-3, 0.5 * map.getScale());
        for (int attempt = 0; attempt < max_attempts; ++attempt)
        {
            Eigen::Vector3d midpoint;
            if (!sampleFreePoint(map, map_low, map_high, local_range, max_attempts, rng, midpoint))
            {
                return false;
            }

            Eigen::Vector3d dir(normal(rng), normal(rng), normal(rng));
            if (dir.norm() < 1.0e-9)
            {
                continue;
            }
            dir.normalize();
            const Eigen::Vector3d a = midpoint - 0.5 * line_length * dir;
            const Eigen::Vector3d b = midpoint + 0.5 * line_length * dir;
            if ((a.array() < map_low.array()).any() || (a.array() > map_high.array()).any() ||
                (b.array() < map_low.array()).any() || (b.array() > map_high.array()).any())
            {
                continue;
            }
            if (!lineSegmentFree(map, a, b, step))
            {
                continue;
            }

            region_case = RegionCase();
            region_case.case_id = case_id;
            region_case.map_id = map_id;
            region_case.density = density;
            region_case.seed_type = "line";
            region_case.a = a;
            region_case.b = b;
            region_case.seed_length = line_length;
            region_case.boundary = localBoundary(a, b, map_low, map_high, local_range);
            if (!pointInsideHpoly(region_case.boundary, a, 1.0e-9) ||
                !pointInsideHpoly(region_case.boundary, b, 1.0e-9))
            {
                continue;
            }
            region_case.deterministic_seed = seed;
            region_case.global_surface_points = surface_points.size();

            SteadyTimer timer;
            region_case.local_points = cropLocalPoints(surface_points, region_case.boundary);
            region_case.crop_ms = timer.elapsedMs();
            region_case.input_hash = stableInputHash(region_case.boundary,
                                                     region_case.local_points,
                                                     region_case.a,
                                                     region_case.b);
            return true;
        }
        return false;
    }

    inline bool samplePlanningCase(const voxel_map::VoxelMap &map,
                                   const Eigen::Vector3d &map_low,
                                   const Eigen::Vector3d &map_high,
                                   const double local_range,
                                   const double min_distance,
                                   const double max_distance,
                                   const int max_attempts,
                                   const std::string &case_id,
                                   std::uint64_t seed,
                                   PlanningCase &planning_case)
    {
        std::mt19937_64 rng(seed);
        for (int attempt = 0; attempt < max_attempts; ++attempt)
        {
            Eigen::Vector3d start;
            Eigen::Vector3d goal;
            if (!sampleFreePoint(map, map_low, map_high, local_range, max_attempts, rng, start) ||
                !sampleFreePoint(map, map_low, map_high, local_range, max_attempts, rng, goal))
            {
                return false;
            }
            const double distance = (goal - start).norm();
            if (distance >= min_distance && distance <= max_distance)
            {
                planning_case.case_id = case_id;
                planning_case.start = start;
                planning_case.goal = goal;
                planning_case.deterministic_seed = seed;
                return true;
            }
        }
        return false;
    }
}

#endif
