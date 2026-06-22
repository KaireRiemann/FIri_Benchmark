#ifndef GCOPTER_BENCHMARK_TYPES_HPP
#define GCOPTER_BENCHMARK_TYPES_HPP

#include "gcopter/geo_utils.hpp"
#include "gcopter/quickhull.hpp"
#include "gcopter/region_inflation/region_types.hpp"

#include <Eigen/Eigen>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace firi_benchmark
{
    struct RegionCase
    {
        std::string case_id;
        std::string map_id;
        std::string density;
        std::string seed_type;
        Eigen::Vector3d a = Eigen::Vector3d::Zero();
        Eigen::Vector3d b = Eigen::Vector3d::Zero();
        Eigen::MatrixX4d boundary;
        Eigen::Matrix3Xd local_points;
        double seed_length = 0.0;
        double crop_ms = 0.0;
        std::uint64_t deterministic_seed = 0;
        std::uint64_t input_hash = 0;
        std::size_t global_surface_points = 0;
    };

    struct PathSeedCase
    {
        Eigen::Vector3d a = Eigen::Vector3d::Zero();
        Eigen::Vector3d b = Eigen::Vector3d::Zero();
        Eigen::MatrixX4d boundary;
        Eigen::Matrix3Xd local_points;
        double crop_ms = 0.0;
        std::uint64_t input_hash = 0;
    };

    struct PlanningCase
    {
        std::string case_id;
        Eigen::Vector3d start = Eigen::Vector3d::Zero();
        Eigen::Vector3d goal = Eigen::Vector3d::Zero();
        std::uint64_t deterministic_seed = 0;
    };

    struct PolytopeMetrics
    {
        double volume = 0.0;
        std::size_t face_count = 0;
        std::size_t vertex_count = 0;
    };

    struct LocalPointStats
    {
        double mean = 0.0;
        double stddev = 0.0;
        double median = 0.0;
        double p95 = 0.0;
    };

    struct CorridorTrialStats
    {
        bool success = false;
        std::string failure_reason;
        int candidate_seed_count = 0;
        int region_call_count = 0;
        int gap_region_count = 0;
        int regions_before_shortcut = 0;
        int regions_after_shortcut = 0;
        std::size_t local_points_total = 0;
        double seed_preparation_ms = 0.0;
        double crop_ms = 0.0;
        double region_core_ms = 0.0;
        double gap_region_ms = 0.0;
        double shortcut_ms = 0.0;
        double overlap_check_ms = 0.0;
        double corridor_total_ms = 0.0;
        int faces_total = 0;
        double faces_mean = 0.0;
        double volume_sum = 0.0;
        double volume_mean = 0.0;
        int seed_inclusion_failures = 0;
        int overlap_failures = 0;
        int interior_obstacle_count = 0;
        std::vector<Eigen::MatrixX4d> hpolys;
    };

    inline const std::vector<std::string> &regionTrialHeader()
    {
        static const std::vector<std::string> header = {
            "run_id", "map_id", "density", "case_id", "seed_type", "seed_length",
            "seed_ax", "seed_ay", "seed_az", "seed_bx", "seed_by", "seed_bz",
            "method", "repeat_id", "order_index", "global_surface_points", "local_points",
            "input_hash", "success", "failure_stage", "outer_build_count", "mvie_call_count",
            "mvie_iterations_total", "mvie_objective_evals_total", "crop_ms",
            "constraint_prepare_ms", "constraint_build_ms", "mvie_ms", "certification_ms",
            "repair_ms", "region_core_ms", "region_online_ms", "polytope_faces",
            "polytope_vertices", "polytope_volume", "seed_included", "interior_obstacle_count",
            "max_mvie_residual", "final_logdet"};
        return header;
    }

    inline const std::vector<std::string> &mvieReplayHeader()
    {
        static const std::vector<std::string> header = {
            "run_id", "map_id", "density", "region_case_id", "outer_iteration", "seed_type",
            "seed_length", "facet_count", "local_point_count", "input_hash", "warm_start_hash",
            "solver", "repeat_id", "order_index", "success", "status", "iterations",
            "objective_evaluations", "solve_ms", "logdet_l", "max_mu",
            "max_constraint_residual", "active_constraint_count", "finite_output",
            "positive_diagonal", "failure_reason"};
        return header;
    }

    inline const std::vector<std::string> &corridorTrialHeader()
    {
        static const std::vector<std::string> header = {
            "run_id", "map_id", "density", "planning_case_id", "corridor_mode", "method",
            "repeat_id", "route_hash", "route_point_count", "route_length",
            "candidate_seed_count", "region_call_count", "gap_region_count",
            "regions_before_shortcut", "regions_after_shortcut", "local_points_total",
            "crop_ms", "region_core_ms", "gap_region_ms", "shortcut_ms", "overlap_check_ms",
            "corridor_total_ms", "faces_total", "faces_mean", "volume_sum", "volume_mean",
            "seed_inclusion_failures", "overlap_failures", "interior_obstacle_count",
            "success", "failure_reason"};
        return header;
    }

    inline const std::vector<std::string> &planningTrialHeader()
    {
        static const std::vector<std::string> header = {
            "run_id", "map_id", "density", "planning_case_id", "method", "repeat_id",
            "method_order", "start_x", "start_y", "start_z", "goal_x", "goal_y", "goal_z",
            "route_hash", "route_points", "route_length", "path_success",
            "path_search_ms_shared", "surface_extract_ms_shared", "corridor_success",
            "regions", "corridor_total_ms", "trajectory_setup_success", "trajectory_setup_ms",
            "trajectory_optimize_success", "trajectory_optimize_ms", "trajectory_cost",
            "trajectory_duration", "trajectory_pieces", "planning_backend_ms",
            "end_to_end_ms", "sampled_collision_count", "trajectory_collision_free",
            "success", "failure_stage", "failure_reason"};
        return header;
    }

    inline Eigen::Matrix<double, 6, 4> makeBoxBoundary(const Eigen::Vector3d &low,
                                                       const Eigen::Vector3d &high)
    {
        Eigen::Matrix<double, 6, 4> bd = Eigen::Matrix<double, 6, 4>::Zero();
        bd(0, 0) = 1.0;
        bd(1, 0) = -1.0;
        bd(2, 1) = 1.0;
        bd(3, 1) = -1.0;
        bd(4, 2) = 1.0;
        bd(5, 2) = -1.0;
        bd(0, 3) = -high.x();
        bd(1, 3) = low.x();
        bd(2, 3) = -high.y();
        bd(3, 3) = low.y();
        bd(4, 3) = -high.z();
        bd(5, 3) = low.z();
        return bd;
    }

    inline Eigen::Matrix3Xd makePointMatrix(const std::vector<Eigen::Vector3d> &points)
    {
        Eigen::Matrix3Xd pc(3, points.size());
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            pc.col(i) = points[i];
        }
        return pc;
    }

    inline double routeLength(const std::vector<Eigen::Vector3d> &route)
    {
        double length = 0.0;
        for (std::size_t i = 1; i < route.size(); ++i)
        {
            length += (route[i] - route[i - 1]).norm();
        }
        return length;
    }

    inline bool pointInsideHpoly(const Eigen::MatrixX4d &hpoly,
                                 const Eigen::Vector3d &point,
                                 const double tol)
    {
        if (hpoly.rows() == 0)
        {
            return false;
        }
        const Eigen::Vector4d ph(point(0), point(1), point(2), 1.0);
        return (hpoly * ph).maxCoeff() <= tol;
    }

    inline bool seedIncluded(const Eigen::MatrixX4d &hpoly,
                             const Eigen::Vector3d &a,
                             const Eigen::Vector3d &b,
                             const double tol)
    {
        return pointInsideHpoly(hpoly, a, tol) && pointInsideHpoly(hpoly, b, tol);
    }

    inline int countInteriorObstacles(const Eigen::MatrixX4d &hpoly,
                                      const Eigen::Matrix3Xd &points,
                                      const double strict_tol)
    {
        int count = 0;
        for (int i = 0; i < points.cols(); ++i)
        {
            const Eigen::Vector4d ph(points(0, i), points(1, i), points(2, i), 1.0);
            if ((hpoly * ph).maxCoeff() < -strict_tol)
            {
                ++count;
            }
        }
        return count;
    }

    inline Eigen::MatrixX4d normalizedHpolyForMeasurement(const Eigen::MatrixX4d &hpoly)
    {
        std::vector<int> rows;
        rows.reserve(static_cast<std::size_t>(hpoly.rows()));
        for (int i = 0; i < hpoly.rows(); ++i)
        {
            const double norm = hpoly.block<1, 3>(i, 0).norm();
            if (std::isfinite(norm) && norm > 1.0e-10 && hpoly.row(i).array().isFinite().all())
            {
                rows.push_back(i);
            }
        }

        Eigen::MatrixX4d normalized(rows.size(), 4);
        for (std::size_t r = 0; r < rows.size(); ++r)
        {
            const int i = rows[r];
            normalized.row(static_cast<int>(r)) = hpoly.row(i) / hpoly.block<1, 3>(i, 0).norm();
        }
        return normalized;
    }

    inline Eigen::MatrixX4d stackHpolys(const Eigen::MatrixX4d &a,
                                        const Eigen::MatrixX4d &b)
    {
        Eigen::MatrixX4d combined(a.rows() + b.rows(), 4);
        if (a.rows() > 0)
        {
            combined.topRows(a.rows()) = a;
        }
        if (b.rows() > 0)
        {
            combined.bottomRows(b.rows()) = b;
        }
        return combined;
    }

    inline PolytopeMetrics measurePolytope(const Eigen::MatrixX4d &hpoly)
    {
        PolytopeMetrics metrics;
        const Eigen::MatrixX4d H = normalizedHpolyForMeasurement(hpoly);
        metrics.face_count = static_cast<std::size_t>(H.rows());
        if (H.rows() < 4)
        {
            return metrics;
        }

        std::vector<Eigen::Vector3d> candidates;
        const double solve_tol = 1.0e-9;
        const double feasible_tol = 1.0e-7;
        for (int i = 0; i < H.rows(); ++i)
        {
            for (int j = i + 1; j < H.rows(); ++j)
            {
                for (int k = j + 1; k < H.rows(); ++k)
                {
                    Eigen::Matrix3d A;
                    A.row(0) = H.block<1, 3>(i, 0);
                    A.row(1) = H.block<1, 3>(j, 0);
                    A.row(2) = H.block<1, 3>(k, 0);
                    if (std::abs(A.determinant()) < solve_tol)
                    {
                        continue;
                    }
                    const Eigen::Vector3d b(-H(i, 3), -H(j, 3), -H(k, 3));
                    const Eigen::Vector3d v = A.fullPivLu().solve(b);
                    if (!v.array().isFinite().all())
                    {
                        continue;
                    }
                    const Eigen::Vector4d vh(v(0), v(1), v(2), 1.0);
                    if ((H * vh).maxCoeff() <= feasible_tol)
                    {
                        bool duplicate = false;
                        for (const Eigen::Vector3d &existing : candidates)
                        {
                            if ((existing - v).norm() < 1.0e-6)
                            {
                                duplicate = true;
                                break;
                            }
                        }
                        if (!duplicate)
                        {
                            candidates.push_back(v);
                        }
                    }
                }
            }
        }
        Eigen::Matrix<double, 3, -1, Eigen::ColMajor> vertices = makePointMatrix(candidates);

        metrics.vertex_count = static_cast<std::size_t>(vertices.cols());
        if (vertices.cols() < 4)
        {
            return metrics;
        }

        quickhull::QuickHull<double> qh;
        const auto hull = qh.getConvexHull(vertices.data(), vertices.cols(), false, true);
        const auto &idx = hull.getIndexBuffer();
        const int tris = idx.size() / 3;
        const Eigen::Vector3d centroid = vertices.rowwise().mean();
        for (int i = 0; i < tris; ++i)
        {
            const Eigen::Vector3d p1 = vertices.col(idx[3 * i + 0]);
            const Eigen::Vector3d p2 = vertices.col(idx[3 * i + 1]);
            const Eigen::Vector3d p3 = vertices.col(idx[3 * i + 2]);
            metrics.volume += std::abs((p1 - centroid).dot((p2 - centroid).cross(p3 - centroid))) / 6.0;
        }
        return metrics;
    }

    inline PolytopeMetrics measureBoundedPolytope(const Eigen::MatrixX4d &hpoly,
                                                  const Eigen::MatrixX4d &boundary)
    {
        return measurePolytope(stackHpolys(hpoly, boundary));
    }

    inline LocalPointStats summarizeLocalPointCounts(std::vector<double> counts)
    {
        LocalPointStats stats;
        if (counts.empty())
        {
            return stats;
        }
        const double sum = std::accumulate(counts.begin(), counts.end(), 0.0);
        stats.mean = sum / static_cast<double>(counts.size());
        double var = 0.0;
        for (const double value : counts)
        {
            const double diff = value - stats.mean;
            var += diff * diff;
        }
        stats.stddev = std::sqrt(var / static_cast<double>(counts.size()));
        std::sort(counts.begin(), counts.end());
        stats.median = counts[counts.size() / 2];
        const std::size_t p95_idx =
            std::min<std::size_t>(counts.size() - 1, static_cast<std::size_t>(std::ceil(0.95 * counts.size())) - 1);
        stats.p95 = counts[p95_idx];
        return stats;
    }
}

#endif
