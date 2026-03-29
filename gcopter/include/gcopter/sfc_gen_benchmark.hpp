/*
    MIT License

    Copyright (c) 2021 Zhepei Wang (wangzhepei@live.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies or substantial portions of the Software.
*/

#ifndef SFC_GEN_BENCHMARK_HPP
#define SFC_GEN_BENCHMARK_HPP

#include "geo_utils.hpp"
#include "firi_mvie_diagnostics.hpp"
#include "firi.hpp"
#include "firi_socp.hpp"
#include "firi_opt.hpp"
#include "firi_nd.hpp"

#include <chrono>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace sfc_gen_benchmark
{
    struct PolytopeMetrics
    {
        double volume = 0.0;
        std::size_t face_count = 0;
        std::size_t vertex_count = 0;
    };

    struct SolverBenchmarkStats
    {
        std::string name;
        std::vector<Eigen::MatrixX4d> hpolys;
        double total_time_ms = 0.0;
        double total_volume = 0.0;
        std::size_t call_count = 0;
        std::size_t failure_count = 0;
        std::size_t total_face_count = 0;
        std::size_t total_vertex_count = 0;
        std::vector<firi_common::MVIEDiagnostic> mvie_records;

        std::size_t polytopeCount() const
        {
            return hpolys.size();
        }

        double averageTimeMs() const
        {
            return call_count == 0 ? 0.0 : total_time_ms / static_cast<double>(call_count);
        }

        double averageVolume() const
        {
            return polytopeCount() == 0 ? 0.0 : total_volume / static_cast<double>(polytopeCount());
        }

        double averageFaceCount() const
        {
            return polytopeCount() == 0 ? 0.0 : static_cast<double>(total_face_count) / static_cast<double>(polytopeCount());
        }

        double averageVertexCount() const
        {
            return polytopeCount() == 0 ? 0.0 : static_cast<double>(total_vertex_count) / static_cast<double>(polytopeCount());
        }

        std::size_t mvieCount() const
        {
            std::size_t count = 0;
            for (const auto &record : mvie_records)
            {
                if (record.optimizer_ran)
                {
                    ++count;
                }
            }
            return count;
        }

        double averageMVIEIterations() const
        {
            std::size_t count = 0;
            long long total = 0;
            for (const auto &record : mvie_records)
            {
                if (record.optimizer_ran)
                {
                    total += record.lbfgs_iterations;
                    ++count;
                }
            }
            if (count == 0)
            {
                return 0.0;
            }
            return static_cast<double>(total) / static_cast<double>(count);
        }

        int maxMVIEIterations() const
        {
            int max_iter = 0;
            for (const auto &record : mvie_records)
            {
                if (record.optimizer_ran)
                {
                    max_iter = std::max(max_iter, record.lbfgs_iterations);
                }
            }
            return max_iter;
        }

        double averageMaxConstraintResidual() const
        {
            std::size_t count = 0;
            double total = 0.0;
            for (const auto &record : mvie_records)
            {
                if (record.optimizer_ran)
                {
                    total += record.max_constraint_residual;
                    ++count;
                }
            }
            if (count == 0)
            {
                return 0.0;
            }
            return total / static_cast<double>(count);
        }

        double worstMaxConstraintResidual() const
        {
            double worst = -std::numeric_limits<double>::infinity();
            for (const auto &record : mvie_records)
            {
                if (record.optimizer_ran)
                {
                    worst = std::max(worst, record.max_constraint_residual);
                }
            }
            return std::isfinite(worst) ? worst : 0.0;
        }

        double averageLogDetL() const
        {
            std::size_t count = 0;
            double total = 0.0;
            for (const auto &record : mvie_records)
            {
                if (record.optimizer_ran)
                {
                    total += record.logdet_l;
                    ++count;
                }
            }
            if (count == 0)
            {
                return 0.0;
            }
            return total / static_cast<double>(count);
        }

        double averageActiveConstraintCount() const
        {
            std::size_t count = 0;
            double total = 0.0;
            for (const auto &record : mvie_records)
            {
                if (record.optimizer_ran)
                {
                    total += static_cast<double>(record.active_constraint_count);
                    ++count;
                }
            }
            if (count == 0)
            {
                return 0.0;
            }
            return total / static_cast<double>(count);
        }

        std::string mvieIterationSequence() const
        {
            std::ostringstream oss;
            oss << "[";
            bool first = true;
            for (std::size_t i = 0; i < mvie_records.size(); ++i)
            {
                if (!mvie_records[i].optimizer_ran)
                {
                    continue;
                }
                if (!first)
                {
                    oss << ", ";
                }
                first = false;
                oss << mvie_records[i].lbfgs_iterations;
            }
            oss << "]";
            return oss.str();
        }

        std::string mvieStatusSequence() const
        {
            std::ostringstream oss;
            oss << "[";
            for (std::size_t i = 0; i < mvie_records.size(); ++i)
            {
                if (i != 0)
                {
                    oss << ", ";
                }
                oss << mvie_records[i].lbfgs_status;
            }
            oss << "]";
            return oss.str();
        }
    };

    struct BenchmarkResult
    {
        SolverBenchmarkStats firi{"firi::firi"};
        SolverBenchmarkStats firi_socp{"firi_socp::firi_socp"};
        SolverBenchmarkStats firi_opt{"firi_opt::firi_opt"};
        SolverBenchmarkStats firi_nd{"firi_nd::firi_opt<3>"};

        void printBenchmarkResult(std::ostream &os = std::cout) const
        {
            const std::ios::fmtflags old_flags = os.flags();
            const std::streamsize old_precision = os.precision();

            os << "\n================ SFC Benchmark ================\n";
            os << std::left
               << std::setw(24) << "Solver"
               << std::setw(8) << "Calls"
               << std::setw(8) << "Polys"
               << std::setw(8) << "Fails"
               << std::setw(14) << "TotalMs"
               << std::setw(14) << "AvgMs"
               << std::setw(14) << "TotalVol"
               << std::setw(14) << "AvgVol"
               << std::setw(14) << "AvgFaces"
               << std::setw(14) << "AvgVerts"
               << '\n';
            os << std::string(132, '-') << '\n';

            os << std::fixed << std::setprecision(3);
            const auto printRow = [&os](const SolverBenchmarkStats &stats)
            {
                os << std::left
                   << std::setw(24) << stats.name
                   << std::setw(8) << stats.call_count
                   << std::setw(8) << stats.polytopeCount()
                   << std::setw(8) << stats.failure_count
                   << std::setw(14) << stats.total_time_ms
                   << std::setw(14) << stats.averageTimeMs()
                   << std::setw(14) << stats.total_volume
                   << std::setw(14) << stats.averageVolume()
                   << std::setw(14) << stats.averageFaceCount()
                   << std::setw(14) << stats.averageVertexCount()
                   << '\n';
            };

            printRow(firi);
            printRow(firi_socp);
            printRow(firi_opt);
            printRow(firi_nd);
            os << "===============================================\n";

            os << "\n---------------- MVIE Summary -----------------\n";
            os << std::left
               << std::setw(24) << "Solver"
               << std::setw(8) << "MVIEs"
               << std::setw(12) << "AvgIter"
               << std::setw(12) << "MaxIter"
               << std::setw(14) << "AvgMaxRes"
               << std::setw(14) << "WorstRes"
               << std::setw(14) << "AvgLogDetL"
               << std::setw(14) << "AvgActive"
               << '\n';
            os << std::string(112, '-') << '\n';

            const auto printMVIESummaryRow = [&os](const SolverBenchmarkStats &stats)
            {
                os << std::left
                   << std::setw(24) << stats.name
                   << std::setw(8) << stats.mvieCount()
                   << std::setw(12) << stats.averageMVIEIterations()
                   << std::setw(12) << stats.maxMVIEIterations()
                   << std::setw(14) << stats.averageMaxConstraintResidual()
                   << std::setw(14) << stats.worstMaxConstraintResidual()
                   << std::setw(14) << stats.averageLogDetL()
                   << std::setw(14) << stats.averageActiveConstraintCount()
                   << '\n';
            };

            printMVIESummaryRow(firi);
            printMVIESummaryRow(firi_socp);
            printMVIESummaryRow(firi_opt);
            printMVIESummaryRow(firi_nd);
            os << "-----------------------------------------------\n";

            os << "\nMVIE optimizer iterations:\n";
            os << std::left << std::setw(24) << firi.name << firi.mvieIterationSequence() << '\n';
            os << std::left << std::setw(24) << firi_socp.name << firi_socp.mvieIterationSequence() << '\n';
            os << std::left << std::setw(24) << firi_opt.name << firi_opt.mvieIterationSequence() << '\n';
            os << std::left << std::setw(24) << firi_nd.name << firi_nd.mvieIterationSequence() << '\n';

            os << "\nMVIE optimizer status codes:\n";
            os << std::left << std::setw(24) << firi.name << firi.mvieStatusSequence() << '\n';
            os << std::left << std::setw(24) << firi_socp.name << firi_socp.mvieStatusSequence() << '\n';
            os << std::left << std::setw(24) << firi_opt.name << firi_opt.mvieStatusSequence() << '\n';
            os << std::left << std::setw(24) << firi_nd.name << firi_nd.mvieStatusSequence() << '\n';

            os.flags(old_flags);
            os.precision(old_precision);
        }
    };

    inline Eigen::Matrix3Xd makePointCloudMatrix(const std::vector<Eigen::Vector3d> &points)
    {
        Eigen::Matrix3Xd pc(3, points.size());
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            pc.col(i) = points[i];
        }
        return pc;
    }

    inline PolytopeMetrics measurePolytope(const Eigen::MatrixX4d &hPoly)
    {
        PolytopeMetrics metrics;
        metrics.face_count = static_cast<std::size_t>(hPoly.rows());

        Eigen::Matrix<double, 3, -1, Eigen::ColMajor> vPoly;
        if (!geo_utils::enumerateVs(hPoly, vPoly))
        {
            return metrics;
        }

        metrics.vertex_count = static_cast<std::size_t>(vPoly.cols());
        if (vPoly.cols() < 4)
        {
            return metrics;
        }

        quickhull::QuickHull<double> tinyQH;
        const auto polyHull = tinyQH.getConvexHull(vPoly.data(), vPoly.cols(), false, true);
        const auto &idxBuffer = polyHull.getIndexBuffer();
        const int numTris = idxBuffer.size() / 3;

        for (int i = 0; i < numTris; ++i)
        {
            const Eigen::Vector3d p1 = vPoly.col(idxBuffer[3 * i + 0]);
            const Eigen::Vector3d p2 = vPoly.col(idxBuffer[3 * i + 1]);
            const Eigen::Vector3d p3 = vPoly.col(idxBuffer[3 * i + 2]);
            metrics.volume += p1.dot(p2.cross(p3));
        }

        metrics.volume = std::abs(metrics.volume) / 6.0;
        return metrics;
    }

    inline double calculateExactPolyhedronVolume(const Eigen::MatrixX4d &hPoly)
    {
        return measurePolytope(hPoly).volume;
    }

    inline void appendPolytopeStats(SolverBenchmarkStats &stats,
                                    const Eigen::MatrixX4d &hPoly)
    {
        stats.hpolys.emplace_back(hPoly);
        const PolytopeMetrics metrics = measurePolytope(hPoly);
        stats.total_volume += metrics.volume;
        stats.total_face_count += metrics.face_count;
        stats.total_vertex_count += metrics.vertex_count;
    }

    inline void appendMVIEDiagnostics(SolverBenchmarkStats &stats,
                                      const std::vector<firi_common::MVIEDiagnostic> &mvie_records)
    {
        stats.mvie_records.insert(stats.mvie_records.end(), mvie_records.begin(), mvie_records.end());
    }

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

    template <typename Solver>
    inline bool solvePolytope(SolverBenchmarkStats &stats,
                              const Solver &solver,
                              const Eigen::Matrix<double, 6, 4> &bd,
                              const Eigen::Matrix3Xd &pc,
                              const Eigen::Vector3d &a,
                              const Eigen::Vector3d &b,
                              Eigen::MatrixX4d &hPoly,
                              std::vector<firi_common::MVIEDiagnostic> &mvie_records,
                              const int iterations,
                              const double epsilon)
    {
        const auto t1 = std::chrono::high_resolution_clock::now();
        const bool success = solver(bd, pc, a, b, hPoly, iterations, epsilon, &mvie_records);
        const auto t2 = std::chrono::high_resolution_clock::now();

        stats.total_time_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
        ++stats.call_count;

        if (!success)
        {
            ++stats.failure_count;
            hPoly.resize(0, 4);
        }

        return success;
    }

    template <typename Solver>
    inline void benchmarkSolverStep(SolverBenchmarkStats &stats,
                                    const Solver &solver,
                                    const Eigen::Matrix<double, 6, 4> &bd,
                                    const Eigen::Matrix3Xd &pc,
                                    const Eigen::Vector3d &a,
                                    const Eigen::Vector3d &b,
                                    const double eps)
    {
        std::vector<firi_common::MVIEDiagnostic> mvie_records;
        Eigen::MatrixX4d hp;
        const bool success = solvePolytope(stats, solver, bd, pc, a, b, hp, mvie_records, 4, eps);
        appendMVIEDiagnostics(stats, mvie_records);
        if (!success)
        {
            return;
        }

        if (needGapPolytope(stats.hpolys, hp, a, eps))
        {
            std::vector<firi_common::MVIEDiagnostic> gap_mvie_records;
            Eigen::MatrixX4d gap;
            const bool gap_success = solvePolytope(stats, solver, bd, pc, a, a, gap, gap_mvie_records, 1, eps);
            appendMVIEDiagnostics(stats, gap_mvie_records);
            if (gap_success)
            {
                appendPolytopeStats(stats, gap);
            }
        }

        appendPolytopeStats(stats, hp);
    }

    inline BenchmarkResult convexCover(const std::vector<Eigen::Vector3d> &path,
                                       const std::vector<Eigen::Vector3d> &points,
                                       const Eigen::Vector3d &lowCorner,
                                       const Eigen::Vector3d &highCorner,
                                       const double &progress,
                                       const double &range,
                                       const double eps = 1.0e-6)
    {
        BenchmarkResult result;
        if (path.size() < 2)
        {
            return result;
        }

        const int n = path.size();
        Eigen::Matrix<double, 6, 4> bd = Eigen::Matrix<double, 6, 4>::Zero();
        bd(0, 0) = 1.0;
        bd(1, 0) = -1.0;
        bd(2, 1) = 1.0;
        bd(3, 1) = -1.0;
        bd(4, 2) = 1.0;
        bd(5, 2) = -1.0;

        Eigen::Vector3d a;
        Eigen::Vector3d b = path.front();
        std::vector<Eigen::Vector3d> valid_pc;
        valid_pc.reserve(points.size());

        const auto run_firi = [](const auto &bd_in,
                                 const auto &pc_in,
                                 const auto &a_in,
                                 const auto &b_in,
                                 auto &hpoly_out,
                                 const int iterations,
                                 const double epsilon,
                                 auto *mvie_records)
        {
            return firi::firi(bd_in, pc_in, a_in, b_in, hpoly_out, iterations, epsilon, mvie_records);
        };
        const auto run_firi_opt = [](const auto &bd_in,
                                     const auto &pc_in,
                                     const auto &a_in,
                                     const auto &b_in,
                                     auto &hpoly_out,
                                     const int iterations,
                                     const double epsilon,
                                     auto *mvie_records)
        {
            return firi_opt::firi_opt(bd_in, pc_in, a_in, b_in, hpoly_out, iterations, epsilon, mvie_records);
        };
        const auto run_firi_socp = [](const auto &bd_in,
                                      const auto &pc_in,
                                      const auto &a_in,
                                      const auto &b_in,
                                      auto &hpoly_out,
                                      const int iterations,
                                      const double epsilon,
                                      auto *mvie_records)
        {
            return firi_socp::firi_socp(bd_in, pc_in, a_in, b_in, hpoly_out, iterations, epsilon, mvie_records);
        };
        const auto run_firi_nd = [](const auto &bd_in,
                                    const auto &pc_in,
                                    const auto &a_in,
                                    const auto &b_in,
                                    auto &hpoly_out,
                                    const int iterations,
                                    const double epsilon,
                                    auto *mvie_records)
        {
            return firi_nd::firi_opt<3>(bd_in, pc_in, a_in, b_in, hpoly_out, iterations, epsilon, mvie_records);
        };

        for (int i = 1; i < n;)
        {
            a = b;
            if ((a - path[i]).norm() > progress)
            {
                b = (path[i] - a).normalized() * progress + a;
            }
            else
            {
                b = path[i];
                ++i;
            }

            bd(0, 3) = -std::min(std::max(a(0), b(0)) + range, highCorner(0));
            bd(1, 3) = +std::max(std::min(a(0), b(0)) - range, lowCorner(0));
            bd(2, 3) = -std::min(std::max(a(1), b(1)) + range, highCorner(1));
            bd(3, 3) = +std::max(std::min(a(1), b(1)) - range, lowCorner(1));
            bd(4, 3) = -std::min(std::max(a(2), b(2)) + range, highCorner(2));
            bd(5, 3) = +std::max(std::min(a(2), b(2)) - range, lowCorner(2));

            valid_pc.clear();
            for (const Eigen::Vector3d &p : points)
            {
                if ((bd.leftCols<3>() * p + bd.rightCols<1>()).maxCoeff() < 0.0)
                {
                    valid_pc.emplace_back(p);
                }
            }
            const Eigen::Matrix3Xd pc = makePointCloudMatrix(valid_pc);

            benchmarkSolverStep(result.firi, run_firi, bd, pc, a, b, eps);
            benchmarkSolverStep(result.firi_socp, run_firi_socp, bd, pc, a, b, eps);
            benchmarkSolverStep(result.firi_opt, run_firi_opt, bd, pc, a, b, eps);
            benchmarkSolverStep(result.firi_nd, run_firi_nd, bd, pc, a, b, eps);
        }

        return result;
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
            const Eigen::MatrixX4d headPoly = htemp.front();
            htemp.insert(htemp.begin(), headPoly);
        }
        hpolys.clear();

        const int M = htemp.size();
        bool overlap;
        std::deque<int> indices;
        indices.push_front(M - 1);
        for (int i = M - 1; i >= 0; i--)
        {
            for (int j = 0; j < i; j++)
            {
                if (j < i - 1)
                {
                    overlap = geo_utils::overlap(htemp[i], htemp[j], 0.01);
                }
                else
                {
                    overlap = true;
                }
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
}

#endif
