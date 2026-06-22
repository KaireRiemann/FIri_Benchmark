#ifndef GCOPTER_REGION_INFLATION_FIRI_CONSTRAINT_BUILDER_HPP
#define GCOPTER_REGION_INFLATION_FIRI_CONSTRAINT_BUILDER_HPP

#include "gcopter/region_inflation/constraint_builder.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>

namespace region_inflation
{
    class FullFiriConstraintBuilder final : public ConstraintBuilder3D
    {
    public:
        explicit FullFiriConstraintBuilder(const double epsilon = 1.0e-6)
            : epsilon_(epsilon) {}

        bool prepare(const RegionInput &input,
                     ConstraintBuildStats &stats) override
        {
            const auto start = std::chrono::steady_clock::now();
            stats.full_point_count = static_cast<std::size_t>(input.point_cloud.cols());
            stats.working_point_count = stats.full_point_count;
            stats.prepare_ms += elapsedMs(start);
            return true;
        }

        bool build(const RegionInput &input,
                   const Ellipsoid3D &current,
                   Eigen::MatrixX4d &hpoly,
                   ConstraintBuildStats &stats) override
        {
            const auto start = std::chrono::steady_clock::now();
            const Eigen::MatrixX4d &bd = input.boundary;
            const Eigen::Matrix3Xd &pc = input.point_cloud;
            const Eigen::Vector3d &a = input.a;
            const Eigen::Vector3d &b = input.b;

            const int M = bd.rows();
            const int N = pc.cols();
            if (M == 0)
            {
                hpoly.resize(0, 4);
                stats.build_ms += elapsedMs(start);
                return false;
            }

            const Eigen::Matrix3d forward = current.radii.cwiseInverse().asDiagonal() * current.R.transpose();
            const Eigen::Matrix3d backward = current.R * current.radii.asDiagonal();
            const Eigen::MatrixX3d forwardB = bd.leftCols<3>() * backward;
            const Eigen::VectorXd forwardD = bd.rightCols<1>() + bd.leftCols<3>() * current.center;
            const Eigen::Matrix3Xd forwardPC = forward * (pc.colwise() - current.center);
            const Eigen::Vector3d fwd_a = forward * (a - current.center);
            const Eigen::Vector3d fwd_b = forward * (b - current.center);

            const Eigen::VectorXd distDs = forwardD.cwiseAbs().cwiseQuotient(forwardB.rowwise().norm());
            Eigen::MatrixX4d tangents(N, 4);
            Eigen::VectorXd distRs(N);

            for (int i = 0; i < N; ++i)
            {
                distRs(i) = forwardPC.col(i).norm();
                if (!(distRs(i) > 0.0) || !std::isfinite(distRs(i)))
                {
                    distRs(i) = std::numeric_limits<double>::epsilon();
                }
                tangents(i, 3) = -distRs(i);
                tangents.block<1, 3>(i, 0) = forwardPC.col(i).transpose() / distRs(i);
                if (tangents.block<1, 3>(i, 0).dot(fwd_a) + tangents(i, 3) > epsilon_)
                {
                    const Eigen::Vector3d delta = forwardPC.col(i) - fwd_a;
                    tangents.block<1, 3>(i, 0) =
                        fwd_a - (delta.dot(fwd_a) / delta.squaredNorm()) * delta;
                    distRs(i) = tangents.block<1, 3>(i, 0).norm();
                    tangents(i, 3) = -distRs(i);
                    tangents.block<1, 3>(i, 0) /= distRs(i);
                }
                if (tangents.block<1, 3>(i, 0).dot(fwd_b) + tangents(i, 3) > epsilon_)
                {
                    const Eigen::Vector3d delta = forwardPC.col(i) - fwd_b;
                    tangents.block<1, 3>(i, 0) =
                        fwd_b - (delta.dot(fwd_b) / delta.squaredNorm()) * delta;
                    distRs(i) = tangents.block<1, 3>(i, 0).norm();
                    tangents(i, 3) = -distRs(i);
                    tangents.block<1, 3>(i, 0) /= distRs(i);
                }
                if (tangents.block<1, 3>(i, 0).dot(fwd_a) + tangents(i, 3) > epsilon_)
                {
                    tangents.block<1, 3>(i, 0) =
                        (fwd_a - forwardPC.col(i)).cross(fwd_b - forwardPC.col(i)).normalized();
                    tangents(i, 3) = -tangents.block<1, 3>(i, 0).dot(fwd_a);
                    tangents.row(i) *= tangents(i, 3) > 0.0 ? -1.0 : 1.0;
                }
            }

            Eigen::Matrix<std::uint8_t, -1, 1> bdFlags =
                Eigen::Matrix<std::uint8_t, -1, 1>::Constant(M, 1);
            Eigen::Matrix<std::uint8_t, -1, 1> pcFlags =
                Eigen::Matrix<std::uint8_t, -1, 1>::Constant(N, 1);

            Eigen::MatrixX4d forwardH(M + N, 4);
            int nH = 0;
            bool completed = false;
            int bdMinId = 0;
            int pcMinId = 0;
            double minSqrD = distDs.minCoeff(&bdMinId);
            double minSqrR = INFINITY;
            if (distRs.size() != 0)
            {
                minSqrR = distRs.minCoeff(&pcMinId);
            }

            for (int i = 0; !completed && i < (M + N); ++i)
            {
                if (minSqrD < minSqrR)
                {
                    forwardH.block<1, 3>(nH, 0) = forwardB.row(bdMinId);
                    forwardH(nH, 3) = forwardD(bdMinId);
                    bdFlags(bdMinId) = 0;
                }
                else
                {
                    forwardH.row(nH) = tangents.row(pcMinId);
                    pcFlags(pcMinId) = 0;
                }

                completed = true;
                minSqrD = INFINITY;
                for (int j = 0; j < M; ++j)
                {
                    if (bdFlags(j))
                    {
                        completed = false;
                        if (minSqrD > distDs(j))
                        {
                            bdMinId = j;
                            minSqrD = distDs(j);
                        }
                    }
                }
                minSqrR = INFINITY;
                for (int j = 0; j < N; ++j)
                {
                    if (pcFlags(j))
                    {
                        if (forwardH.block<1, 3>(nH, 0).dot(forwardPC.col(j)) + forwardH(nH, 3) > -epsilon_)
                        {
                            pcFlags(j) = 0;
                        }
                        else
                        {
                            completed = false;
                            if (minSqrR > distRs(j))
                            {
                                pcMinId = j;
                                minSqrR = distRs(j);
                            }
                        }
                    }
                }
                ++nH;
            }

            hpoly.resize(nH, 4);
            for (int i = 0; i < nH; ++i)
            {
                hpoly.block<1, 3>(i, 0) = forwardH.block<1, 3>(i, 0) * forward;
                hpoly(i, 3) = forwardH(i, 3) - hpoly.block<1, 3>(i, 0).dot(current.center);
            }

            stats.generated_plane_count = nH;
            stats.build_ms += elapsedMs(start);
            return nH > 0;
        }

        bool finalize(const RegionInput &,
                      const Ellipsoid3D &,
                      Eigen::MatrixX4d &,
                      ConstraintBuildStats &stats) override
        {
            const auto start = std::chrono::steady_clock::now();
            stats.certification_ms += elapsedMs(start);
            stats.repair_ms += 0.0;
            stats.repair_count += 0;
            return true;
        }

    private:
        double epsilon_;

        static double elapsedMs(const std::chrono::steady_clock::time_point &start)
        {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
    };
}

#endif
