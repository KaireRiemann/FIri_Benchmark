/*
    MIT License

    Copyright (c) 2021 Zhepei Wang (wangzhepei@live.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#ifndef FIRI_SOCP_HPP
#define FIRI_SOCP_HPP

#include "firi_mvie_diagnostics.hpp"
#include "sdqp.hpp"
#include "sdlp.hpp"

#include <Eigen/Eigen>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace firi_socp
{
    constexpr double kAffineScalingTau = 0.95;
    constexpr double kAffineScalingMinStep = 1.0e-6;
    constexpr double kAffineScalingFeasTol = 1.0e-12;
    constexpr double kAffineScalingRelTol = 1.0e-7;
    constexpr int kAffineScalingMaxIterations = 64;
    constexpr int kAffineScalingSuccess = 0;
    constexpr int kAffineScalingInfeasible = -1;
    constexpr int kAffineScalingNumerical = -2;
    constexpr int kAffineScalingLineSearchFail = -3;
    constexpr int kAffineScalingMaxIter = 1;
    constexpr int kPaperCheckNoClearance = -10;
    constexpr int kPaperCheckBoundaryIntersectsUnitBall = -11;
    constexpr int kPaperCheckObstacleIntersectsUnitBall = -12;
    constexpr int kPaperCheckRestrictiveHalfspaceFail = -13;
    constexpr int kPaperCheckRestrictiveHalfspaceInvalid = -14;
    constexpr int kPaperCheckMVIEWarmStartFail = -15;

    struct SOCPConstraint
    {
        Eigen::VectorXd c;
        double d = 0.0;
        Eigen::MatrixXd A;
        Eigen::MatrixXd gram;
    };

    constexpr int kSocpStateSize = 12;
    constexpr int kCenterOffset = 6;
    constexpr int kObjectiveIndex = 9;
    constexpr int kGeomAux1Index = 10;
    constexpr int kGeomAux2Index = 11;

    inline Eigen::Matrix3d vectorToLowerTriangular(const Eigen::VectorXd &x)
    {
        Eigen::Matrix3d L = Eigen::Matrix3d::Zero();
        L(0, 0) = x(0);
        L(1, 0) = x(1);
        L(1, 1) = x(2);
        L(2, 0) = x(3);
        L(2, 1) = x(4);
        L(2, 2) = x(5);
        return L;
    }

    inline double coneSlack(const SOCPConstraint &constraint,
                            const Eigen::VectorXd &x)
    {
        const double s = constraint.c.dot(x) + constraint.d;
        const Eigen::VectorXd y = constraint.A.transpose() * x;
        return s * s - y.squaredNorm();
    }

    inline bool strictlyFeasible(const std::vector<SOCPConstraint> &constraints,
                                 const Eigen::VectorXd &x,
                                 const double tol = kAffineScalingFeasTol)
    {
        for (const auto &constraint : constraints)
        {
            if (!(coneSlack(constraint, x) > tol))
            {
                return false;
            }
        }
        return true;
    }

    inline void setFailureDiagnostic(const int status,
                                     firi_common::MVIEDiagnostic *diagnostic)
    {
        if (diagnostic != nullptr)
        {
            *diagnostic = firi_common::MVIEDiagnostic();
            diagnostic->optimizer_ran = false;
            diagnostic->lbfgs_status = status;
        }
    }

    inline double boundaryClearance(const Eigen::MatrixX4d &bd,
                                    const Eigen::Vector3d &point)
    {
        double clearance = INFINITY;
        for (int i = 0; i < bd.rows(); ++i)
        {
            const Eigen::Vector3d normal = bd.row(i).head<3>();
            const double norm = normal.norm();
            if (!(norm > kAffineScalingFeasTol))
            {
                continue;
            }
            const double signed_distance = -(normal.dot(point) + bd(i, 3)) / norm;
            clearance = std::min(clearance, signed_distance);
        }
        return clearance;
    }

    inline double obstacleClearance(const Eigen::Matrix3Xd &pc,
                                    const Eigen::Vector3d &point)
    {
        if (pc.cols() == 0)
        {
            return INFINITY;
        }

        double clearance = INFINITY;
        for (int i = 0; i < pc.cols(); ++i)
        {
            clearance = std::min(clearance, (pc.col(i) - point).norm());
        }
        return clearance;
    }

    inline bool initializeObstacleFreeEllipsoid(const Eigen::MatrixX4d &bd,
                                                const Eigen::Matrix3Xd &pc,
                                                const Eigen::Vector3d &a,
                                                const Eigen::Vector3d &b,
                                                Eigen::Matrix3d &R,
                                                Eigen::Vector3d &p,
                                                Eigen::Vector3d &r)
    {
        R.setIdentity();
        p = 0.5 * (a + b);

        const double boundary_margin = boundaryClearance(bd, p);
        const double obstacle_margin = obstacleClearance(pc, p);
        const double clearance = std::min(boundary_margin, obstacle_margin);
        if (!std::isfinite(clearance) || !(clearance > 10.0 * kAffineScalingFeasTol))
        {
            return false;
        }

        const double radius = std::max(0.45 * clearance, 100.0 * kAffineScalingFeasTol);
        r = Eigen::Vector3d::Constant(radius);
        return true;
    }

    inline bool transformedProblemSatisfiesPaperConditions(const Eigen::VectorXd &distDs,
                                                           const std::vector<Eigen::Vector3d> &forwardPC,
                                                           const double epsilon)
    {
        const double tol = std::max(10.0 * epsilon, 1.0e-6);
        if (distDs.size() > 0 && distDs.minCoeff() < 1.0 - tol)
        {
            return false;
        }

        for (const Eigen::Vector3d &point : forwardPC)
        {
            if (point.norm() < 1.0 - tol)
            {
                return false;
            }
        }
        return true;
    }

    inline bool validateRestrictiveHalfspace(const std::vector<Eigen::Vector3d> &seed_vertices,
                                             const Eigen::Vector3d &obstacle,
                                             const Eigen::Vector3d &a,
                                             const double epsilon)
    {
        const double rhs = a.squaredNorm();
        if (!std::isfinite(rhs) || !(rhs > kAffineScalingFeasTol))
        {
            return false;
        }
        if (std::sqrt(rhs) < 1.0 - epsilon)
        {
            return false;
        }
        for (const Eigen::Vector3d &vertex : seed_vertices)
        {
            if (vertex.dot(a) > rhs + epsilon)
            {
                return false;
            }
        }
        if (obstacle.dot(a) < rhs - epsilon)
        {
            return false;
        }
        return true;
    }

    inline void initializeGeometricMeanAuxiliaries(Eigen::VectorXd &x)
    {
        const double d1 = std::max<double>(x(0), kAffineScalingFeasTol);
        const double d2 = std::max<double>(x(2), kAffineScalingFeasTol);
        const double d3 = std::max<double>(x(5), kAffineScalingFeasTol);
        const double product = d1 * d2 * d3;
        const double u1 = 0.99 * std::sqrt(d1 * d2);

        double t = 0.95 * std::cbrt(std::max<double>(product, kAffineScalingFeasTol));
        double u2 = 0.0;
        bool initialized = false;
        for (int k = 0; k < 16; ++k)
        {
            const double upper = 0.99 * std::sqrt(std::max<double>(d3 * t, kAffineScalingFeasTol));
            const double lower = 1.01 * t * t / std::max<double>(u1, kAffineScalingFeasTol);
            if (lower < upper)
            {
                u2 = 0.5 * (lower + upper);
                initialized = true;
                break;
            }
            t *= 0.8;
        }

        if (!initialized)
        {
            t = 0.5 * std::cbrt(std::max<double>(product, kAffineScalingFeasTol));
            u2 = std::sqrt(std::max<double>(d3 * t, kAffineScalingFeasTol));
        }

        x(kObjectiveIndex) = t;
        x(kGeomAux1Index) = u1;
        x(kGeomAux2Index) = u2;
    }

    inline bool makeWarmStartStrictlyFeasible(const std::vector<SOCPConstraint> &constraints,
                                              Eigen::VectorXd &x)
    {
        if (strictlyFeasible(constraints, x))
        {
            return true;
        }

        const Eigen::VectorXd base = x;
        for (int k = 0; k < 16; ++k)
        {
            const double scale = std::pow(0.5, static_cast<double>(k + 1));
            x = base;
            x.head<6>() *= scale;
            x.segment<3>(6) *= scale;
            initializeGeometricMeanAuxiliaries(x);
            if (strictlyFeasible(constraints, x))
            {
                return true;
            }
        }

        x = base;
        return false;
    }

    inline bool affineScalingSOCP(const std::vector<SOCPConstraint> &constraints,
                                  Eigen::VectorXd &x,
                                  firi_common::MVIEDiagnostic *diagnostic = nullptr)
    {
        if (diagnostic != nullptr)
        {
            *diagnostic = firi_common::MVIEDiagnostic();
        }

        if (!strictlyFeasible(constraints, x))
        {
            if (diagnostic != nullptr)
            {
                diagnostic->lbfgs_status = kAffineScalingInfeasible;
            }
            return false;
        }

        Eigen::VectorXd cK = Eigen::VectorXd::Zero(x.size());
        cK(kObjectiveIndex) = -1.0;

        double prev_t = x(kObjectiveIndex);
        for (int iter = 1; iter <= kAffineScalingMaxIterations; ++iter)
        {
            Eigen::MatrixXd H = Eigen::MatrixXd::Zero(x.size(), x.size());
            for (const auto &constraint : constraints)
            {
                const double s = constraint.c.dot(x) + constraint.d;
                const Eigen::VectorXd y = constraint.A.transpose() * x;
                const double f = s * s - y.squaredNorm();
                if (!(f > kAffineScalingFeasTol))
                {
                    if (diagnostic != nullptr)
                    {
                        diagnostic->lbfgs_iterations = iter - 1;
                        diagnostic->lbfgs_status = kAffineScalingInfeasible;
                    }
                    return false;
                }

                const Eigen::VectorXd grad = 2.0 * s * constraint.c - 2.0 * constraint.A * y;
                const Eigen::MatrixXd hess = 2.0 * constraint.c * constraint.c.transpose() - 2.0 * constraint.gram;
                H.noalias() += (grad * grad.transpose()) / (f * f) - hess / f;
            }

            const Eigen::LDLT<Eigen::MatrixXd> ldlt(H);
            if (ldlt.info() != Eigen::Success)
            {
                if (diagnostic != nullptr)
                {
                    diagnostic->lbfgs_iterations = iter - 1;
                    diagnostic->lbfgs_status = kAffineScalingNumerical;
                }
                return false;
            }

            const Eigen::VectorXd h_inv_ck = ldlt.solve(cK);
            const double denom_sq = cK.dot(h_inv_ck);
            if (!(denom_sq > 0.0) || !std::isfinite(denom_sq))
            {
                if (diagnostic != nullptr)
                {
                    diagnostic->lbfgs_iterations = iter - 1;
                    diagnostic->lbfgs_status = kAffineScalingNumerical;
                }
                return false;
            }

            const Eigen::VectorXd direction = -h_inv_ck / std::sqrt(denom_sq);

            double step = kAffineScalingTau;
            Eigen::VectorXd x_next = x;
            bool accepted = false;
            while (step >= kAffineScalingMinStep)
            {
                x_next = x + step * direction;
                if (strictlyFeasible(constraints, x_next))
                {
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }

            if (!accepted)
            {
                if (diagnostic != nullptr)
                {
                    diagnostic->lbfgs_iterations = iter - 1;
                    diagnostic->lbfgs_status = kAffineScalingLineSearchFail;
                }
                return false;
            }

            const double cur_t = x_next(kObjectiveIndex);
            const double rel_improve = std::abs(cur_t - prev_t) / std::max(1.0, std::abs(prev_t));
            x = x_next;
            prev_t = cur_t;

            if (diagnostic != nullptr)
            {
                diagnostic->lbfgs_iterations = iter;
            }

            if (rel_improve < kAffineScalingRelTol)
            {
                if (diagnostic != nullptr)
                {
                    diagnostic->lbfgs_status = kAffineScalingSuccess;
                }
                return true;
            }
        }

        if (diagnostic != nullptr)
        {
            diagnostic->lbfgs_status = kAffineScalingMaxIter;
        }
        return true;
    }

    inline void addPolytopeConeConstraint(const Eigen::Vector3d &a,
                                          std::vector<SOCPConstraint> &constraints)
    {
        SOCPConstraint constraint;
        constraint.c = Eigen::VectorXd::Zero(kSocpStateSize);
        constraint.c.segment<3>(kCenterOffset) = -a;
        constraint.d = 1.0;
        constraint.A = Eigen::MatrixXd::Zero(kSocpStateSize, 3);
        constraint.A(0, 0) = a(0);
        constraint.A(1, 0) = a(1);
        constraint.A(3, 0) = a(2);
        constraint.A(2, 1) = a(1);
        constraint.A(4, 1) = a(2);
        constraint.A(5, 2) = a(2);
        constraint.gram = constraint.A * constraint.A.transpose();
        constraints.emplace_back(std::move(constraint));
    }

    inline void addGeometricMeanConstraints(std::vector<SOCPConstraint> &constraints)
    {
        SOCPConstraint first;
        first.c = Eigen::VectorXd::Zero(kSocpStateSize);
        first.c(0) = 1.0;
        first.c(2) = 1.0;
        first.A = Eigen::MatrixXd::Zero(kSocpStateSize, 2);
        first.A(kGeomAux1Index, 0) = 2.0;
        first.A(0, 1) = 1.0;
        first.A(2, 1) = -1.0;
        first.gram = first.A * first.A.transpose();
        constraints.emplace_back(std::move(first));

        SOCPConstraint second;
        second.c = Eigen::VectorXd::Zero(kSocpStateSize);
        second.c(kGeomAux1Index) = 1.0;
        second.c(kGeomAux2Index) = 1.0;
        second.A = Eigen::MatrixXd::Zero(kSocpStateSize, 2);
        second.A(kObjectiveIndex, 0) = 2.0;
        second.A(kGeomAux1Index, 1) = 1.0;
        second.A(kGeomAux2Index, 1) = -1.0;
        second.gram = second.A * second.A.transpose();
        constraints.emplace_back(std::move(second));

        SOCPConstraint third;
        third.c = Eigen::VectorXd::Zero(kSocpStateSize);
        third.c(5) = 1.0;
        third.c(kObjectiveIndex) = 1.0;
        third.A = Eigen::MatrixXd::Zero(kSocpStateSize, 2);
        third.A(kGeomAux2Index, 0) = 2.0;
        third.A(5, 1) = 1.0;
        third.A(kObjectiveIndex, 1) = -1.0;
        third.gram = third.A * third.A.transpose();
        constraints.emplace_back(std::move(third));
    }

    inline bool returnLastValidPolytopeIfAvailable(const Eigen::MatrixX4d &last_valid_hpoly,
                                                   const bool has_last_valid_hpoly,
                                                   Eigen::MatrixX4d &hPoly)
    {
        if (has_last_valid_hpoly)
        {
            hPoly = last_valid_hpoly;
            return true;
        }
        return false;
    }

    inline bool maxVolInsEllipsoid(const Eigen::MatrixX4d &hPoly,
                                   Eigen::Matrix3d &R,
                                   Eigen::Vector3d &p,
                                   Eigen::Vector3d &r,
                                   firi_common::MVIEDiagnostic *diagnostic = nullptr)
    {
        const int M = hPoly.rows();
        Eigen::MatrixX4d Alp(M, 4);
        Eigen::VectorXd blp(M);
        Eigen::Vector4d clp, xlp;
        const Eigen::ArrayXd hNorm = hPoly.leftCols<3>().rowwise().norm();
        Alp.leftCols<3>() = hPoly.leftCols<3>().array().colwise() / hNorm;
        Alp.rightCols<1>().setConstant(1.0);
        blp = -hPoly.rightCols<1>().array() / hNorm;
        clp.setZero();
        clp(3) = -1.0;
        const double maxdepth = -sdlp::linprog<4>(clp, Alp, blp, xlp);
        if (!(maxdepth > 0.0) || std::isinf(maxdepth))
        {
            if (diagnostic != nullptr)
            {
                *diagnostic = firi_common::MVIEDiagnostic();
                diagnostic->lbfgs_status = kAffineScalingInfeasible;
            }
            return false;
        }
        const Eigen::Vector3d interior = xlp.head<3>();

        const Eigen::MatrixX3d A = Alp.leftCols<3>().array().colwise() /
                                   (blp - Alp.leftCols<3>() * interior).array();

        std::vector<SOCPConstraint> constraints;
        constraints.reserve(static_cast<std::size_t>(M) + 3);
        for (int i = 0; i < M; ++i)
        {
            addPolytopeConeConstraint(A.row(i).transpose(), constraints);
        }
        addGeometricMeanConstraints(constraints);

        const Eigen::Matrix3d Q = R * (r.cwiseProduct(r)).asDiagonal() * R.transpose();
        const Eigen::Matrix3d L0 = Q.llt().matrixL();
        Eigen::VectorXd x = Eigen::VectorXd::Zero(kSocpStateSize);
        x(0) = L0(0, 0);
        x(1) = L0(1, 0);
        x(2) = L0(1, 1);
        x(3) = L0(2, 0);
        x(4) = L0(2, 1);
        x(5) = L0(2, 2);
        x.segment<3>(kCenterOffset) = p - interior;
        initializeGeometricMeanAuxiliaries(x);

        if (!makeWarmStartStrictlyFeasible(constraints, x))
        {
            setFailureDiagnostic(kPaperCheckMVIEWarmStartFail, diagnostic);
            return false;
        }

        if (!affineScalingSOCP(constraints, x, diagnostic))
        {
            return false;
        }

        const Eigen::Matrix3d L = vectorToLowerTriangular(x);
        const Eigen::Vector3d p_shift = x.segment<3>(kCenterOffset);
        p = p_shift + interior;
        firi_common::finalizeMVIEDiagnostic(A, L, p_shift, diagnostic);

        Eigen::JacobiSVD<Eigen::Matrix3d, Eigen::FullPivHouseholderQRPreconditioner> svd(L, Eigen::ComputeFullU);
        const Eigen::Matrix3d U = svd.matrixU();
        const Eigen::Vector3d S = svd.singularValues();
        if (U.determinant() < 0.0)
        {
            R.col(0) = U.col(1);
            R.col(1) = U.col(0);
            R.col(2) = U.col(2);
            r(0) = S(1);
            r(1) = S(0);
            r(2) = S(2);
        }
        else
        {
            R = U;
            r = S;
        }

        return true;
    }

    inline bool computeRestrictiveHalfspace(const std::vector<Eigen::Vector3d> &seed_vertices,
                                            const Eigen::Vector3d &obstacle,
                                            Eigen::Vector3d &a)
    {
        Eigen::Matrix<double, -1, 3> A(seed_vertices.size() + 1, 3);
        Eigen::VectorXd b(seed_vertices.size() + 1);
        for (std::size_t i = 0; i < seed_vertices.size(); ++i)
        {
            A.row(static_cast<Eigen::Index>(i)) = seed_vertices[i].transpose();
            b(static_cast<Eigen::Index>(i)) = 1.0;
        }
        A.row(static_cast<Eigen::Index>(seed_vertices.size())) = -obstacle.transpose();
        b(static_cast<Eigen::Index>(seed_vertices.size())) = -1.0;

        Eigen::Vector3d dual;
        const double min_norm = sdqp::sdmn<3>(A, b, dual);
        const double sq_norm = dual.squaredNorm();
        if (!std::isfinite(min_norm) || !(sq_norm > kAffineScalingFeasTol))
        {
            return false;
        }

        a = dual / sq_norm;
        return true;
    }

    inline bool firi_socp(const Eigen::MatrixX4d &bd,
                          const Eigen::Matrix3Xd &pc,
                          const Eigen::Vector3d &a,
                          const Eigen::Vector3d &b,
                          Eigen::MatrixX4d &hPoly,
                          const int iterations = 4,
                          const double epsilon = 1.0e-6,
                          std::vector<firi_common::MVIEDiagnostic> *mvie_records = nullptr)
    {
        const Eigen::Vector4d ah(a(0), a(1), a(2), 1.0);
        const Eigen::Vector4d bh(b(0), b(1), b(2), 1.0);

        if ((bd * ah).maxCoeff() > 0.0 || (bd * bh).maxCoeff() > 0.0)
        {
            return false;
        }

        const int M = bd.rows();
        const int N = pc.cols();

        Eigen::Matrix3d R;
        Eigen::Vector3d p;
        Eigen::Vector3d r;
        if (!initializeObstacleFreeEllipsoid(bd, pc, a, b, R, p, r))
        {
            if (mvie_records != nullptr)
            {
                mvie_records->emplace_back();
                mvie_records->back().optimizer_ran = false;
                mvie_records->back().lbfgs_status = kPaperCheckNoClearance;
            }
            return false;
        }
        Eigen::MatrixX4d forwardH(M + N, 4);
        int nH = 0;
        Eigen::MatrixX4d last_valid_hpoly;
        bool has_last_valid_hpoly = false;

        for (int loop = 0; loop < iterations; ++loop)
        {
            const Eigen::Matrix3d forward = r.cwiseInverse().asDiagonal() * R.transpose();
            const Eigen::Matrix3d backward = R * r.asDiagonal();
            const Eigen::MatrixX3d forwardB = bd.leftCols<3>() * backward;
            const Eigen::VectorXd forwardD = bd.rightCols<1>() + bd.leftCols<3>() * p;

            const Eigen::Vector3d fwd_a = forward * (a - p);
            const Eigen::Vector3d fwd_b = forward * (b - p);
            std::vector<Eigen::Vector3d> seed_vertices;
            seed_vertices.reserve(2);
            seed_vertices.emplace_back(fwd_a);
            if ((fwd_b - fwd_a).norm() > epsilon)
            {
                seed_vertices.emplace_back(fwd_b);
            }

            std::vector<Eigen::Vector3d> forwardPC(static_cast<std::size_t>(N));
            std::vector<Eigen::Vector3d> normals(static_cast<std::size_t>(N));
            Eigen::VectorXd distDs = forwardD.cwiseAbs().cwiseQuotient(forwardB.rowwise().norm());
            Eigen::VectorXd distRs = Eigen::VectorXd::Constant(N, INFINITY);

            for (int i = 0; i < N; ++i)
            {
                forwardPC[static_cast<std::size_t>(i)] = forward * (pc.col(i) - p);
            }

            if (!transformedProblemSatisfiesPaperConditions(distDs, forwardPC, epsilon))
            {
                if (mvie_records != nullptr)
                {
                    mvie_records->emplace_back();
                    mvie_records->back().optimizer_ran = false;
                    mvie_records->back().lbfgs_status =
                        distDs.size() > 0 && distDs.minCoeff() < 1.0 - std::max(10.0 * epsilon, 1.0e-6)
                            ? kPaperCheckBoundaryIntersectsUnitBall
                            : kPaperCheckObstacleIntersectsUnitBall;
                }
                return returnLastValidPolytopeIfAvailable(last_valid_hpoly, has_last_valid_hpoly, hPoly);
            }

            for (int i = 0; i < N; ++i)
            {
                if (!computeRestrictiveHalfspace(seed_vertices,
                                                 forwardPC[static_cast<std::size_t>(i)],
                                                 normals[static_cast<std::size_t>(i)]))
                {
                    if (mvie_records != nullptr)
                    {
                        mvie_records->emplace_back();
                        mvie_records->back().optimizer_ran = false;
                        mvie_records->back().lbfgs_status = kPaperCheckRestrictiveHalfspaceFail;
                    }
                    return returnLastValidPolytopeIfAvailable(last_valid_hpoly, has_last_valid_hpoly, hPoly);
                }
                if (!validateRestrictiveHalfspace(seed_vertices,
                                                 forwardPC[static_cast<std::size_t>(i)],
                                                 normals[static_cast<std::size_t>(i)],
                                                 epsilon))
                {
                    if (mvie_records != nullptr)
                    {
                        mvie_records->emplace_back();
                        mvie_records->back().optimizer_ran = false;
                        mvie_records->back().lbfgs_status = kPaperCheckRestrictiveHalfspaceInvalid;
                    }
                    return returnLastValidPolytopeIfAvailable(last_valid_hpoly, has_last_valid_hpoly, hPoly);
                }
                distRs(i) = normals[static_cast<std::size_t>(i)].norm();
            }

            Eigen::Matrix<uint8_t, -1, 1> bdFlags = Eigen::Matrix<uint8_t, -1, 1>::Constant(M, 1);
            Eigen::Matrix<uint8_t, -1, 1> pcFlags = Eigen::Matrix<uint8_t, -1, 1>::Constant(N, 1);

            nH = 0;
            bool completed = false;
            int bdMinId = 0, pcMinId = 0;
            double minDistD = distDs.minCoeff(&bdMinId);
            double minDistR = INFINITY;
            if (distRs.size() != 0)
            {
                minDistR = distRs.minCoeff(&pcMinId);
            }

            for (int i = 0; !completed && i < (M + N); ++i)
            {
                if (minDistD < minDistR)
                {
                    forwardH.block<1, 3>(nH, 0) = forwardB.row(bdMinId);
                    forwardH(nH, 3) = forwardD(bdMinId);
                    bdFlags(bdMinId) = 0;
                }
                else
                {
                    const Eigen::Vector3d &normal = normals[static_cast<std::size_t>(pcMinId)];
                    forwardH.block<1, 3>(nH, 0) = normal.transpose();
                    forwardH(nH, 3) = -normal.squaredNorm();
                    pcFlags(pcMinId) = 0;
                }

                completed = true;
                minDistD = INFINITY;
                for (int j = 0; j < M; ++j)
                {
                    if (bdFlags(j))
                    {
                        completed = false;
                        if (distDs(j) < minDistD)
                        {
                            minDistD = distDs(j);
                            bdMinId = j;
                        }
                    }
                }

                minDistR = INFINITY;
                for (int j = 0; j < N; ++j)
                {
                    if (pcFlags(j))
                    {
                        const Eigen::Vector3d &point = forwardPC[static_cast<std::size_t>(j)];
                        if (forwardH.block<1, 3>(nH, 0).dot(point) + forwardH(nH, 3) > -epsilon)
                        {
                            pcFlags(j) = 0;
                        }
                        else
                        {
                            completed = false;
                            if (distRs(j) < minDistR)
                            {
                                minDistR = distRs(j);
                                pcMinId = j;
                            }
                        }
                    }
                }
                ++nH;
            }

            hPoly.resize(nH, 4);
            for (int i = 0; i < nH; ++i)
            {
                hPoly.block<1, 3>(i, 0) = forwardH.block<1, 3>(i, 0) * forward;
                hPoly(i, 3) = forwardH(i, 3) - hPoly.block<1, 3>(i, 0).dot(p);
            }
            last_valid_hpoly = hPoly;
            has_last_valid_hpoly = true;

            if (loop == iterations - 1)
            {
                break;
            }

            firi_common::MVIEDiagnostic *mvie_record = nullptr;
            if (mvie_records != nullptr)
            {
                mvie_records->emplace_back();
                mvie_record = &mvie_records->back();
            }
            if (!maxVolInsEllipsoid(hPoly, R, p, r, mvie_record))
            {
                return returnLastValidPolytopeIfAvailable(last_valid_hpoly, has_last_valid_hpoly, hPoly);
            }
        }

        return true;
    }
}

#endif
