/*
    MIT License

    Copyright (c) 2021 Zhepei Wang (wangzhepei@live.com)
    Modifications for Hom-Opt MVIE by KaiChen Guo(kaireriemann2025@163.com)

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

#ifndef FIRI_OPT_HPP
#define FIRI_OPT_HPP

#include "lbfgs.hpp"
#include "firi_lbfgs_defaults.hpp"
#include "firi_mvie_diagnostics.hpp"
#include "sdlp.hpp"

#include <Eigen/Eigen>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <vector>

namespace firi_opt
{
    struct MVIECostData
    {
        Eigen::MatrixX3d A;
        firi_common::MVIEDiagnostic *diagnostic = nullptr;
    };

    inline int progressMVIE(void *data,
                            const Eigen::VectorXd &,
                            const Eigen::VectorXd &,
                            const double,
                            const double,
                            const int k,
                            const int)
    {
        MVIECostData *optData = static_cast<MVIECostData *>(data);
        if (optData != nullptr && optData->diagnostic != nullptr)
        {
            optData->diagnostic->lbfgs_iterations = k;
        }
        return 0;
    }


    inline void chol3d(const Eigen::Matrix3d &A,
                       Eigen::Matrix3d &L)
    {
        L(0, 0) = std::sqrt(A(0, 0));
        L(0, 1) = 0.0;
        L(0, 2) = 0.0;
        L(1, 0) = 0.5 * (A(0, 1) + A(1, 0)) / L(0, 0);
        L(1, 1) = std::sqrt(A(1, 1) - L(1, 0) * L(1, 0));
        L(1, 2) = 0.0;
        L(2, 0) = 0.5 * (A(0, 2) + A(2, 0)) / L(0, 0);
        L(2, 1) = (0.5 * (A(1, 2) + A(2, 1)) - L(2, 0) * L(1, 0)) / L(1, 1);
        L(2, 2) = std::sqrt(A(2, 2) - L(2, 0) * L(2, 0) - L(2, 1) * L(2, 1));
        return;
    }

    // =========================================================================
    // Hom-Opt MVIE: Topological Mapping to Unit Sphere with Analytical Gradient
    // =========================================================================
    inline double costMVIE_HomOpt(void *data,
                                  const Eigen::VectorXd &y,
                                  Eigen::VectorXd &grad)
    {
        const MVIECostData *optData = static_cast<const MVIECostData *>(data);
        const int M = optData->A.rows();
        const Eigen::Ref<const Eigen::MatrixX3d> A = optData->A;

        double cost = 0.0;
        grad.setZero();

        if (y(3) <= 1e-7 || y(4) <= 1e-7 || y(5) <= 1e-7)
        {
            cost = 1e6;
            if (y(3) <= 1e-7) { cost += 1e5 * (1e-7 - y(3)); grad(3) = -1e5; }
            if (y(4) <= 1e-7) { cost += 1e5 * (1e-7 - y(4)); grad(4) = -1e5; }
            if (y(5) <= 1e-7) { cost += 1e5 * (1e-7 - y(5)); grad(5) = -1e5; }
            return cost;
        }

        Eigen::VectorXd mu(M);
        Eigen::VectorXd n(M);
        Eigen::MatrixX3d v(M, 3);

        // 2. 计算 Minkowski Gauge 泛函 \mu_i(y) 及其向量分量
        for (int i = 0; i < M; ++i)
        {
            // v = A_i * L(y)
            v(i, 0) = A(i, 0) * y(3) + A(i, 1) * y(6) + A(i, 2) * y(8);
            v(i, 1) = A(i, 1) * y(4) + A(i, 2) * y(7);
            v(i, 2) = A(i, 2) * y(5);

            // norm(v)
            n(i) = std::sqrt(v(i, 0) * v(i, 0) + v(i, 1) * v(i, 1) + v(i, 2) * v(i, 2) + 1e-12);
            
            // \mu_i(y) = ||A_i * L(y)||_2 + A_i * p(y)
            mu(i) = n(i) + A(i, 0) * y(0) + A(i, 1) * y(1) + A(i, 2) * y(2);
        }

        const double alpha = 50.0;
        double mu_max = mu.maxCoeff();
        Eigen::VectorXd exp_term = (alpha * (mu.array() - mu_max)).exp();
        double sum_exp = exp_term.sum();
        double mu_val = mu_max + std::log(sum_exp) / alpha;
        Eigen::VectorXd w = exp_term / sum_exp; 


        cost = -std::log(y(3)) - std::log(y(4)) - std::log(y(5)) + 3.0 * std::log(mu_val);

        Eigen::VectorXd grad_mu = Eigen::VectorXd::Zero(9);
        for (int i = 0; i < M; ++i)
        {
            double inv_n = 1.0 / n(i);
            
            double dy3 = v(i, 0) * inv_n * A(i, 0);
            double dy4 = v(i, 1) * inv_n * A(i, 1);
            double dy5 = v(i, 2) * inv_n * A(i, 2);
            double dy6 = v(i, 0) * inv_n * A(i, 1);
            double dy7 = v(i, 1) * inv_n * A(i, 2);
            double dy8 = v(i, 0) * inv_n * A(i, 2);

            grad_mu(0) += w(i) * A(i, 0);
            grad_mu(1) += w(i) * A(i, 1);
            grad_mu(2) += w(i) * A(i, 2);
            grad_mu(3) += w(i) * dy3;
            grad_mu(4) += w(i) * dy4;
            grad_mu(5) += w(i) * dy5;
            grad_mu(6) += w(i) * dy6;
            grad_mu(7) += w(i) * dy7;
            grad_mu(8) += w(i) * dy8;
        }

        grad = (3.0 / mu_val) * grad_mu;
        grad(3) -= 1.0 / y(3);
        grad(4) -= 1.0 / y(4);
        grad(5) -= 1.0 / y(5);

        const double lambda = 500.0;
        double norm_y_sqr = y.squaredNorm();
        double diff = norm_y_sqr - 1.0;
        cost += 0.5 * lambda * diff * diff;
        grad += 2.0 * lambda * diff * y;

        return cost;
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
            return false;
        }
        const Eigen::Vector3d interior = xlp.head<3>();

        MVIECostData optData;
        optData.A = Alp.leftCols<3>().array().colwise() / (blp - Alp.leftCols<3>() * interior).array();
        optData.diagnostic = diagnostic;
        if (diagnostic != nullptr)
        {
            *diagnostic = firi_common::MVIEDiagnostic();
        }

        Eigen::VectorXd x(9);
        const Eigen::Matrix3d Q = R * (r.cwiseProduct(r)).asDiagonal() * R.transpose();
        Eigen::Matrix3d L;
        chol3d(Q, L);

        x.head<3>() = p - interior;
        x(3) = L(0, 0); 
        x(4) = L(1, 1);
        x(5) = L(2, 2);
        x(6) = L(1, 0);
        x(7) = L(2, 1);
        x(8) = L(2, 0);

        Eigen::VectorXd y = x;
        if (y.norm() > 1e-6) {
            y.normalize();
        } else {
            y.setZero();
            y(3) = 1.0; y(4) = 1.0; y(5) = 1.0;
            y.normalize();
        }

        double minCost;
        const lbfgs::lbfgs_parameter_t paramsMVIE = firi_common::defaultMVIELbfgsParameters();


        int ret = lbfgs::lbfgs_optimize(y,
                                        minCost,
                                        &costMVIE_HomOpt,
                                        nullptr,
                                        diagnostic != nullptr ? &progressMVIE : nullptr,
                                        &optData,
                                        paramsMVIE);
        if (diagnostic != nullptr)
        {
            diagnostic->lbfgs_status = ret;
        }

        Eigen::VectorXd mu_vec(M);
        for (int i = 0; i < M; ++i) {
            double v0 = optData.A(i, 0) * y(3) + optData.A(i, 1) * y(6) + optData.A(i, 2) * y(8);
            double v1 = optData.A(i, 1) * y(4) + optData.A(i, 2) * y(7);
            double v2 = optData.A(i, 2) * y(5);
            double ni = std::sqrt(v0*v0 + v1*v1 + v2*v2 + 1e-12);
            mu_vec(i) = ni + optData.A(i, 0) * y(0) + optData.A(i, 1) * y(1) + optData.A(i, 2) * y(2);
        }
        const double alpha = 50.0;
        double mu_max = mu_vec.maxCoeff();
        double sum_exp = (alpha * (mu_vec.array() - mu_max)).exp().sum();
        double final_mu = mu_max + std::log(sum_exp) / alpha;

        x = y / final_mu;

        const Eigen::Vector3d p_shift = x.head<3>();
        p = p_shift + interior;
        L(0, 0) = x(3); 
        L(0, 1) = 0.0;
        L(0, 2) = 0.0;
        L(1, 0) = x(6);
        L(1, 1) = x(4);
        L(1, 2) = 0.0;
        L(2, 0) = x(8);
        L(2, 1) = x(7);
        L(2, 2) = x(5);
        firi_common::finalizeMVIEDiagnostic(optData.A, L, p_shift, diagnostic);
        
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
        return ret >= 0;
    }

    inline bool firi_opt(const Eigen::MatrixX4d &bd,
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

        if ((bd * ah).maxCoeff() > 0.0 ||
            (bd * bh).maxCoeff() > 0.0)
        {
            return false;
        }

        const int M = bd.rows();
        const int N = pc.cols();

        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        Eigen::Vector3d p = 0.5 * (a + b);
        Eigen::Vector3d r = Eigen::Vector3d::Ones();
        Eigen::MatrixX4d forwardH(M + N, 4);
        int nH = 0;

        for (int loop = 0; loop < iterations; ++loop)
        {
            const Eigen::Matrix3d forward = r.cwiseInverse().asDiagonal() * R.transpose();
            const Eigen::Matrix3d backward = R * r.asDiagonal();
            const Eigen::MatrixX3d forwardB = bd.leftCols<3>() * backward;
            const Eigen::VectorXd forwardD = bd.rightCols<1>() + bd.leftCols<3>() * p;
            const Eigen::Matrix3Xd forwardPC = forward * (pc.colwise() - p);
            const Eigen::Vector3d fwd_a = forward * (a - p);
            const Eigen::Vector3d fwd_b = forward * (b - p);

            const Eigen::VectorXd distDs = forwardD.cwiseAbs().cwiseQuotient(forwardB.rowwise().norm());
            Eigen::MatrixX4d tangents(N, 4);
            Eigen::VectorXd distRs(N);

            for (int i = 0; i < N; i++)
            {
                distRs(i) = forwardPC.col(i).norm();
                tangents(i, 3) = -distRs(i);
                tangents.block<1, 3>(i, 0) = forwardPC.col(i).transpose() / distRs(i);
                if (tangents.block<1, 3>(i, 0).dot(fwd_a) + tangents(i, 3) > epsilon)
                {
                    const Eigen::Vector3d delta = forwardPC.col(i) - fwd_a;
                    tangents.block<1, 3>(i, 0) = fwd_a - (delta.dot(fwd_a) / delta.squaredNorm()) * delta;
                    distRs(i) = tangents.block<1, 3>(i, 0).norm();
                    tangents(i, 3) = -distRs(i);
                    tangents.block<1, 3>(i, 0) /= distRs(i);
                }
                if (tangents.block<1, 3>(i, 0).dot(fwd_b) + tangents(i, 3) > epsilon)
                {
                    const Eigen::Vector3d delta = forwardPC.col(i) - fwd_b;
                    tangents.block<1, 3>(i, 0) = fwd_b - (delta.dot(fwd_b) / delta.squaredNorm()) * delta;
                    distRs(i) = tangents.block<1, 3>(i, 0).norm();
                    tangents(i, 3) = -distRs(i);
                    tangents.block<1, 3>(i, 0) /= distRs(i);
                }
                if (tangents.block<1, 3>(i, 0).dot(fwd_a) + tangents(i, 3) > epsilon)
                {
                    tangents.block<1, 3>(i, 0) = (fwd_a - forwardPC.col(i)).cross(fwd_b - forwardPC.col(i)).normalized();
                    tangents(i, 3) = -tangents.block<1, 3>(i, 0).dot(fwd_a);
                    tangents.row(i) *= tangents(i, 3) > 0.0 ? -1.0 : 1.0;
                }
            }

            Eigen::Matrix<uint8_t, -1, 1> bdFlags = Eigen::Matrix<uint8_t, -1, 1>::Constant(M, 1);
            Eigen::Matrix<uint8_t, -1, 1> pcFlags = Eigen::Matrix<uint8_t, -1, 1>::Constant(N, 1);

            nH = 0;
            bool completed = false;
            int bdMinId = 0, pcMinId = 0;
            double minSqrD = distDs.minCoeff(&bdMinId);
            double minSqrR = INFINITY;
            if (distRs.size() != 0) { minSqrR = distRs.minCoeff(&pcMinId); }
            
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
                        if (forwardH.block<1, 3>(nH, 0).dot(forwardPC.col(j)) + forwardH(nH, 3) > -epsilon)
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

            hPoly.resize(nH, 4);
            for (int i = 0; i < nH; ++i)
            {
                hPoly.block<1, 3>(i, 0) = forwardH.block<1, 3>(i, 0) * forward;
                hPoly(i, 3) = forwardH(i, 3) - hPoly.block<1, 3>(i, 0).dot(p);
            }

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
            maxVolInsEllipsoid(hPoly, R, p, r, mvie_record);
        }

        return true;
    }
}

#endif
