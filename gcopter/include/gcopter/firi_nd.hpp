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

#ifndef FIRI_ND_HPP
#define FIRI_ND_HPP

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
#include <algorithm>

namespace firi_nd
{
    template <int Dim>
    struct MVIECostData
    {
        Eigen::Matrix<double, Eigen::Dynamic, Dim> A;
        firi_common::MVIEDiagnostic *diagnostic = nullptr;
    };

    template <int Dim>
    inline int progressMVIE(void *data,
                            const Eigen::VectorXd &,
                            const Eigen::VectorXd &,
                            const double,
                            const double,
                            const int k,
                            const int)
    {
        MVIECostData<Dim> *optData = static_cast<MVIECostData<Dim> *>(data);
        if (optData != nullptr && optData->diagnostic != nullptr)
        {
            optData->diagnostic->lbfgs_iterations = k;
        }
        return 0;
    }

    template <int Dim>
    struct NormalComputer {
        static Eigen::Matrix<double, Dim, 1> compute(const Eigen::Matrix<double, Dim, 1>& da, 
                                                     const Eigen::Matrix<double, Dim, 1>& db) {
            // 高维空间: SVD 求解 Null Space
            Eigen::Matrix<double, 2, Dim> mat;
            mat.row(0) = da.transpose();
            mat.row(1) = db.transpose();
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeFullV);
            return svd.matrixV().col(Dim - 1).normalized();
        }
    };

    // 2D 偏特化: 逆时针旋转 90 度
    template <>
    struct NormalComputer<2> {
        static Eigen::Matrix<double, 2, 1> compute(const Eigen::Matrix<double, 2, 1>& da, 
                                                   const Eigen::Matrix<double, 2, 1>& db) {
            return Eigen::Matrix<double, 2, 1>(-da(1), da(0)).normalized();
        }
    };

    template <>
    struct NormalComputer<3> {
        static Eigen::Matrix<double, 3, 1> compute(const Eigen::Matrix<double, 3, 1>& da, 
                                                   const Eigen::Matrix<double, 3, 1>& db) {
            return da.cross(db).normalized();
        }
    };

    template <int Dim>
    inline Eigen::Matrix<double, Dim, 1> computeNormal(const Eigen::Matrix<double, Dim, 1>& da, 
                                                       const Eigen::Matrix<double, Dim, 1>& db) {
        return NormalComputer<Dim>::compute(da, db);
    }


    // =========================================================================
    // Hom-Opt MVIE: Topological Mapping to Unit Sphere with Analytical Gradient
    // =========================================================================
    template <int Dim>
    inline double costMVIE_HomOpt(void *data,
                                  const Eigen::VectorXd &y,
                                  Eigen::VectorXd &grad)
    {
        constexpr int K = Dim + Dim * (Dim + 1) / 2;
        const MVIECostData<Dim> *optData = static_cast<const MVIECostData<Dim> *>(data);
        const int M = optData->A.rows();
        const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Dim>> A = optData->A;

        double cost = 0.0;
        grad.setZero();

        bool out_of_bounds = false;
        int diag_idx = Dim;
        for (int col = 0; col < Dim; ++col) {
            if (y(diag_idx) <= 1e-7) { out_of_bounds = true; break; }
            diag_idx += (Dim - col);
        }

        if (out_of_bounds)
        {
            cost = 1e6;
            diag_idx = Dim;
            for (int col = 0; col < Dim; ++col) {
                if (y(diag_idx) <= 1e-7) { 
                    cost += 1e5 * (1e-7 - y(diag_idx)); 
                    grad(diag_idx) = -1e5; 
                }
                diag_idx += (Dim - col);
            }
            return cost;
        }

        Eigen::Matrix<double, Eigen::Dynamic, Dim> v = Eigen::Matrix<double, Eigen::Dynamic, Dim>::Zero(M, Dim);
        int idx = Dim;
        for (int col = 0; col < Dim; ++col) {
            for (int row = col; row < Dim; ++row) {
                double L_val = y(idx++);
                v.col(col) += A.col(row) * L_val;
            }
        }

        Eigen::VectorXd mu(M), n(M);
        for (int i = 0; i < M; ++i)
        {
            n(i) = std::sqrt(v.row(i).squaredNorm() + 1e-12);
            mu(i) = n(i) + A.row(i).dot(y.head(Dim));
        }

        const double alpha = 50.0;
        double mu_max = mu.maxCoeff();
        Eigen::VectorXd exp_term = (alpha * (mu.array() - mu_max)).exp();
        double sum_exp = exp_term.sum();
        double mu_val = mu_max + std::log(sum_exp) / alpha;
        Eigen::VectorXd w = exp_term / sum_exp; 

        double log_det = 0.0;
        diag_idx = Dim;
        for (int col = 0; col < Dim; ++col) {
            log_det += std::log(y(diag_idx));
            diag_idx += (Dim - col);
        }
        cost = -log_det + double(Dim) * std::log(mu_val);

        Eigen::VectorXd grad_mu = Eigen::VectorXd::Zero(K);
        for (int i = 0; i < M; ++i)
        {
            double inv_n = 1.0 / n(i);
            for (int d = 0; d < Dim; ++d) {
                grad_mu(d) += w(i) * A(i, d);
            }
            idx = Dim;
            for (int col = 0; col < Dim; ++col) {
                for (int row = col; row < Dim; ++row) {
                    double dy_idx = v(i, col) * inv_n * A(i, row);
                    grad_mu(idx++) += w(i) * dy_idx;
                }
            }
        }

        grad = (double(Dim) / mu_val) * grad_mu;
        diag_idx = Dim;
        for (int col = 0; col < Dim; ++col) {
            grad(diag_idx) -= 1.0 / y(diag_idx);
            diag_idx += (Dim - col);
        }

        const double lambda = 500.0;
        double norm_y_sqr = y.squaredNorm();
        double diff = norm_y_sqr - 1.0;
        cost += 0.5 * lambda * diff * diff;
        grad += 2.0 * lambda * diff * y;

        return cost;
    }

    template <int Dim>
    inline bool maxVolInsEllipsoid(const Eigen::Matrix<double, Eigen::Dynamic, Dim + 1> &hPoly,
                                   Eigen::Matrix<double, Dim, Dim> &R,
                                   Eigen::Matrix<double, Dim, 1> &p,
                                   Eigen::Matrix<double, Dim, 1> &r,
                                   firi_common::MVIEDiagnostic *diagnostic = nullptr)
    {
        const int M = hPoly.rows();
        Eigen::Matrix<double, Eigen::Dynamic, Dim + 1> Alp(M, Dim + 1);
        Eigen::VectorXd blp(M);
        Eigen::Matrix<double, Dim + 1, 1> clp, xlp;
        
        const Eigen::ArrayXd hNorm = hPoly.template leftCols<Dim>().rowwise().norm();
        Alp.template leftCols<Dim>() = hPoly.template leftCols<Dim>().array().colwise() / hNorm;
        Alp.template rightCols<1>().setConstant(1.0);
        blp = -hPoly.template rightCols<1>().array() / hNorm;
        clp.setZero();
        clp(Dim) = -1.0;
        
        const double maxdepth = -sdlp::linprog<Dim + 1>(clp, Alp, blp, xlp);
        if (!(maxdepth > 0.0) || std::isinf(maxdepth)) return false;
        
        const Eigen::Matrix<double, Dim, 1> interior = xlp.head(Dim);

        MVIECostData<Dim> optData;
        optData.A = Alp.template leftCols<Dim>().array().colwise() / (blp - Alp.template leftCols<Dim>() * interior).array();
        optData.diagnostic = diagnostic;
        if (diagnostic != nullptr)
        {
            *diagnostic = firi_common::MVIEDiagnostic();
        }

        constexpr int K = Dim + Dim * (Dim + 1) / 2;
        Eigen::VectorXd x(K);
        
        const Eigen::Matrix<double, Dim, Dim> Q = R * (r.cwiseProduct(r)).asDiagonal() * R.transpose();
        Eigen::Matrix<double, Dim, Dim> L = Q.llt().matrixL();

        x.head(Dim) = p - interior;
        
        int idx = Dim;
        for (int col = 0; col < Dim; ++col) {
            for (int row = col; row < Dim; ++row) {
                x(idx++) = L(row, col);
            }
        }

        Eigen::VectorXd y = x;
        if (y.norm() > 1e-6) {
            y.normalize();
        } else {
            y.setZero();
            int diag_idx = Dim;
            for (int col = 0; col < Dim; ++col) {
                y(diag_idx) = 1.0;
                diag_idx += (Dim - col);
            }
            y.normalize();
        }

        double minCost;
        const lbfgs::lbfgs_parameter_t paramsMVIE = firi_common::defaultMVIELbfgsParameters();

        int ret = lbfgs::lbfgs_optimize(y,
                                        minCost,
                                        &costMVIE_HomOpt<Dim>,
                                        nullptr,
                                        diagnostic != nullptr ? &progressMVIE<Dim> : nullptr,
                                        &optData,
                                        paramsMVIE);
        if (diagnostic != nullptr)
        {
            diagnostic->lbfgs_status = ret;
        }

        Eigen::Matrix<double, Dim, Dim> L_opt = Eigen::Matrix<double, Dim, Dim>::Zero();
        idx = Dim;
        for (int col = 0; col < Dim; ++col) {
            for (int row = col; row < Dim; ++row) {
                L_opt(row, col) = y(idx++);
            }
        }

        Eigen::VectorXd mu_exact(M);
        Eigen::Matrix<double, Eigen::Dynamic, Dim> v_opt = optData.A * L_opt;
        for (int i = 0; i < M; ++i) {
            mu_exact(i) = v_opt.row(i).norm() + optData.A.row(i).dot(y.head(Dim));
        }
        
        double final_mu = mu_exact.maxCoeff(); 
        x = y / final_mu;

        const Eigen::Matrix<double, Dim, 1> p_shift = x.head(Dim);
        p = p_shift + interior;
        
        idx = Dim;
        for (int col = 0; col < Dim; ++col) 
        {
            for(int row = col; row < Dim; ++row) 
            {
                L(row, col) = x(idx++);
            }
        }
        firi_common::finalizeMVIEDiagnostic(optData.A, L, p_shift, diagnostic);
        
        Eigen::JacobiSVD<Eigen::Matrix<double, Dim, Dim>> svd(L, Eigen::ComputeFullU);
        const Eigen::Matrix<double, Dim, Dim> U = svd.matrixU();
        const Eigen::Matrix<double, Dim, 1> S = svd.singularValues();
        
        if(U.determinant() < 0.0) 
        {
            R = U;
            R.col(0) = U.col(1);
            R.col(1) = U.col(0);
            r = S;
            std::swap(r(0), r(1));
        } 
        else 
        {
            R = U;
            r = S;
        }
        return ret >= 0;
    }

    template <int Dim>
    inline bool firi_opt(const Eigen::Matrix<double, Eigen::Dynamic, Dim + 1> &bd,
                         const Eigen::Matrix<double, Dim, Eigen::Dynamic> &pc,
                         const Eigen::Matrix<double, Dim, 1> &a,
                         const Eigen::Matrix<double, Dim, 1> &b,
                         Eigen::Matrix<double, Eigen::Dynamic, Dim + 1> &hPoly,
                         const int iterations = 4,
                         const double epsilon = 1.0e-6,
                         std::vector<firi_common::MVIEDiagnostic> *mvie_records = nullptr)
    {
        Eigen::Matrix<double, Dim + 1, 1> ah, bh;
        ah.head(Dim) = a; ah(Dim) = 1.0;
        bh.head(Dim) = b; bh(Dim) = 1.0;

        if ((bd * ah).maxCoeff() > 0.0 || (bd * bh).maxCoeff() > 0.0) return false;

        const int M = bd.rows();
        const int N = pc.cols();

        Eigen::Matrix<double, Dim, Dim> R = Eigen::Matrix<double, Dim, Dim>::Identity();
        Eigen::Matrix<double, Dim, 1> p = 0.5 * (a + b);
        Eigen::Matrix<double, Dim, 1> r = Eigen::Matrix<double, Dim, 1>::Ones();
        Eigen::Matrix<double, Eigen::Dynamic, Dim + 1> forwardH(M + N, Dim + 1);
        int nH = 0;

        for (int loop = 0; loop < iterations; ++loop)
        {
            const Eigen::Matrix<double, Dim, Dim> forward = r.cwiseInverse().asDiagonal() * R.transpose();
            const Eigen::Matrix<double, Dim, Dim> backward = R * r.asDiagonal();
            const Eigen::Matrix<double, Eigen::Dynamic, Dim> forwardB = bd.template leftCols<Dim>() * backward;
            const Eigen::VectorXd forwardD = bd.template rightCols<1>() + bd.template leftCols<Dim>() * p;
            
            const Eigen::Matrix<double, Dim, Eigen::Dynamic> forwardPC = forward * (pc.colwise() - p);
            const Eigen::Matrix<double, Dim, 1> fwd_a = forward * (a - p);
            const Eigen::Matrix<double, Dim, 1> fwd_b = forward * (b - p);

            const Eigen::VectorXd distDs = forwardD.cwiseAbs().cwiseQuotient(forwardB.rowwise().norm());
            Eigen::Matrix<double, Eigen::Dynamic, Dim + 1> tangents(N, Dim + 1);
            Eigen::VectorXd distRs(N);

            for (int i = 0; i < N; i++)
            {
                distRs(i) = forwardPC.col(i).norm();
                tangents(i, Dim) = -distRs(i);
                
                // 替换为 .template block<1, Dim> 以确保它是向量
                tangents.template block<1, Dim>(i, 0) = forwardPC.col(i).transpose() / distRs(i);
                
                if (tangents.template block<1, Dim>(i, 0).dot(fwd_a) + tangents(i, Dim) > epsilon) 
                {
                    const Eigen::Matrix<double, Dim, 1> delta = forwardPC.col(i) - fwd_a;
                    tangents.template block<1, Dim>(i, 0) = (fwd_a - (delta.dot(fwd_a) / delta.squaredNorm()) * delta).transpose();
                    distRs(i) = tangents.template block<1, Dim>(i, 0).norm();
                    tangents(i, Dim) = -distRs(i);
                    tangents.template block<1, Dim>(i, 0) /= distRs(i);
                }
                
                if (tangents.template block<1, Dim>(i, 0).dot(fwd_b) + tangents(i, Dim) > epsilon) 
                {
                    const Eigen::Matrix<double, Dim, 1> delta = forwardPC.col(i) - fwd_b;
                    tangents.template block<1, Dim>(i, 0) = (fwd_b - (delta.dot(fwd_b) / delta.squaredNorm()) * delta).transpose();
                    distRs(i) = tangents.template block<1, Dim>(i, 0).norm();
                    tangents(i, Dim) = -distRs(i);
                    tangents.template block<1, Dim>(i, 0) /= distRs(i);
                }
                
                if (tangents.template block<1, Dim>(i, 0).dot(fwd_a) + tangents(i, Dim) > epsilon) {
                    Eigen::Matrix<double, Dim, 1> normal = computeNormal<Dim>(fwd_a - forwardPC.col(i), fwd_b - forwardPC.col(i));
                    tangents.template block<1, Dim>(i, 0) = normal.transpose();
                    tangents(i, Dim) = -tangents.template block<1, Dim>(i, 0).dot(fwd_a);
                    if (tangents(i, Dim) > 0.0) tangents.row(i) *= -1.0;
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
                if (minSqrD < minSqrR) {
                    forwardH.template block<1, Dim>(nH, 0) = forwardB.row(bdMinId);
                    forwardH(nH, Dim) = forwardD(bdMinId);
                    bdFlags(bdMinId) = 0;
                } else {
                    forwardH.row(nH) = tangents.row(pcMinId);
                    pcFlags(pcMinId) = 0;
                }

                completed = true;
                minSqrD = INFINITY;
                for (int j = 0; j < M; ++j) {
                    if (bdFlags(j)) {
                        completed = false;
                        if (minSqrD > distDs(j)) { bdMinId = j; minSqrD = distDs(j); }
                    }
                }
                minSqrR = INFINITY;
                for (int j = 0; j < N; ++j) {
                    if (pcFlags(j)) {
                        if (forwardH.template block<1, Dim>(nH, 0).dot(forwardPC.col(j)) + forwardH(nH, Dim) > -epsilon) {
                            pcFlags(j) = 0;
                        } else {
                            completed = false;
                            if (minSqrR > distRs(j)) { pcMinId = j; minSqrR = distRs(j); }
                        }
                    }
                }
                ++nH;
            }

            hPoly.resize(nH, Dim + 1);
            for (int i = 0; i < nH; ++i) {
                hPoly.template block<1, Dim>(i, 0) = forwardH.template block<1, Dim>(i, 0) * forward;
                hPoly(i, Dim) = forwardH(i, Dim) - hPoly.template block<1, Dim>(i, 0).dot(p);
            }

            if (loop == iterations - 1) break;

            firi_common::MVIEDiagnostic *mvie_record = nullptr;
            if (mvie_records != nullptr)
            {
                mvie_records->emplace_back();
                mvie_record = &mvie_records->back();
            }
            maxVolInsEllipsoid<Dim>(hPoly, R, p, r, mvie_record);
        }
        return true;
    }
}

#endif // FIRI_OPT_HPP
