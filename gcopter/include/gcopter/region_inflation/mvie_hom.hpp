#ifndef GCOPTER_REGION_INFLATION_MVIE_HOM_HPP
#define GCOPTER_REGION_INFLATION_MVIE_HOM_HPP

#include "gcopter/region_inflation/mvie_solver.hpp"
#include "gcopter/sdlp.hpp"

#include <chrono>
#include <cmath>

namespace region_inflation
{
    namespace detail
    {
        struct HomGaugeCostData
        {
            Eigen::MatrixX3d A;
            double alpha = 50.0;
            double normalization_penalty_lambda = 500.0;
            double min_positive_diagonal = 1.0e-7;
            int objective_evaluations = 0;
            MvieStats *stats = nullptr;
        };

        inline double homGaugeCost(void *data,
                                   const Eigen::VectorXd &y,
                                   Eigen::VectorXd &grad)
        {
            HomGaugeCostData *cost_data = static_cast<HomGaugeCostData *>(data);
            ++cost_data->objective_evaluations;

            const int M = cost_data->A.rows();
            const Eigen::Ref<const Eigen::MatrixX3d> A = cost_data->A;
            const double min_diag = cost_data->min_positive_diagonal;
            const double alpha = cost_data->alpha;
            const double lambda = cost_data->normalization_penalty_lambda;

            grad.setZero();
            if (y(3) <= min_diag || y(6) <= min_diag || y(8) <= min_diag)
            {
                double cost = 1.0e6;
                if (y(3) <= min_diag)
                {
                    cost += 1.0e5 * (min_diag - y(3));
                    grad(3) = -1.0e5;
                }
                if (y(6) <= min_diag)
                {
                    cost += 1.0e5 * (min_diag - y(6));
                    grad(6) = -1.0e5;
                }
                if (y(8) <= min_diag)
                {
                    cost += 1.0e5 * (min_diag - y(8));
                    grad(8) = -1.0e5;
                }
                return cost;
            }

            const Eigen::Matrix3d L = vectorToLower3D(y, 3);
            const Eigen::MatrixX3d v = A * L;
            Eigen::VectorXd mu(M);
            Eigen::VectorXd norm_v(M);
            for (int i = 0; i < M; ++i)
            {
                norm_v(i) = std::sqrt(v.row(i).squaredNorm() + 1.0e-12);
                mu(i) = norm_v(i) + A.row(i).dot(y.head<3>());
            }

            const double mu_max = mu.maxCoeff();
            const Eigen::VectorXd exp_term = (alpha * (mu.array() - mu_max)).exp();
            const double sum_exp = exp_term.sum();
            const double mu_smooth = mu_max + std::log(sum_exp) / alpha;
            const Eigen::VectorXd w = exp_term / sum_exp;

            double cost = -std::log(y(3)) - std::log(y(6)) - std::log(y(8)) +
                          3.0 * std::log(mu_smooth);

            Eigen::VectorXd grad_mu = Eigen::VectorXd::Zero(9);
            for (int i = 0; i < M; ++i)
            {
                const double inv_norm = 1.0 / norm_v(i);
                grad_mu.head<3>() += w(i) * A.row(i).transpose();

                int idx = 3;
                for (int col = 0; col < 3; ++col)
                {
                    for (int row = col; row < 3; ++row)
                    {
                        grad_mu(idx++) += w(i) * v(i, col) * inv_norm * A(i, row);
                    }
                }
            }

            grad = (3.0 / mu_smooth) * grad_mu;
            grad(3) -= 1.0 / y(3);
            grad(6) -= 1.0 / y(6);
            grad(8) -= 1.0 / y(8);

            const double diff = y.squaredNorm() - 1.0;
            cost += 0.5 * lambda * diff * diff;
            grad += 2.0 * lambda * diff * y;

            return cost;
        }

        inline int homGaugeProgress(void *data,
                                    const Eigen::VectorXd &,
                                    const Eigen::VectorXd &,
                                    const double,
                                    const double,
                                    const int k,
                                    const int)
        {
            HomGaugeCostData *cost_data = static_cast<HomGaugeCostData *>(data);
            if (cost_data != nullptr && cost_data->stats != nullptr)
            {
                cost_data->stats->iterations = k;
            }
            return 0;
        }
    }

    class HomGaugeMvieSolver3D final : public IMvieSolver3D
    {
    public:
        const char *name() const override
        {
            return "firi_hom";
        }

        bool solve(const Eigen::MatrixX4d &hpoly,
                   Ellipsoid3D &ellipsoid,
                   const MvieOptions &options,
                   MvieStats &stats) const override
        {
            stats = MvieStats();
            stats.facet_count = hpoly.rows();
            const auto start = std::chrono::steady_clock::now();

            const int M = hpoly.rows();
            if (M < 4)
            {
                stats.failure_reason = "too_few_facets";
                stats.solve_ms = elapsedMs(start);
                return false;
            }

            Eigen::MatrixX4d Alp(M, 4);
            Eigen::VectorXd blp(M);
            Eigen::Vector4d clp, xlp;
            const Eigen::ArrayXd hNorm = hpoly.leftCols<3>().rowwise().norm();
            if (!(hNorm > 0.0).all())
            {
                stats.failure_reason = "zero_halfspace_normal";
                stats.solve_ms = elapsedMs(start);
                return false;
            }

            Alp.leftCols<3>() = hpoly.leftCols<3>().array().colwise() / hNorm;
            Alp.rightCols<1>().setConstant(1.0);
            blp = -hpoly.rightCols<1>().array() / hNorm;
            clp.setZero();
            clp(3) = -1.0;
            const double maxdepth = -sdlp::linprog<4>(clp, Alp, blp, xlp);
            if (!(maxdepth > 0.0) || std::isinf(maxdepth))
            {
                stats.failure_reason = "no_strict_interior";
                stats.solve_ms = elapsedMs(start);
                return false;
            }
            const Eigen::Vector3d interior = xlp.head<3>();
            const Eigen::VectorXd denom = blp - Alp.leftCols<3>() * interior;
            if (!(denom.array() > 0.0).all())
            {
                stats.failure_reason = "nonpositive_interior_slack";
                stats.solve_ms = elapsedMs(start);
                return false;
            }

            detail::HomGaugeCostData opt_data;
            opt_data.A = Alp.leftCols<3>().array().colwise() / denom.array();
            opt_data.alpha = options.alpha;
            opt_data.normalization_penalty_lambda = options.normalization_penalty_lambda;
            opt_data.min_positive_diagonal = options.min_positive_diagonal;
            opt_data.stats = &stats;

            Eigen::VectorXd x(9);
            x.head<3>() = ellipsoid.center - interior;
            lowerToVector3D(ellipsoidLowerTriangular(ellipsoid), x, 3);

            Eigen::VectorXd y = x;
            if (y.norm() > 1.0e-6)
            {
                y.normalize();
            }
            else
            {
                y.setZero();
                y(3) = 1.0;
                y(6) = 1.0;
                y(8) = 1.0;
                y.normalize();
            }

            double min_cost = 0.0;
            const int ret = lbfgs::lbfgs_optimize(y,
                                                  min_cost,
                                                  &detail::homGaugeCost,
                                                  nullptr,
                                                  &detail::homGaugeProgress,
                                                  &opt_data,
                                                  options.lbfgs_params);

            stats.status = ret;
            stats.objective_evaluations = opt_data.objective_evaluations;

            const Eigen::Matrix3d L_hat = vectorToLower3D(y, 3);
            const Eigen::MatrixX3d AL_hat = opt_data.A * L_hat;
            Eigen::VectorXd mu_exact(M);
            for (int i = 0; i < M; ++i)
            {
                mu_exact(i) = AL_hat.row(i).norm() + opt_data.A.row(i).dot(y.head<3>());
            }

            const double max_mu_hat = mu_exact.maxCoeff();
            if (!std::isfinite(max_mu_hat) || !(max_mu_hat > 0.0))
            {
                stats.failure_reason = "invalid_exact_recovery_scale";
                stats.solve_ms = elapsedMs(start);
                return false;
            }

            const double rho = (1.0 - options.recovery_margin) / max_mu_hat;
            x = rho * y;

            const Eigen::Vector3d p_shift = x.head<3>();
            const Eigen::Matrix3d L = vectorToLower3D(x, 3);

            stats.positive_diagonal = lowerPositiveDiagonal3D(L, options.min_positive_diagonal);
            stats.finite_output = allFinite(L) && p_shift.array().isFinite().all();
            fillMvieFeasibilityStats(opt_data.A, L, p_shift, options.active_constraint_tolerance, stats);

            if (stats.finite_output && stats.positive_diagonal)
            {
                svdToEllipsoid(L, p_shift + interior, ellipsoid);
            }

            stats.solve_ms = elapsedMs(start);
            stats.success = ret >= 0 &&
                            stats.finite_output &&
                            stats.positive_diagonal &&
                            stats.max_constraint_residual <= options.feasibility_tolerance;
            if (!stats.success && stats.failure_reason.empty())
            {
                if (ret < 0)
                {
                    stats.failure_reason = lbfgs::lbfgs_strerror(ret);
                }
                else if (!stats.finite_output)
                {
                    stats.failure_reason = "nonfinite_output";
                }
                else if (!stats.positive_diagonal)
                {
                    stats.failure_reason = "nonpositive_diagonal";
                }
                else
                {
                    stats.failure_reason = "infeasible_exact_recovery";
                }
            }
            return stats.success;
        }

    private:
        static double elapsedMs(const std::chrono::steady_clock::time_point &start)
        {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
    };
}

#endif
