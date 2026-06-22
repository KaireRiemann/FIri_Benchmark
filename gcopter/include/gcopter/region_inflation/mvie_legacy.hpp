#ifndef GCOPTER_REGION_INFLATION_MVIE_LEGACY_HPP
#define GCOPTER_REGION_INFLATION_MVIE_LEGACY_HPP

#include "gcopter/firi.hpp"
#include "gcopter/region_inflation/mvie_solver.hpp"
#include "gcopter/sdlp.hpp"

#include <chrono>
#include <cfloat>
#include <cmath>

namespace region_inflation
{
    namespace detail
    {
        struct LegacyCostData
        {
            firi::MVIECostData inner;
            int objective_evaluations = 0;
            MvieStats *stats = nullptr;
        };

        inline double legacyCost(void *data,
                                 const Eigen::VectorXd &x,
                                 Eigen::VectorXd &grad)
        {
            LegacyCostData *cost_data = static_cast<LegacyCostData *>(data);
            ++cost_data->objective_evaluations;
            return firi::costMVIE(&cost_data->inner, x, grad);
        }

        inline int legacyProgress(void *data,
                                  const Eigen::VectorXd &,
                                  const Eigen::VectorXd &,
                                  const double,
                                  const double,
                                  const int k,
                                  const int)
        {
            LegacyCostData *cost_data = static_cast<LegacyCostData *>(data);
            if (cost_data != nullptr && cost_data->stats != nullptr)
            {
                cost_data->stats->iterations = k;
            }
            return 0;
        }
    }

    class LegacyPenaltyMvieSolver3D final : public IMvieSolver3D
    {
    public:
        const char *name() const override
        {
            return "firi_legacy";
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

            detail::LegacyCostData opt_data;
            opt_data.inner.A = Alp.leftCols<3>().array().colwise() / denom.array();
            opt_data.stats = &stats;

            Eigen::VectorXd x(9);
            Eigen::Matrix3d L;
            const Eigen::Matrix3d Q =
                ellipsoid.R * ellipsoid.radii.cwiseProduct(ellipsoid.radii).asDiagonal() * ellipsoid.R.transpose();
            firi::chol3d(Q, L);

            x.head<3>() = ellipsoid.center - interior;
            x(3) = std::sqrt(std::max(L(0, 0), DBL_MIN));
            x(4) = std::sqrt(std::max(L(1, 1), DBL_MIN));
            x(5) = std::sqrt(std::max(L(2, 2), DBL_MIN));
            x(6) = L(1, 0);
            x(7) = L(2, 1);
            x(8) = L(2, 0);

            double min_cost = 0.0;
            const int ret = lbfgs::lbfgs_optimize(x,
                                                  min_cost,
                                                  &detail::legacyCost,
                                                  nullptr,
                                                  &detail::legacyProgress,
                                                  &opt_data,
                                                  options.lbfgs_params);

            stats.status = ret;
            stats.objective_evaluations = opt_data.objective_evaluations;

            const Eigen::Vector3d p_shift = x.head<3>();
            L.setZero();
            L(0, 0) = x(3) * x(3);
            L(1, 0) = x(6);
            L(1, 1) = x(4) * x(4);
            L(2, 0) = x(8);
            L(2, 1) = x(7);
            L(2, 2) = x(5) * x(5);

            stats.positive_diagonal = lowerPositiveDiagonal3D(L, 0.0);
            stats.finite_output = allFinite(L) && p_shift.array().isFinite().all();
            fillMvieFeasibilityStats(opt_data.inner.A, L, p_shift, options.active_constraint_tolerance, stats);

            if (stats.finite_output && stats.positive_diagonal)
            {
                svdToEllipsoid(L, p_shift + interior, ellipsoid);
            }

            stats.solve_ms = elapsedMs(start);
            stats.success = ret >= 0 && stats.finite_output && stats.positive_diagonal;
            if (!stats.success && stats.failure_reason.empty())
            {
                stats.failure_reason = ret < 0 ? lbfgs::lbfgs_strerror(ret) : "invalid_output";
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
