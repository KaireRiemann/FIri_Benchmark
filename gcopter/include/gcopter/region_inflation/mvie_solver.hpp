#ifndef GCOPTER_REGION_INFLATION_MVIE_SOLVER_HPP
#define GCOPTER_REGION_INFLATION_MVIE_SOLVER_HPP

#include "gcopter/firi_mvie_diagnostics.hpp"
#include "gcopter/region_inflation/region_types.hpp"

#include <Eigen/Eigen>

#include <cmath>
#include <cstdint>
#include <cstring>

namespace region_inflation
{
    class IMvieSolver3D
    {
    public:
        virtual ~IMvieSolver3D() = default;

        virtual const char *name() const = 0;

        virtual bool solve(const Eigen::MatrixX4d &hpoly,
                           Ellipsoid3D &ellipsoid,
                           const MvieOptions &options,
                           MvieStats &stats) const = 0;
    };

    template <typename Derived>
    inline bool allFinite(const Eigen::MatrixBase<Derived> &x)
    {
        return x.array().isFinite().all();
    }

    inline Eigen::Matrix3d ellipsoidLowerTriangular(const Ellipsoid3D &ellipsoid)
    {
        const Eigen::Matrix3d q =
            ellipsoid.R * ellipsoid.radii.cwiseProduct(ellipsoid.radii).asDiagonal() * ellipsoid.R.transpose();
        Eigen::LLT<Eigen::Matrix3d> llt(q);
        if (llt.info() == Eigen::Success)
        {
            return llt.matrixL();
        }
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(q, Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Eigen::Vector3d s = svd.singularValues().cwiseMax(1.0e-16).cwiseSqrt();
        return svd.matrixU() * s.asDiagonal();
    }

    inline void lowerToVector3D(const Eigen::Matrix3d &L, Eigen::VectorXd &x, const int offset)
    {
        x(offset + 0) = L(0, 0);
        x(offset + 1) = L(1, 0);
        x(offset + 2) = L(2, 0);
        x(offset + 3) = L(1, 1);
        x(offset + 4) = L(2, 1);
        x(offset + 5) = L(2, 2);
    }

    inline Eigen::Matrix3d vectorToLower3D(const Eigen::VectorXd &x, const int offset)
    {
        Eigen::Matrix3d L = Eigen::Matrix3d::Zero();
        L(0, 0) = x(offset + 0);
        L(1, 0) = x(offset + 1);
        L(2, 0) = x(offset + 2);
        L(1, 1) = x(offset + 3);
        L(2, 1) = x(offset + 4);
        L(2, 2) = x(offset + 5);
        return L;
    }

    inline bool lowerPositiveDiagonal3D(const Eigen::Matrix3d &L, const double min_diag)
    {
        return L(0, 0) > min_diag && L(1, 1) > min_diag && L(2, 2) > min_diag;
    }

    inline std::uint64_t fnv1aBytes(const void *data, const std::size_t n, std::uint64_t hash = 1469598103934665603ULL)
    {
        const unsigned char *bytes = static_cast<const unsigned char *>(data);
        for (std::size_t i = 0; i < n; ++i)
        {
            hash ^= static_cast<std::uint64_t>(bytes[i]);
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    inline std::uint64_t hashDouble(const double value, std::uint64_t hash)
    {
        long long quantized = static_cast<long long>(std::llround(value * 1.0e9));
        return fnv1aBytes(&quantized, sizeof(quantized), hash);
    }

    template <typename Derived>
    inline std::uint64_t hashEigenDense(const Eigen::MatrixBase<Derived> &matrix, std::uint64_t hash = 1469598103934665603ULL)
    {
        const long long rows = static_cast<long long>(matrix.rows());
        const long long cols = static_cast<long long>(matrix.cols());
        hash = fnv1aBytes(&rows, sizeof(rows), hash);
        hash = fnv1aBytes(&cols, sizeof(cols), hash);
        for (int j = 0; j < matrix.cols(); ++j)
        {
            for (int i = 0; i < matrix.rows(); ++i)
            {
                hash = hashDouble(matrix(i, j), hash);
            }
        }
        return hash;
    }

    inline std::uint64_t hashEllipsoidWarmStart(const Ellipsoid3D &ellipsoid)
    {
        std::uint64_t h = hashEigenDense(ellipsoid.R);
        h = hashEigenDense(ellipsoid.center, h);
        h = hashEigenDense(ellipsoid.radii, h);
        return h;
    }

    inline void fillMvieFeasibilityStats(const Eigen::MatrixX3d &A,
                                         const Eigen::Matrix3d &L,
                                         const Eigen::Vector3d &p,
                                         const double active_tol,
                                         MvieStats &stats)
    {
        stats.logdet_l = firi_common::lowerTriangularLogDet(L);
        if (A.rows() == 0)
        {
            stats.max_mu = 0.0;
            stats.max_constraint_residual = 0.0;
            stats.active_constraint_count = 0;
            return;
        }
        const Eigen::VectorXd mu = (A * L).rowwise().norm() + A * p;
        stats.max_mu = mu.maxCoeff();
        stats.max_constraint_residual = stats.max_mu - 1.0;
        stats.active_constraint_count =
            (mu.array() >= (stats.max_mu - active_tol)).cast<int>().sum();
    }

    inline void svdToEllipsoid(const Eigen::Matrix3d &L,
                               const Eigen::Vector3d &p,
                               Ellipsoid3D &ellipsoid)
    {
        Eigen::JacobiSVD<Eigen::Matrix3d, Eigen::FullPivHouseholderQRPreconditioner> svd(L, Eigen::ComputeFullU);
        const Eigen::Matrix3d U = svd.matrixU();
        const Eigen::Vector3d S = svd.singularValues();
        ellipsoid.center = p;
        if (U.determinant() < 0.0)
        {
            ellipsoid.R.col(0) = U.col(1);
            ellipsoid.R.col(1) = U.col(0);
            ellipsoid.R.col(2) = U.col(2);
            ellipsoid.radii(0) = S(1);
            ellipsoid.radii(1) = S(0);
            ellipsoid.radii(2) = S(2);
        }
        else
        {
            ellipsoid.R = U;
            ellipsoid.radii = S;
        }
    }
}

#endif
