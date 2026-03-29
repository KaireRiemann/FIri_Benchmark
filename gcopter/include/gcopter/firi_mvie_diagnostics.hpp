/*
    MIT License

    Copyright (c) 2021 Zhepei Wang (wangzhepei@live.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies or substantial portions of the Software.
*/

#ifndef FIRI_MVIE_DIAGNOSTICS_HPP
#define FIRI_MVIE_DIAGNOSTICS_HPP

#include "lbfgs.hpp"

#include <Eigen/Eigen>

#include <cfloat>
#include <cmath>

namespace firi_common
{
    struct MVIEDiagnostic
    {
        bool optimizer_ran = true;
        int lbfgs_iterations = 0;
        int lbfgs_status = lbfgs::LBFGSERR_UNKNOWNERROR;
        double max_constraint_residual = 0.0;
        double max_mu = 0.0;
        double logdet_l = 0.0;
        int active_constraint_count = 0;
    };

    template <typename LType>
    inline double lowerTriangularLogDet(const LType &L)
    {
        double logdet = 0.0;
        for (int i = 0; i < L.rows(); ++i)
        {
            logdet += std::log(std::max<double>(L(i, i), DBL_MIN));
        }
        return logdet;
    }

    template <typename AType, typename LType, typename PType>
    inline void finalizeMVIEDiagnostic(const AType &A,
                                       const LType &L,
                                       const PType &p,
                                       MVIEDiagnostic *diagnostic,
                                       const double active_tol = 1.0e-3)
    {
        if (diagnostic == nullptr)
        {
            return;
        }

        diagnostic->logdet_l = lowerTriangularLogDet(L);

        if (A.rows() == 0)
        {
            diagnostic->max_mu = 0.0;
            diagnostic->max_constraint_residual = 0.0;
            diagnostic->active_constraint_count = 0;
            return;
        }

        const Eigen::VectorXd mu = (A * L).rowwise().norm() + A * p;
        diagnostic->max_mu = mu.maxCoeff();
        diagnostic->max_constraint_residual = diagnostic->max_mu - 1.0;
        diagnostic->active_constraint_count =
            (mu.array() >= (diagnostic->max_mu - active_tol)).cast<int>().sum();
    }
}

#endif
