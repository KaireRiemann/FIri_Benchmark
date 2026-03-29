/*
    MIT License

    Copyright (c) 2021 Zhepei Wang (wangzhepei@live.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies or substantial portions of the Software.
*/

#ifndef FIRI_LBFGS_DEFAULTS_HPP
#define FIRI_LBFGS_DEFAULTS_HPP

#include "lbfgs.hpp"

namespace firi_common
{
    inline lbfgs::lbfgs_parameter_t defaultMVIELbfgsParameters()
    {
        lbfgs::lbfgs_parameter_t params;
        params.mem_size = 18;
        params.g_epsilon = 0.0;
        params.min_step = 1.0e-32;
        params.past = 3;
        params.delta = 1.0e-7;
        return params;
    }
}

#endif
