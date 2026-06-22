#ifndef GCOPTER_REGION_INFLATION_CONSTRAINT_BUILDER_HPP
#define GCOPTER_REGION_INFLATION_CONSTRAINT_BUILDER_HPP

#include "gcopter/region_inflation/region_types.hpp"

namespace region_inflation
{
    class ConstraintBuilder3D
    {
    public:
        virtual ~ConstraintBuilder3D() = default;

        virtual bool prepare(const RegionInput &input,
                             ConstraintBuildStats &stats) = 0;

        virtual bool build(const RegionInput &input,
                           const Ellipsoid3D &current,
                           Eigen::MatrixX4d &hpoly,
                           ConstraintBuildStats &stats) = 0;

        virtual bool finalize(const RegionInput &input,
                              const Ellipsoid3D &final_ellipsoid,
                              Eigen::MatrixX4d &hpoly,
                              ConstraintBuildStats &stats) = 0;
    };
}

#endif
