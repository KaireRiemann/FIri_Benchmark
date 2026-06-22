#ifndef GCOPTER_REGION_INFLATION_REGION_TYPES_HPP
#define GCOPTER_REGION_INFLATION_REGION_TYPES_HPP

#include "gcopter/firi_lbfgs_defaults.hpp"

#include <Eigen/Eigen>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace region_inflation
{
    struct Ellipsoid3D
    {
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        Eigen::Vector3d radii = Eigen::Vector3d::Ones();

        inline double volumeProxy() const
        {
            return radii.prod();
        }

        inline double logDetL() const
        {
            return std::log(std::max(radii(0), std::numeric_limits<double>::min())) +
                   std::log(std::max(radii(1), std::numeric_limits<double>::min())) +
                   std::log(std::max(radii(2), std::numeric_limits<double>::min()));
        }
    };

    enum class OuterStopMode
    {
        FixedIterations,
        RelativeEllipsoidVolume
    };

    struct MvieOptions
    {
        lbfgs::lbfgs_parameter_t lbfgs_params = firi_common::defaultMVIELbfgsParameters();
        double alpha = 50.0;
        double normalization_penalty_lambda = 500.0;
        double min_positive_diagonal = 1.0e-7;
        double recovery_margin = 1.0e-10;
        double feasibility_tolerance = 1.0e-7;
        double active_constraint_tolerance = 1.0e-3;
        bool collect_detailed_diagnostics = true;
    };

    struct FiriOptions
    {
        int max_outer_iterations = 4;
        double relative_volume_tolerance = 0.02;
        double geometric_epsilon = 1.0e-6;
        bool enable_monotonic_acceptance = false;
        double acceptance_tolerance = 1.0e-9;
        OuterStopMode outer_stop_mode = OuterStopMode::FixedIterations;
        MvieOptions mvie_options;

        inline static FiriOptions compatibility()
        {
            FiriOptions options;
            options.max_outer_iterations = 4;
            options.outer_stop_mode = OuterStopMode::FixedIterations;
            return options;
        }

        inline static FiriOptions convergence()
        {
            FiriOptions options;
            options.max_outer_iterations = 10;
            options.relative_volume_tolerance = 0.02;
            options.outer_stop_mode = OuterStopMode::RelativeEllipsoidVolume;
            return options;
        }
    };

    struct MvieStats
    {
        bool success = false;
        int status = lbfgs::LBFGSERR_UNKNOWNERROR;
        int iterations = 0;
        int objective_evaluations = 0;
        double solve_ms = 0.0;
        double logdet_l = 0.0;
        double max_mu = 0.0;
        double max_constraint_residual = 0.0;
        int active_constraint_count = 0;
        int facet_count = 0;
        bool finite_output = false;
        bool positive_diagonal = false;
        std::string failure_reason;
    };

    struct ConstraintBuildStats
    {
        double prepare_ms = 0.0;
        double build_ms = 0.0;
        double certification_ms = 0.0;
        double repair_ms = 0.0;
        std::size_t full_point_count = 0;
        std::size_t working_point_count = 0;
        int generated_plane_count = 0;
        int repair_count = 0;
    };

    struct RegionInput
    {
        Eigen::MatrixX4d boundary;
        Eigen::Matrix3Xd point_cloud;
        Eigen::Vector3d a = Eigen::Vector3d::Zero();
        Eigen::Vector3d b = Eigen::Vector3d::Zero();
        std::string map_id;
        std::string density;
        std::string region_case_id;
        std::string seed_type;
        double seed_length = 0.0;
        std::uint64_t input_hash = 0;
        std::size_t global_surface_points = 0;
    };

    struct RegionOutput
    {
        Eigen::MatrixX4d hpoly;
        Ellipsoid3D final_ellipsoid;
    };

    struct MvieReplayCase
    {
        Eigen::MatrixX4d hpoly;
        Ellipsoid3D warm_start;
        std::string map_id;
        std::string density;
        std::string region_case_id;
        int outer_iteration = 0;
        std::string seed_type;
        double seed_length = 0.0;
        std::size_t local_point_count = 0;
        std::uint64_t input_hash = 0;
        std::uint64_t warm_start_hash = 0;
    };

    struct RegionStats
    {
        bool success = false;
        std::string failure_stage;
        std::string failure_reason;
        int outer_build_count = 0;
        int mvie_call_count = 0;
        int total_mvie_iterations = 0;
        int total_objective_evaluations = 0;
        double total_region_ms = 0.0;
        double total_constraint_build_ms = 0.0;
        double total_mvie_ms = 0.0;
        double final_logdet = 0.0;
        int rejected_mvie_updates = 0;
        ConstraintBuildStats constraint_stats;
        std::vector<MvieStats> mvie_calls;
    };

    enum class ConstraintBuilderKind
    {
        FullFiri
        // Future: StarFilter
    };

    enum class MvieSolverKind
    {
        LegacyPenalty,
        HomGauge
        // Optional reference: Socp
    };

    struct MethodConfig
    {
        std::string name;
        ConstraintBuilderKind constraint_builder = ConstraintBuilderKind::FullFiri;
        MvieSolverKind mvie_solver = MvieSolverKind::LegacyPenalty;
    };

    inline std::vector<MethodConfig> defaultMethodRegistry()
    {
        return {
            {"firi_legacy", ConstraintBuilderKind::FullFiri, MvieSolverKind::LegacyPenalty},
            {"firi_hom", ConstraintBuilderKind::FullFiri, MvieSolverKind::HomGauge}};
    }
}

#endif
