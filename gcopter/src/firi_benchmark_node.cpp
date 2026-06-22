#include "gcopter/benchmark/corridor_benchmark.hpp"
#include "gcopter/benchmark/corridor_visualization.hpp"
#include "gcopter/benchmark/local_region_benchmark.hpp"
#include "gcopter/benchmark/planning_benchmark.hpp"
#include "gcopter/benchmark/seed_sampler.hpp"
#include "gcopter/sfc_gen.hpp"
#include "gcopter/voxel_map.hpp"

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <Eigen/Eigen>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace
{
std::string shellQuote(const std::string &value)
{
    std::string out = "'";
    for (const char c : value)
    {
        if (c == '\'')
        {
            out += "'\\''";
        }
        else
        {
            out += c;
        }
    }
    out += "'";
    return out;
}

void ensureDirectory(const std::string &path)
{
    const std::string cmd = "mkdir -p " + shellQuote(path);
    std::system(cmd.c_str());
}

std::string runCommand(const std::string &cmd)
{
    FILE *pipe = popen(cmd.c_str(), "r");
    if (pipe == nullptr)
    {
        return "unknown";
    }
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
    {
        result += buffer;
    }
    pclose(pipe);
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
    {
        result.pop_back();
    }
    return result.empty() ? "unknown" : result;
}

std::vector<double> getDoubleVectorParam(const ros::NodeHandle &nh,
                                         const std::string &name,
                                         const std::vector<double> &fallback)
{
    std::vector<double> values;
    if (nh.getParam(name, values))
    {
        return values;
    }
    return fallback;
}

std::string defaultRunId(const std::string &density,
                         const int map_seed,
                         const std::uint64_t master_seed)
{
    std::ostringstream oss;
    oss << density << "_map" << map_seed << "_master" << master_seed;
    return oss.str();
}

struct BenchmarkConfig
{
    std::string map_topic = "/voxel_map";
    std::string output_dir = "/tmp/firi_benchmark";
    std::string run_id;
    std::string density = "medium";
    std::string benchmark_mode = "all";
    std::string manifest_in;
    std::string manifest_out;
    int map_seed = 1024;
    std::uint64_t master_seed = 20260622ULL;
    double dilate_radius = 0.5;
    double voxel_width = 0.25;
    std::vector<double> map_bound = {-25.0, 25.0, -25.0, 25.0, 0.0, 5.0};
    double timeout_rrt = 0.02;
    double local_range = 3.0;
    double corridor_progress = 7.0;
    int point_seed_count = 50;
    int line_seed_count_per_length = 50;
    int planning_case_count = 20;
    int repeats = 5;
    int warmup_repeats = 1;
    int max_sampling_attempts = 10000;
    bool enable_visualization = false;
    int visualization_local_case_count = 8;
    std::string visualization_frame_id = "map";
    double planning_min_distance = 8.0;
    double planning_max_distance = 45.0;
    std::vector<double> line_lengths = {0.5, 1.5, 3.0};
    region_inflation::FiriOptions firi_options = region_inflation::FiriOptions::convergence();
    firi_benchmark::TrajectoryOptimizerConfig trajectory_config;

    explicit BenchmarkConfig(const ros::NodeHandle &nh)
    {
        nh.param("MapTopic", map_topic, map_topic);
        nh.param("OutputDir", output_dir, output_dir);
        nh.param("RunId", run_id, run_id);
        nh.param("DensityLabel", density, density);
        nh.param("BenchmarkMode", benchmark_mode, benchmark_mode);
        nh.param("ManifestIn", manifest_in, manifest_in);
        nh.param("ManifestOut", manifest_out, manifest_out);
        nh.param("MapSeed", map_seed, map_seed);
        int master_seed_i = static_cast<int>(master_seed);
        nh.param("MasterSeed", master_seed_i, master_seed_i);
        master_seed = static_cast<std::uint64_t>(master_seed_i);
        nh.param("DilateRadius", dilate_radius, dilate_radius);
        nh.param("VoxelWidth", voxel_width, voxel_width);
        nh.getParam("MapBound", map_bound);
        double map_size_x = map_bound[1] - map_bound[0];
        double map_size_y = map_bound[3] - map_bound[2];
        double map_size_z = map_bound[5] - map_bound[4];
        double map_x_origin = map_bound[0];
        double map_y_origin = map_bound[2];
        double map_z_origin = map_bound[4];
        const bool has_explicit_map_box =
            nh.getParam("MapSizeX", map_size_x) |
            nh.getParam("MapSizeY", map_size_y) |
            nh.getParam("MapSizeZ", map_size_z) |
            nh.getParam("MapOriginX", map_x_origin) |
            nh.getParam("MapOriginY", map_y_origin) |
            nh.getParam("MapOriginZ", map_z_origin);
        if (has_explicit_map_box)
        {
            map_bound = {map_x_origin,
                         map_x_origin + map_size_x,
                         map_y_origin,
                         map_y_origin + map_size_y,
                         map_z_origin,
                         map_z_origin + map_size_z};
        }
        nh.param("TimeoutRRT", timeout_rrt, timeout_rrt);
        nh.param("LocalRange", local_range, local_range);
        nh.param("CorridorProgress", corridor_progress, corridor_progress);
        nh.param("PointSeedCount", point_seed_count, point_seed_count);
        nh.param("LineSeedCountPerLength", line_seed_count_per_length, line_seed_count_per_length);
        nh.param("PlanningCaseCount", planning_case_count, planning_case_count);
        nh.param("Repeats", repeats, repeats);
        nh.param("WarmupRepeats", warmup_repeats, warmup_repeats);
        nh.param("MaxSamplingAttempts", max_sampling_attempts, max_sampling_attempts);
        nh.param("EnableVisualization", enable_visualization, enable_visualization);
        nh.param("VisualizationLocalCaseCount", visualization_local_case_count, visualization_local_case_count);
        nh.param("VisualizationFrameId", visualization_frame_id, visualization_frame_id);
        nh.param("PlanningMinDistance", planning_min_distance, planning_min_distance);
        nh.param("PlanningMaxDistance", planning_max_distance, planning_max_distance);
        line_lengths = getDoubleVectorParam(nh, "LineLengths", line_lengths);

        std::string outer_stop_mode = "convergence";
        nh.param("OuterStopMode", outer_stop_mode, outer_stop_mode);
        if (outer_stop_mode == "fixed" || outer_stop_mode == "compatibility")
        {
            firi_options = region_inflation::FiriOptions::compatibility();
        }
        else
        {
            firi_options = region_inflation::FiriOptions::convergence();
        }
        nh.param("MaxOuterIterations", firi_options.max_outer_iterations, firi_options.max_outer_iterations);
        nh.param("RelativeVolumeTolerance", firi_options.relative_volume_tolerance, firi_options.relative_volume_tolerance);
        nh.param("GeometricEpsilon", firi_options.geometric_epsilon, firi_options.geometric_epsilon);
        nh.param("EnableMonotonicAcceptance", firi_options.enable_monotonic_acceptance, firi_options.enable_monotonic_acceptance);
        nh.param("AcceptanceTolerance", firi_options.acceptance_tolerance, firi_options.acceptance_tolerance);
        nh.param("HomAlpha", firi_options.mvie_options.alpha, firi_options.mvie_options.alpha);
        nh.param("HomNormalizationPenaltyLambda",
                 firi_options.mvie_options.normalization_penalty_lambda,
                 firi_options.mvie_options.normalization_penalty_lambda);
        nh.param("HomMinPositiveDiagonal",
                 firi_options.mvie_options.min_positive_diagonal,
                 firi_options.mvie_options.min_positive_diagonal);
        nh.param("HomRecoveryMargin",
                 firi_options.mvie_options.recovery_margin,
                 firi_options.mvie_options.recovery_margin);
        nh.param("MvieFeasibilityTolerance",
                 firi_options.mvie_options.feasibility_tolerance,
                 firi_options.mvie_options.feasibility_tolerance);
        nh.param("MvieActiveConstraintTolerance",
                 firi_options.mvie_options.active_constraint_tolerance,
                 firi_options.mvie_options.active_constraint_tolerance);

        nh.param("MaxVelMag", trajectory_config.max_vel_mag, trajectory_config.max_vel_mag);
        nh.param("MaxBdrMag", trajectory_config.max_bdr_mag, trajectory_config.max_bdr_mag);
        nh.param("MaxTiltAngle", trajectory_config.max_tilt_angle, trajectory_config.max_tilt_angle);
        nh.param("MinThrust", trajectory_config.min_thrust, trajectory_config.min_thrust);
        nh.param("MaxThrust", trajectory_config.max_thrust, trajectory_config.max_thrust);
        nh.param("VehicleMass", trajectory_config.vehicle_mass, trajectory_config.vehicle_mass);
        nh.param("GravAcc", trajectory_config.grav_acc, trajectory_config.grav_acc);
        nh.param("HorizDrag", trajectory_config.horiz_drag, trajectory_config.horiz_drag);
        nh.param("VertDrag", trajectory_config.vert_drag, trajectory_config.vert_drag);
        nh.param("ParasDrag", trajectory_config.paras_drag, trajectory_config.paras_drag);
        nh.param("SpeedEps", trajectory_config.speed_eps, trajectory_config.speed_eps);
        nh.param("WeightT", trajectory_config.weight_t, trajectory_config.weight_t);
        std::vector<double> chi = getDoubleVectorParam(nh, "ChiVec", {1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e5});
        trajectory_config.chi_vec.resize(5);
        for (int i = 0; i < 5; ++i)
        {
            trajectory_config.chi_vec(i) = i < static_cast<int>(chi.size()) ? chi[i] : 1.0e4;
        }
        nh.param("SmoothingEps", trajectory_config.smoothing_eps, trajectory_config.smoothing_eps);
        nh.param("IntegralIntervs", trajectory_config.integral_intervals, trajectory_config.integral_intervals);
        nh.param("RelCostTol", trajectory_config.rel_cost_tol, trajectory_config.rel_cost_tol);
        nh.param("TrajectoryValidationDt", trajectory_config.validation_dt, trajectory_config.validation_dt);

        if (run_id.empty())
        {
            run_id = defaultRunId(density, map_seed, master_seed);
        }
        if (manifest_out.empty())
        {
            manifest_out = output_dir + "/case_manifest.csv";
        }
    }

    bool wants(const std::string &mode) const
    {
        return benchmark_mode == "all" || benchmark_mode == mode;
    }

    Eigen::Vector3d mapLow() const
    {
        return Eigen::Vector3d(map_bound[0], map_bound[2], map_bound[4]);
    }

    Eigen::Vector3d mapHigh() const
    {
        return Eigen::Vector3d(map_bound[1], map_bound[3], map_bound[5]);
    }

    std::string mapId() const
    {
        std::ostringstream oss;
        oss << density << "_seed_" << map_seed;
        return oss.str();
    }
};

class FiriBenchmarkNode
{
public:
    FiriBenchmarkNode(const BenchmarkConfig &config, ros::NodeHandle &nh)
        : config_(config), nh_(nh)
    {
        const Eigen::Vector3i xyz((config_.map_bound[1] - config_.map_bound[0]) / config_.voxel_width,
                                  (config_.map_bound[3] - config_.map_bound[2]) / config_.voxel_width,
                                  (config_.map_bound[5] - config_.map_bound[4]) / config_.voxel_width);
        voxel_map_ = voxel_map::VoxelMap(xyz, config_.mapLow(), config_.voxel_width);
        map_sub_ = nh_.subscribe(config_.map_topic, 1, &FiriBenchmarkNode::mapCallback, this,
                                 ros::TransportHints().tcpNoDelay());
        if (config_.enable_visualization)
        {
            local_corridor_pub_ =
                nh_.advertise<visualization_msgs::MarkerArray>("/firi_benchmark/local_corridors", 1, true);
            full_corridor_pub_ =
                nh_.advertise<visualization_msgs::MarkerArray>("/firi_benchmark/full_corridors", 1, true);
        }
    }

    bool ready() const
    {
        return map_initialized_;
    }

    bool run()
    {
        ensureDirectory(config_.output_dir);
        const std::vector<region_inflation::MethodConfig> methods =
            region_inflation::defaultMethodRegistry();

        firi_benchmark::SteadyTimer surf_timer;
        std::vector<Eigen::Vector3d> surface_points;
        voxel_map_.getSurf(surface_points);
        const double surface_extract_ms = surf_timer.elapsedMs();

        firi_benchmark::CsvWriter region_writer(config_.output_dir + "/region_trials.csv",
                                                firi_benchmark::regionTrialHeader());
        firi_benchmark::CsvWriter replay_writer(config_.output_dir + "/mvie_replay.csv",
                                                firi_benchmark::mvieReplayHeader());
        firi_benchmark::CsvWriter corridor_writer(config_.output_dir + "/corridor_trials.csv",
                                                  firi_benchmark::corridorTrialHeader());
        firi_benchmark::CsvWriter planning_writer(config_.output_dir + "/planning_trials.csv",
                                                  firi_benchmark::planningTrialHeader());
        firi_benchmark::CsvWriter manifest_writer(config_.manifest_out,
                                                  {"run_id", "map_id", "density", "case_id", "case_type",
                                                   "seed_type", "seed_length", "ax", "ay", "az", "bx", "by", "bz",
                                                   "input_hash", "deterministic_seed", "local_points"});

        std::vector<firi_benchmark::RegionCase> region_cases;
        std::vector<double> local_point_counts;
        generateRegionCases(surface_points, region_cases, local_point_counts, manifest_writer);

        if (config_.wants("region"))
        {
            firi_benchmark::runRegionTrials(config_.run_id,
                                            region_cases,
                                            methods,
                                            config_.firi_options,
                                            config_.repeats,
                                            region_writer);
        }
        if (config_.enable_visualization)
        {
            appendLocalVisualization(region_cases, methods);
        }

        if (config_.wants("mvie_replay"))
        {
            std::vector<region_inflation::MvieReplayCase> replay_cases;
            firi_benchmark::collectReplayCases(region_cases, config_.firi_options, replay_cases);
            firi_benchmark::runMvieReplayBenchmark(config_.run_id,
                                                   replay_cases,
                                                   methods,
                                                   config_.firi_options.mvie_options,
                                                   config_.repeats,
                                                   config_.warmup_repeats,
                                                   replay_writer);
        }

        std::vector<firi_benchmark::PlanningCase> planning_cases;
        generatePlanningCases(planning_cases, manifest_writer);
        if (config_.wants("corridor") || config_.wants("planning"))
        {
            runCorridorAndPlanning(surface_points,
                                   surface_extract_ms,
                                   planning_cases,
                                   methods,
                                   corridor_writer,
                                   planning_writer);
        }

        writeMetadata(surface_points.size(),
                      surface_extract_ms,
                      firi_benchmark::summarizeLocalPointCounts(local_point_counts));
        publishVisualization();
        return true;
    }

private:
    BenchmarkConfig config_;
    ros::NodeHandle nh_;
    ros::Subscriber map_sub_;
    ros::Publisher local_corridor_pub_;
    ros::Publisher full_corridor_pub_;
    visualization_msgs::MarkerArray local_markers_;
    visualization_msgs::MarkerArray full_markers_;
    voxel_map::VoxelMap voxel_map_;
    bool map_initialized_ = false;
    double map_preprocess_ms_ = 0.0;

    void mapCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        if (map_initialized_)
        {
            return;
        }
        firi_benchmark::SteadyTimer timer;
        const size_t total = msg->data.size() / msg->point_step;
        const float *fdata = reinterpret_cast<const float *>(&msg->data[0]);
        for (size_t i = 0; i < total; ++i)
        {
            const size_t cur = msg->point_step / sizeof(float) * i;
            if (std::isnan(fdata[cur + 0]) || std::isinf(fdata[cur + 0]) ||
                std::isnan(fdata[cur + 1]) || std::isinf(fdata[cur + 1]) ||
                std::isnan(fdata[cur + 2]) || std::isinf(fdata[cur + 2]))
            {
                continue;
            }
            voxel_map_.setOccupied(Eigen::Vector3d(fdata[cur + 0], fdata[cur + 1], fdata[cur + 2]));
        }
        voxel_map_.dilate(std::ceil(config_.dilate_radius / voxel_map_.getScale()));
        map_preprocess_ms_ = timer.elapsedMs();
        map_initialized_ = true;
    }

    void generateRegionCases(const std::vector<Eigen::Vector3d> &surface_points,
                             std::vector<firi_benchmark::RegionCase> &cases,
                             std::vector<double> &local_point_counts,
                             firi_benchmark::CsvWriter &manifest_writer)
    {
        const Eigen::Vector3d map_low = config_.mapLow();
        const Eigen::Vector3d map_high = config_.mapHigh();
        const std::string map_id = config_.mapId();

        for (int i = 0; i < config_.point_seed_count; ++i)
        {
            firi_benchmark::RegionCase region_case;
            const std::uint64_t seed = firi_benchmark::mixSeed(config_.master_seed, 100000ULL + i);
            if (!firi_benchmark::makePointRegionCase(voxel_map_, surface_points, map_low, map_high,
                                                     map_id, config_.density,
                                                     firi_benchmark::caseId("point", i),
                                                     config_.local_range,
                                                     config_.max_sampling_attempts,
                                                     seed,
                                                     region_case))
            {
                ROS_ERROR_STREAM("Failed to sample required point region case " << i);
                continue;
            }
            cases.push_back(region_case);
            local_point_counts.push_back(region_case.local_points.cols());
            writeManifestRegion(manifest_writer, region_case, "region");
        }

        for (double length : config_.line_lengths)
        {
            for (int i = 0; i < config_.line_seed_count_per_length; ++i)
            {
                firi_benchmark::RegionCase region_case;
                const std::uint64_t seed =
                    firi_benchmark::mixSeed(config_.master_seed,
                                            200000ULL + static_cast<std::uint64_t>(std::llround(length * 1000.0)) * 1000ULL + i);
                if (!firi_benchmark::makeLineRegionCase(voxel_map_, surface_points, map_low, map_high,
                                                        map_id, config_.density,
                                                        firi_benchmark::caseId("line", static_cast<int>(cases.size())),
                                                        config_.local_range,
                                                        length,
                                                        config_.max_sampling_attempts,
                                                        seed,
                                                        region_case))
                {
                    ROS_ERROR_STREAM("Failed to sample required line region case length=" << length << " index=" << i);
                    continue;
                }
                cases.push_back(region_case);
                local_point_counts.push_back(region_case.local_points.cols());
                writeManifestRegion(manifest_writer, region_case, "region");
            }
        }
    }

    void generatePlanningCases(std::vector<firi_benchmark::PlanningCase> &cases,
                               firi_benchmark::CsvWriter &manifest_writer)
    {
        const Eigen::Vector3d map_low = config_.mapLow();
        const Eigen::Vector3d map_high = config_.mapHigh();
        for (int i = 0; i < config_.planning_case_count; ++i)
        {
            firi_benchmark::PlanningCase planning_case;
            const std::uint64_t seed = firi_benchmark::mixSeed(config_.master_seed, 300000ULL + i);
            if (!firi_benchmark::samplePlanningCase(voxel_map_,
                                                    map_low,
                                                    map_high,
                                                    config_.local_range,
                                                    config_.planning_min_distance,
                                                    config_.planning_max_distance,
                                                    config_.max_sampling_attempts,
                                                    firi_benchmark::caseId("planning", i),
                                                    seed,
                                                    planning_case))
            {
                ROS_ERROR_STREAM("Failed to sample required planning case " << i);
                continue;
            }
            cases.push_back(planning_case);
            manifest_writer.writeRow(config_.run_id, config_.mapId(), config_.density,
                                     planning_case.case_id, "planning", "start_goal", 0.0,
                                     planning_case.start.x(), planning_case.start.y(), planning_case.start.z(),
                                     planning_case.goal.x(), planning_case.goal.y(), planning_case.goal.z(),
                                     0, planning_case.deterministic_seed, 0);
        }
    }

    void runCorridorAndPlanning(const std::vector<Eigen::Vector3d> &surface_points,
                                const double surface_extract_ms,
                                const std::vector<firi_benchmark::PlanningCase> &planning_cases,
                                const std::vector<region_inflation::MethodConfig> &methods,
                                firi_benchmark::CsvWriter &corridor_writer,
                                firi_benchmark::CsvWriter &planning_writer)
    {
        const Eigen::Vector3d map_low = config_.mapLow();
        const Eigen::Vector3d map_high = config_.mapHigh();
        for (const firi_benchmark::PlanningCase &planning_case : planning_cases)
        {
            firi_benchmark::SteadyTimer path_timer;
            std::vector<Eigen::Vector3d> route;
            const double path_cost = sfc_gen::planPath<voxel_map::VoxelMap>(planning_case.start,
                                                                            planning_case.goal,
                                                                            voxel_map_.getOrigin(),
                                                                            voxel_map_.getCorner(),
                                                                            &voxel_map_,
                                                                            config_.timeout_rrt,
                                                                            route);
            const double path_search_ms = path_timer.elapsedMs();
            const bool path_success = std::isfinite(path_cost) && route.size() > 1;
            const std::uint64_t route_hash = route.empty() ? 0 :
                region_inflation::hashEigenDense(firi_benchmark::makePointMatrix(route));
            const double route_length = firi_benchmark::routeLength(route);
            const std::vector<firi_benchmark::PathSeedCase> seeds = path_success ?
                firi_benchmark::makePathSeedCases(route,
                                                  surface_points,
                                                  map_low,
                                                  map_high,
                                                  config_.corridor_progress,
                                                  config_.local_range) :
                std::vector<firi_benchmark::PathSeedCase>();

            if (config_.wants("corridor") && path_success)
            {
                runCorridorTrialsForRoute(planning_case,
                                          seeds,
                                          methods,
                                          route_hash,
                                          route.size(),
                                          route_length,
                                          corridor_writer);
            }

            runPlanningTrialsForRoute(planning_case,
                                      seeds,
                                      methods,
                                      path_success,
                                      path_search_ms,
                                      surface_extract_ms,
                                      route_hash,
                                      route.size(),
                                      route_length,
                                      planning_writer);
        }
    }

    void runCorridorTrialsForRoute(const firi_benchmark::PlanningCase &planning_case,
                                   const std::vector<firi_benchmark::PathSeedCase> &seeds,
                                   const std::vector<region_inflation::MethodConfig> &methods,
                                   const std::uint64_t route_hash,
                                   const std::size_t route_points,
                                   const double route_length,
                                   firi_benchmark::CsvWriter &corridor_writer)
    {
        for (int repeat = 0; repeat < config_.repeats; ++repeat)
        {
            for (const auto &method : methods)
            {
                const firi_benchmark::CorridorTrialStats fixed_stats =
                    firi_benchmark::runFixedSeedCorridor(seeds,
                                                         config_.mapId(),
                                                         config_.density,
                                                         method,
                                                         config_.firi_options,
                                                         0);
                firi_benchmark::writeCorridorTrialRow(corridor_writer,
                                                      config_.run_id,
                                                      config_.mapId(),
                                                      config_.density,
                                                      planning_case.case_id,
                                                      "fixed_seed_corridor",
                                                      method,
                                                      repeat,
                                                      route_hash,
                                                      route_points,
                                                      route_length,
                                                      fixed_stats);

                const firi_benchmark::CorridorTrialStats current_stats =
                    firi_benchmark::runCurrentCoverCorridor(seeds,
                                                            config_.mapId(),
                                                            config_.density,
                                                            method,
                                                            config_.firi_options,
                                                            0);
                if (config_.enable_visualization && repeat == 0)
                {
                    appendMarkerArray(full_markers_,
                                      firi_benchmark::makeCorridorMarkers(
                                          current_stats.hpolys,
                                          config_.visualization_frame_id,
                                          "full_" + planning_case.case_id + "_" + method.name,
                                          colorForMethod(method.name)));
                }
                firi_benchmark::writeCorridorTrialRow(corridor_writer,
                                                      config_.run_id,
                                                      config_.mapId(),
                                                      config_.density,
                                                      planning_case.case_id,
                                                      "current_cover",
                                                      method,
                                                      repeat,
                                                      route_hash,
                                                      route_points,
                                                      route_length,
                                                      current_stats);
            }
        }
    }

    void runPlanningTrialsForRoute(const firi_benchmark::PlanningCase &planning_case,
                                   const std::vector<firi_benchmark::PathSeedCase> &seeds,
                                   const std::vector<region_inflation::MethodConfig> &methods,
                                   const bool path_success,
                                   const double path_search_ms,
                                   const double surface_extract_ms,
                                   const std::uint64_t route_hash,
                                   const std::size_t route_points,
                                   const double route_length,
                                   firi_benchmark::CsvWriter &planning_writer)
    {
        if (!config_.wants("planning"))
        {
            return;
        }
        for (int repeat = 0; repeat < config_.repeats; ++repeat)
        {
            std::vector<int> order(methods.size());
            for (std::size_t i = 0; i < methods.size(); ++i)
            {
                order[i] = static_cast<int>(i);
            }
            if (((planning_case.deterministic_seed + static_cast<std::uint64_t>(repeat)) & 1ULL) != 0ULL)
            {
                std::reverse(order.begin(), order.end());
            }

            for (std::size_t order_index = 0; order_index < order.size(); ++order_index)
            {
                const region_inflation::MethodConfig &method = methods[order[order_index]];
                firi_benchmark::CorridorTrialStats corridor_stats;
                firi_benchmark::TrajectoryTrialResult traj_result;
                bool success = false;
                std::string failure_stage;
                std::string failure_reason;
                if (!path_success)
                {
                    failure_stage = "path";
                    failure_reason = "path_search_failed";
                }
                else
                {
                    corridor_stats = firi_benchmark::runCurrentCoverCorridor(seeds,
                                                                              config_.mapId(),
                                                                              config_.density,
                                                                              method,
                                                                              config_.firi_options,
                                                                              0);
                    if (!corridor_stats.success)
                    {
                        failure_stage = "corridor";
                        failure_reason = corridor_stats.failure_reason;
                    }
                    else
                    {
                        traj_result = firi_benchmark::optimizeTrajectoryInCorridor(planning_case.start,
                                                                                   planning_case.goal,
                                                                                   corridor_stats.hpolys,
                                                                                   config_.trajectory_config,
                                                                                   voxel_map_);
                        if (config_.enable_visualization && repeat == 0)
                        {
                            appendMarkerArray(full_markers_,
                                              firi_benchmark::makeCorridorMarkers(
                                                  corridor_stats.hpolys,
                                                  config_.visualization_frame_id,
                                                  "full_" + planning_case.case_id + "_" + method.name,
                                                  colorForMethod(method.name)));
                        }
                        success = traj_result.setup_success &&
                                  traj_result.optimize_success &&
                                  traj_result.pieces > 0 &&
                                  traj_result.collision_free;
                        if (!success)
                        {
                            failure_stage = traj_result.setup_success ? "trajectory_optimize" : "trajectory_setup";
                            failure_reason = traj_result.failure_reason;
                        }
                    }
                }

                const double planning_backend_ms = corridor_stats.corridor_total_ms +
                                                   traj_result.setup_ms +
                                                   traj_result.optimize_ms;
                const double end_to_end_ms = path_search_ms + surface_extract_ms + planning_backend_ms;
                planning_writer.writeRow(config_.run_id,
                                         config_.mapId(),
                                         config_.density,
                                         planning_case.case_id,
                                         method.name,
                                         repeat,
                                         order_index,
                                         planning_case.start.x(),
                                         planning_case.start.y(),
                                         planning_case.start.z(),
                                         planning_case.goal.x(),
                                         planning_case.goal.y(),
                                         planning_case.goal.z(),
                                         route_hash,
                                         route_points,
                                         route_length,
                                         path_success,
                                         path_search_ms,
                                         surface_extract_ms,
                                         corridor_stats.success,
                                         corridor_stats.regions_after_shortcut,
                                         corridor_stats.corridor_total_ms,
                                         traj_result.setup_success,
                                         traj_result.setup_ms,
                                         traj_result.optimize_success,
                                         traj_result.optimize_ms,
                                         traj_result.cost,
                                         traj_result.duration,
                                         traj_result.pieces,
                                         planning_backend_ms,
                                         end_to_end_ms,
                                         traj_result.sampled_collision_count,
                                         traj_result.collision_free,
                                         success,
                                         failure_stage,
                                         failure_reason);
            }
        }
    }

    void writeManifestRegion(firi_benchmark::CsvWriter &manifest_writer,
                             const firi_benchmark::RegionCase &region_case,
                             const std::string &case_type)
    {
        manifest_writer.writeRow(config_.run_id, region_case.map_id, region_case.density,
                                 region_case.case_id, case_type, region_case.seed_type,
                                 region_case.seed_length,
                                 region_case.a.x(), region_case.a.y(), region_case.a.z(),
                                 region_case.b.x(), region_case.b.y(), region_case.b.z(),
                                 region_case.input_hash,
                                 region_case.deterministic_seed,
                                 region_case.local_points.cols());
    }

    void appendLocalVisualization(const std::vector<firi_benchmark::RegionCase> &region_cases,
                                  const std::vector<region_inflation::MethodConfig> &methods)
    {
        const int count = std::min<int>(config_.visualization_local_case_count, region_cases.size());
        for (int i = 0; i < count; ++i)
        {
            for (const auto &method : methods)
            {
                region_inflation::RegionOutput output;
                region_inflation::RegionStats stats;
                firi_benchmark::runRegionMethod(region_cases[i], method, config_.firi_options, output, stats, nullptr);
                if (stats.success)
                {
                    appendMarkerArray(local_markers_,
                                      firi_benchmark::makeCorridorMarkers(
                                          std::vector<Eigen::MatrixX4d>{output.hpoly},
                                          config_.visualization_frame_id,
                                          "local_" + region_cases[i].case_id + "_" + method.name,
                                          colorForMethod(method.name)));
                }
            }
        }
    }

    static firi_benchmark::Rgba colorForMethod(const std::string &method)
    {
        if (method == "firi_hom")
        {
            return {0.0, 0.75, 0.25, 0.18};
        }
        return {0.0, 0.25, 1.0, 0.18};
    }

    static void appendMarkerArray(visualization_msgs::MarkerArray &dst,
                                  const visualization_msgs::MarkerArray &src)
    {
        dst.markers.insert(dst.markers.end(), src.markers.begin(), src.markers.end());
    }

    void publishVisualization()
    {
        if (!config_.enable_visualization)
        {
            return;
        }
        local_corridor_pub_.publish(local_markers_);
        full_corridor_pub_.publish(full_markers_);
        ros::Duration(0.2).sleep();
    }

    void writeMetadata(const std::size_t global_surface_points,
                       const double surface_extract_ms,
                       const firi_benchmark::LocalPointStats &local_stats)
    {
        char hostname[256];
        hostname[0] = '\0';
        gethostname(hostname, sizeof(hostname) - 1);

        std::ofstream os((config_.output_dir + "/metadata.json").c_str());
        os << "{\n";
        os << "  \"run_id\": \"" << config_.run_id << "\",\n";
        os << "  \"git_commit\": \"" << runCommand("git rev-parse HEAD 2>/dev/null") << "\",\n";
        os << "  \"git_dirty_status\": \"" << runCommand("git status --short 2>/dev/null | wc -l") << "\",\n";
        os << "  \"build_type\": \"Release\",\n";
        os << "  \"compiler_version\": \"" << __VERSION__ << "\",\n";
        os << "  \"eigen_version\": \"" << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << "\",\n";
        os << "  \"ros_distribution\": \"" << (std::getenv("ROS_DISTRO") ? std::getenv("ROS_DISTRO") : "unknown") << "\",\n";
        os << "  \"hostname\": \"" << hostname << "\",\n";
        os << "  \"density\": \"" << config_.density << "\",\n";
        os << "  \"map_seed\": " << config_.map_seed << ",\n";
        os << "  \"master_seed\": " << config_.master_seed << ",\n";
        os << "  \"map_preprocess_ms\": " << map_preprocess_ms_ << ",\n";
        os << "  \"surface_extract_ms\": " << surface_extract_ms << ",\n";
        os << "  \"global_surface_point_count\": " << global_surface_points << ",\n";
        os << "  \"local_point_count_mean\": " << local_stats.mean << ",\n";
        os << "  \"local_point_count_std\": " << local_stats.stddev << ",\n";
        os << "  \"local_point_count_median\": " << local_stats.median << ",\n";
        os << "  \"local_point_count_p95\": " << local_stats.p95 << ",\n";
        os << "  \"methods\": [\"firi_legacy\", \"firi_hom\"],\n";
        os << "  \"hom_solver\": {\"alpha\": " << config_.firi_options.mvie_options.alpha
           << ", \"lambda\": " << config_.firi_options.mvie_options.normalization_penalty_lambda
           << ", \"recovery_margin\": " << config_.firi_options.mvie_options.recovery_margin
           << ", \"min_positive_diagonal\": " << config_.firi_options.mvie_options.min_positive_diagonal << "}\n";
        os << "}\n";
    }
};
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "firi_benchmark_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    BenchmarkConfig config(pnh);

    FiriBenchmarkNode node(config, nh);
    ros::Rate rate(50);
    while (ros::ok() && !node.ready())
    {
        ros::spinOnce();
        rate.sleep();
    }
    if (!ros::ok())
    {
        return 1;
    }

    const bool ok = node.run();
    ros::shutdown();
    return ok ? 0 : 1;
}
