#ifndef GCOPTER_BENCHMARK_CORRIDOR_VISUALIZATION_HPP
#define GCOPTER_BENCHMARK_CORRIDOR_VISUALIZATION_HPP

#include "gcopter/benchmark/benchmark_types.hpp"

#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>

#include <algorithm>
#include <string>
#include <vector>

namespace firi_benchmark
{
    struct Rgba
    {
        double r = 0.0;
        double g = 0.0;
        double b = 1.0;
        double a = 0.25;
    };

    inline visualization_msgs::Marker makeDeleteAllMarker(const std::string &frame_id,
                                                          const std::string &ns)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();
        marker.ns = ns;
        marker.id = 0;
        marker.action = visualization_msgs::Marker::DELETEALL;
        return marker;
    }

    inline geometry_msgs::Point toPointMsg(const Eigen::Vector3d &p)
    {
        geometry_msgs::Point out;
        out.x = p.x();
        out.y = p.y();
        out.z = p.z();
        return out;
    }

    inline visualization_msgs::Marker makePolytopeLineMarker(const Eigen::MatrixX4d &hpoly,
                                                             const std::string &frame_id,
                                                             const std::string &ns,
                                                             const int id,
                                                             const Rgba &color)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();
        marker.ns = ns;
        marker.id = id;
        marker.type = visualization_msgs::Marker::LINE_LIST;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.025;
        marker.color.r = color.r;
        marker.color.g = color.g;
        marker.color.b = color.b;
        marker.color.a = std::min(1.0, std::max(0.05, color.a + 0.45));

        Eigen::Matrix3Xd vertices;
        if (!geo_utils::enumerateVs(hpoly, vertices))
        {
            return marker;
        }

        const double tol = 1.0e-5;
        for (int i = 0; i < vertices.cols(); ++i)
        {
            for (int j = i + 1; j < vertices.cols(); ++j)
            {
                int common_planes = 0;
                const Eigen::Vector4d pi(vertices(0, i), vertices(1, i), vertices(2, i), 1.0);
                const Eigen::Vector4d pj(vertices(0, j), vertices(1, j), vertices(2, j), 1.0);
                for (int r = 0; r < hpoly.rows(); ++r)
                {
                    if (std::abs(hpoly.row(r).dot(pi)) < tol &&
                        std::abs(hpoly.row(r).dot(pj)) < tol)
                    {
                        ++common_planes;
                    }
                }
                if (common_planes >= 2)
                {
                    marker.points.push_back(toPointMsg(vertices.col(i)));
                    marker.points.push_back(toPointMsg(vertices.col(j)));
                }
            }
        }
        return marker;
    }

    inline visualization_msgs::Marker makePolytopeMeshMarker(const Eigen::MatrixX4d &hpoly,
                                                             const std::string &frame_id,
                                                             const std::string &ns,
                                                             const int id,
                                                             const Rgba &color)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();
        marker.ns = ns;
        marker.id = id;
        marker.type = visualization_msgs::Marker::TRIANGLE_LIST;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 1.0;
        marker.scale.y = 1.0;
        marker.scale.z = 1.0;
        marker.color.r = color.r;
        marker.color.g = color.g;
        marker.color.b = color.b;
        marker.color.a = color.a;

        Eigen::Matrix3Xd vertices;
        if (!geo_utils::enumerateVs(hpoly, vertices) || vertices.cols() < 4)
        {
            return marker;
        }
        quickhull::QuickHull<double> qh;
        const auto hull = qh.getConvexHull(vertices.data(), vertices.cols(), false, true);
        const auto &idx = hull.getIndexBuffer();
        for (std::size_t i = 0; i + 2 < idx.size(); i += 3)
        {
            marker.points.push_back(toPointMsg(vertices.col(idx[i + 0])));
            marker.points.push_back(toPointMsg(vertices.col(idx[i + 1])));
            marker.points.push_back(toPointMsg(vertices.col(idx[i + 2])));
        }
        return marker;
    }

    inline visualization_msgs::MarkerArray makeCorridorMarkers(const std::vector<Eigen::MatrixX4d> &hpolys,
                                                               const std::string &frame_id,
                                                               const std::string &ns,
                                                               const Rgba &color)
    {
        visualization_msgs::MarkerArray array;
        array.markers.push_back(makeDeleteAllMarker(frame_id, ns));
        int id = 1;
        for (const Eigen::MatrixX4d &hpoly : hpolys)
        {
            array.markers.push_back(makePolytopeMeshMarker(hpoly, frame_id, ns + "_mesh", id++, color));
            array.markers.push_back(makePolytopeLineMarker(hpoly, frame_id, ns + "_edge", id++, color));
        }
        return array;
    }
}

#endif
