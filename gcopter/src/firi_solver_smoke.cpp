#include "gcopter/firi.hpp"
#include "gcopter/firi_opt.hpp"
#include "gcopter/firi_socp.hpp"

#include <Eigen/Eigen>

#include <iostream>
#include <vector>

namespace
{
Eigen::Matrix<double, 6, 4> makeBoundary(const Eigen::Vector3d &low,
                                         const Eigen::Vector3d &high)
{
    Eigen::Matrix<double, 6, 4> bd = Eigen::Matrix<double, 6, 4>::Zero();
    bd(0, 0) = 1.0;
    bd(1, 0) = -1.0;
    bd(2, 1) = 1.0;
    bd(3, 1) = -1.0;
    bd(4, 2) = 1.0;
    bd(5, 2) = -1.0;
    bd(0, 3) = -high.x();
    bd(1, 3) = low.x();
    bd(2, 3) = -high.y();
    bd(3, 3) = low.y();
    bd(4, 3) = -high.z();
    bd(5, 3) = low.z();
    return bd;
}

Eigen::Matrix3Xd makePointCloud(const std::vector<Eigen::Vector3d> &pts)
{
    Eigen::Matrix3Xd pc(3, pts.size());
    for (std::size_t i = 0; i < pts.size(); ++i)
    {
        pc.col(i) = pts[i];
    }
    return pc;
}

void printResult(const char *name,
                 const bool success,
                 const Eigen::MatrixX4d &hPoly,
                 const std::vector<firi_common::MVIEDiagnostic> &diag)
{
    std::cout << name
              << " success=" << success
              << " rows=" << hPoly.rows()
              << " mvie_records=" << diag.size();
    if (!diag.empty())
    {
        std::cout << " status=[";
        for (std::size_t i = 0; i < diag.size(); ++i)
        {
            if (i != 0)
            {
                std::cout << ", ";
            }
            std::cout << diag[i].lbfgs_status;
        }
        std::cout << "]";
    }
    std::cout << std::endl;
}
}

int main()
{
    const Eigen::Matrix<double, 6, 4> bd =
        makeBoundary(Eigen::Vector3d(-5.0, -5.0, -5.0),
                     Eigen::Vector3d(5.0, 5.0, 5.0));
    const Eigen::Vector3d a(-1.0, 0.0, 0.0);
    const Eigen::Vector3d b(1.0, 0.0, 0.0);
    const Eigen::Matrix3Xd pc = makePointCloud({
        Eigen::Vector3d(0.0, 2.0, 0.0),
        Eigen::Vector3d(0.0, -2.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 2.0),
        Eigen::Vector3d(0.0, 0.0, -2.0),
        Eigen::Vector3d(0.0, 2.0, 2.0),
        Eigen::Vector3d(0.0, -2.0, -2.0)
    });

    Eigen::MatrixX4d h_firi;
    Eigen::MatrixX4d h_opt;
    Eigen::MatrixX4d h_socp;
    std::vector<firi_common::MVIEDiagnostic> d_firi;
    std::vector<firi_common::MVIEDiagnostic> d_opt;
    std::vector<firi_common::MVIEDiagnostic> d_socp;

    const bool ok_firi = firi::firi(bd, pc, a, b, h_firi, 4, 1.0e-6, &d_firi);
    const bool ok_opt = firi_opt::firi_opt(bd, pc, a, b, h_opt, 4, 1.0e-6, &d_opt);
    const bool ok_socp = firi_socp::firi_socp(bd, pc, a, b, h_socp, 4, 1.0e-6, &d_socp);

    printResult("firi", ok_firi, h_firi, d_firi);
    printResult("firi_opt", ok_opt, h_opt, d_opt);
    printResult("firi_socp", ok_socp, h_socp, d_socp);

    return (ok_firi && ok_opt && ok_socp) ? 0 : 1;
}
