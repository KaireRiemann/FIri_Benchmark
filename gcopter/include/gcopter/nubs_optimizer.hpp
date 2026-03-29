// #ifndef NUBS_OPTIMIZER_HPP
// #define NUBS_OPTIMIZER_HPP

// #include "gcopter/BSplineTrajectory.hpp" // 确保引用你最新包含 A*P=Q 和 De Boor 的文件
// #include "gcopter/lbfgs.hpp"
// #include <eigen3/Eigen/Eigen>
// #include <vector>
// #include <iostream>

// namespace nubs
// {

// class NUBSOptimizer
// {
// public:
//     typedef Eigen::MatrixX4d PolyhedronH;
//     typedef std::vector<PolyhedronH> PolyhedraH;

// private:
//     int s_order;
//     NUBSTrajectory<3> nubs;
    
//     int spatialDim; // 内部控制点的一维展开大小

//     Eigen::Matrix3d headPVA;
//     Eigen::Matrix3d tailPVA;
//     PolyhedraH hPolytopes;
//     Eigen::VectorXd T_fixed; // 固定的合理时间分配
    
//     double weightSFC;
//     double weightDyn;
//     double smoothEps;
//     double max_v, max_a;

//     lbfgs::lbfgs_parameter_t lbfgs_params;

//     static inline bool smoothedL1(const double &x, const double &mu, double &f, double &df) {
//         if (x < 0.0) return false;
//         else if (x > mu) { f = x - 0.5 * mu; df = 1.0; return true; }
//         else {
//             double xdmu = x / mu, sqrxdmu = xdmu * xdmu, mumxd2 = mu - 0.5 * x;
//             f = mumxd2 * sqrxdmu * xdmu;
//             df = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
//             return true;
//         }
//     }

//     // ★ 核心降维打击：只评估纯空间梯度，彻底抛弃有限差分！
//     inline void evaluateSpatialCostAndGrad(const Eigen::Ref<const Eigen::MatrixXd>& P_inner,
//                                            double& total_cost, 
//                                            Eigen::Ref<Eigen::MatrixXd> gradP_inner)
//     {
//         int M = T_fixed.size();
//         int N_c = nubs.getCtrlPtNum(M);
        
//         // 拼装完整控制点 (首尾被时间 T_fixed 严格约束，保证动力学边界)
//         Eigen::MatrixXd P_full = Eigen::MatrixXd::Zero(N_c, 3);
//         for (int i = 0; i < P_inner.rows(); i++) P_full.row(i + s_order) = P_inner.row(i);
//         nubs.resolveBoundaryCtrlPts(headPVA, tailPVA, T_fixed, P_full);

//         total_cost = 0.0;
//         Eigen::MatrixXd gradP_full = Eigen::MatrixXd::Zero(N_c, 3);

//         // 1. 能量与动力学计算 (复用 NUBSTrajectory 的高斯解析积分，极快)
//         double cost_dyn; Eigen::MatrixXd grad_dyn;
//         nubs.getEnergyAndGradP(T_fixed, P_full, max_v, max_a, weightDyn, cost_dyn, grad_dyn); 
//         total_cost += cost_dyn; 
//         gradP_full += grad_dyn;

//         // 2. 走廊软惩罚 (SFC) 绝对防撞护航
//         int polyN = hPolytopes.size();
//         for (int i = 0; i < N_c; i++) {
//             // 根据控制点索引，启发式分配到对应多面体
//             int poly_idx = std::min(std::max(0, (int)((double)i / N_c * polyN)), polyN - 1); 
//             const auto& polyH = hPolytopes[poly_idx];
            
//             for (int k = 0; k < polyH.rows(); k++) {
//                 Eigen::Vector3d n = polyH.block<1, 3>(k, 0).transpose();
//                 double d = polyH(k, 3);
//                 double dist = n.dot(P_full.row(i).transpose()) + d + smoothEps; 
                
//                 double f, df;
//                 if (smoothedL1(dist, 0.1, f, df)) {
//                     total_cost += weightSFC * f; 
//                     gradP_full.row(i) += weightSFC * df * n.transpose();
//                 }
//             }
//         }

//         // 3. 提取有效梯度，只回传给可优化变量
//         for (int i = 0; i < P_inner.rows(); i++) {
//             gradP_inner.row(i) = gradP_full.row(i + s_order);
//         }
//     }

//     static inline double costFunctional(void *ptr, const Eigen::VectorXd &x, Eigen::VectorXd &g)
//     {
//         NUBSOptimizer &obj = *(NUBSOptimizer *)ptr;
//         int dimP = obj.spatialDim; 

//         // 零拷贝映射内存，速度拉满
//         Eigen::Map<const Eigen::MatrixXd> P_inner(x.data(), dimP / 3, 3);
//         Eigen::Map<Eigen::MatrixXd> gradP_inner(g.data(), dimP / 3, 3);

//         double total_cost = 0.0;
//         obj.evaluateSpatialCostAndGrad(P_inner, total_cost, gradP_inner);

//         return total_cost;
//     }

// public:
//     NUBSOptimizer(int order = 3) : s_order(order), nubs(order) 
//     {
//         weightSFC = 3000.0;
//         weightDyn = 1000.0;
//         smoothEps = 0.1; // 走廊的安全余量 10cm

//         // 放宽 L-BFGS 条件，对于这种纯控制点的软惩罚，容易收敛
//         lbfgs_params.mem_size = 256;
//         lbfgs_params.past = 3;
//         lbfgs_params.delta = 1e-3;
//         lbfgs_params.g_epsilon = 1e-4;
//         lbfgs_params.min_step = 1e-32;
//     }

//     bool setup(const Eigen::Matrix3d &initialPVA, const Eigen::Matrix3d &terminalPVA,
//                const PolyhedraH &safeCorridor, double vmax, double amax)
//     {
//         headPVA = initialPVA;
//         tailPVA = terminalPVA;
//         hPolytopes = safeCorridor;
//         max_v = vmax; max_a = amax;

//         for (size_t i = 0; i < hPolytopes.size(); i++) {
//             const Eigen::ArrayXd norms = hPolytopes[i].leftCols<3>().rowwise().norm();
//             hPolytopes[i].array().colwise() /= norms;
//         }
//         return true;
//     }

//     bool optimize(const Eigen::VectorXd &times_init, const Eigen::MatrixXd &waypoints_init, 
//                   NUBSTrajectory<3> &out_traj)
//     {
//         T_fixed = times_init; // 冻结时间
        
//         // 1. 通过求解 A*P=Q 一次性解出极高质量的插值初始控制点
//         Eigen::MatrixXd P_full;
//         nubs.generateInitialControlPoints(waypoints_init, headPVA, tailPVA, T_fixed, P_full);
        
//         // 2. 剥离固定的边界控制点，仅将内部变量喂给 L-BFGS
//         spatialDim = (P_full.rows() - 2 * s_order) * 3; 
        
//         Eigen::VectorXd x(spatialDim);
//         Eigen::Map<Eigen::MatrixXd> P_inner(x.data(), spatialDim / 3, 3);
//         for(int i = 0; i < spatialDim / 3; i++) P_inner.row(i) = P_full.row(i + s_order);

//         // 3. 毫秒级极速优化
//         double minCost;
//         int ret = lbfgs::lbfgs_optimize(x, minCost, &NUBSOptimizer::costFunctional, nullptr, nullptr, this, lbfgs_params);

//         if (ret >= 0 || ret == lbfgs::LBFGSERR_MAXIMUMITERATION) {
//             // 将优化后的内部点回填
//             for(int i = 0; i < P_inner.rows(); i++) P_full.row(i + s_order) = P_inner.row(i);
//             nubs.resolveBoundaryCtrlPts(headPVA, tailPVA, T_fixed, P_full); 
            
//             out_traj.setTrajectory(T_fixed, P_full);
//             return true;
//         }
        
//         std::cerr << "Optimization Failed: " << lbfgs::lbfgs_strerror(ret) << std::endl;
//         return false;
//     }
// };

// } // namespace nubs
// #endif

// #ifndef NUBS_OPTIMIZER_HPP
// #define NUBS_OPTIMIZER_HPP

// #include "gcopter/BSplineTrajectory.hpp" 
// #include "gcopter/lbfgs.hpp"
// #include <eigen3/Eigen/Eigen>
// #include <vector>
// #include <iostream>

// namespace nubs
// {

// class NUBSOptimizer
// {
// public:
//     typedef Eigen::MatrixX4d PolyhedronH;
//     typedef std::vector<PolyhedronH> PolyhedraH;

// private:
//     int s_order;
//     NUBSTrajectory<3> nubs;
    
//     int spatialDim; 
//     int integralRes; // 采样精度，决定动力学检查的粒度

//     Eigen::Matrix3d headPVA;
//     Eigen::Matrix3d tailPVA;
//     PolyhedraH hPolytopes;
//     Eigen::VectorXd T_fixed; 
    
//     double weightSFC;
//     double weightDyn;
//     double smoothEps;
//     double max_v, max_a;

//     lbfgs::lbfgs_parameter_t lbfgs_params;

//     static inline bool smoothedL1(const double &x, const double &mu, double &f, double &df) {
//         if (x < 0.0) return false;
//         else if (x > mu) { f = x - 0.5 * mu; df = 1.0; return true; }
//         else {
//             double xdmu = x / mu, sqrxdmu = xdmu * xdmu, mumxd2 = mu - 0.5 * x;
//             f = mumxd2 * sqrxdmu * xdmu;
//             df = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
//             return true;
//         }
//     }

//     // ==============================================================================
//     // ★ 统一梯度引擎：采用梯形积分采样状态，利用链式法则反向传播回参数空间
//     // ==============================================================================
//     inline void attachKinematicPenalty(const Eigen::VectorXd& T, 
//                                        const Eigen::MatrixXd& P_full, 
//                                        double& cost, Eigen::MatrixXd& gradP) 
//     {
//         int N_c = P_full.rows();
//         Eigen::VectorXd knots = nubs.generateKnots(T, N_c);
//         int p = nubs.getPDeg();
        
//         double max_v2 = max_v * max_v;
//         double max_a2 = max_a * max_a;
        
//         for (int i = p; i < knots.size() - p - 1; i++) {
//             double t_start = knots(i), t_end = knots(i + 1);
//             if (t_end - t_start < 1e-9) continue;
            
//             double step = (t_end - t_start) / integralRes;
            
//             for (int j = 0; j <= integralRes; j++) {
//                 double t = t_start + j * step;
//                 if (t >= knots(knots.size() - p - 1)) t -= 1e-9;
                
//                 int span = nubs.findSpan(t, N_c, knots);
//                 Eigen::MatrixXd ders;
//                 nubs.dersBasisFuns(2, span, t, knots, ders); // 最多算到加速度(2阶)
                
//                 Eigen::Vector3d vel = Eigen::Vector3d::Zero();
//                 Eigen::Vector3d acc = Eigen::Vector3d::Zero();
                
//                 for (int k = 0; k <= p; k++) {
//                     vel += ders(1, k) * P_full.row(span - p + k).transpose();
//                     acc += ders(2, k) * P_full.row(span - p + k).transpose();
//                 }
                
//                 double violaVel = vel.squaredNorm() - max_v2;
//                 double violaAcc = acc.squaredNorm() - max_a2;
                
//                 Eigen::Vector3d grad_vel = Eigen::Vector3d::Zero();
//                 Eigen::Vector3d grad_acc = Eigen::Vector3d::Zero();
                
//                 double node_w = (j == 0 || j == integralRes) ? 0.5 : 1.0;
//                 double w = node_w * step;
                
//                 double f, df;
//                 if (smoothedL1(violaVel, smoothEps, f, df)) {
//                     cost += w * weightDyn * f;
//                     grad_vel += w * weightDyn * df * 2.0 * vel;
//                 }
//                 if (smoothedL1(violaAcc, smoothEps, f, df)) {
//                     cost += w * weightDyn * f;
//                     grad_acc += w * weightDyn * df * 2.0 * acc;
//                 }
                
//                 // ★ 统一的表达形式：将物理状态梯度精准映射回相关的 p+1 个控制点
//                 for (int k = 0; k <= p; k++) {
//                     gradP.row(span - p + k) += ders(1, k) * grad_vel.transpose() +
//                                                ders(2, k) * grad_acc.transpose();
//                 }
//             }
//         }
//     }

//     inline void evaluateSpatialCostAndGrad(const Eigen::Ref<const Eigen::MatrixXd>& P_inner,
//                                            double& total_cost, 
//                                            Eigen::Ref<Eigen::MatrixXd> gradP_inner)
//     {
//         int M = T_fixed.size();
//         int N_c = nubs.getCtrlPtNum(M);
        
//         Eigen::MatrixXd P_full = Eigen::MatrixXd::Zero(N_c, 3);
//         for (int i = 0; i < P_inner.rows(); i++) P_full.row(i + s_order) = P_inner.row(i);
//         nubs.resolveBoundaryCtrlPts(headPVA, tailPVA, T_fixed, P_full);

//         total_cost = 0.0;
//         Eigen::MatrixXd gradP_full = Eigen::MatrixXd::Zero(N_c, 3);

//         // 1. 能量与解析梯度 (独立模块，仅关注曲线光滑度)
//         double cost_energy; Eigen::MatrixXd grad_energy;
//         nubs.getEnergyAndGradP(T_fixed, P_full, cost_energy, grad_energy); 
//         total_cost += cost_energy; 
//         gradP_full += grad_energy;

//         // 2. 动力学统一惩罚 (剥离到优化器中)
//         double cost_dyn = 0.0; Eigen::MatrixXd grad_dyn = Eigen::MatrixXd::Zero(N_c, 3);
//         attachKinematicPenalty(T_fixed, P_full, cost_dyn, grad_dyn);
//         total_cost += cost_dyn;
//         gradP_full += grad_dyn;

//         // 3. 走廊软惩罚 (SFC)，确保 100% 不撞墙
//         int polyN = hPolytopes.size();
//         for (int i = 0; i < N_c; i++) {
//             int poly_idx = std::min(std::max(0, (int)((double)i / N_c * polyN)), polyN - 1); 
//             const auto& polyH = hPolytopes[poly_idx];
            
//             for (int k = 0; k < polyH.rows(); k++) {
//                 Eigen::Vector3d n = polyH.block<1, 3>(k, 0).transpose();
//                 double d = polyH(k, 3);
//                 double dist = n.dot(P_full.row(i).transpose()) + d + smoothEps; 
                
//                 double f, df;
//                 if (smoothedL1(dist, 0.1, f, df)) {
//                     total_cost += weightSFC * f; 
//                     gradP_full.row(i) += weightSFC * df * n.transpose();
//                 }
//             }
//         }

//         // 4. 将剥除首尾边界的梯度回传给优化变量
//         for (int i = 0; i < P_inner.rows(); i++) {
//             gradP_inner.row(i) = gradP_full.row(i + s_order);
//         }
//     }

//     static inline double costFunctional(void *ptr, const Eigen::VectorXd &x, Eigen::VectorXd &g)
//     {
//         NUBSOptimizer &obj = *(NUBSOptimizer *)ptr;
//         int dimP = obj.spatialDim; 

//         Eigen::Map<const Eigen::MatrixXd> P_inner(x.data(), dimP / 3, 3);
//         Eigen::Map<Eigen::MatrixXd> gradP_inner(g.data(), dimP / 3, 3);

//         double total_cost = 0.0;
//         obj.evaluateSpatialCostAndGrad(P_inner, total_cost, gradP_inner);

//         return total_cost;
//     }

// public:
//     NUBSOptimizer(int order = 3) : s_order(order), nubs(order) 
//     {
//         weightSFC = 3000.0;
//         weightDyn = 1000.0;
//         smoothEps = 0.1;
//         integralRes = 16; // 每段航路采样 16 次，对防撞和物理约束已绰绰有余

//         lbfgs_params.mem_size = 256;
//         lbfgs_params.past = 3;
//         lbfgs_params.delta = 1e-3;
//         lbfgs_params.g_epsilon = 1e-4;
//         lbfgs_params.min_step = 1e-32;
//     }

//     bool setup(const Eigen::Matrix3d &initialPVA, const Eigen::Matrix3d &terminalPVA,
//                const PolyhedraH &safeCorridor, double vmax, double amax)
//     {
//         headPVA = initialPVA;
//         tailPVA = terminalPVA;
//         hPolytopes = safeCorridor;
//         max_v = vmax; max_a = amax;

//         for (size_t i = 0; i < hPolytopes.size(); i++) {
//             const Eigen::ArrayXd norms = hPolytopes[i].leftCols<3>().rowwise().norm();
//             hPolytopes[i].array().colwise() /= norms;
//         }
//         return true;
//     }

//     bool optimize(const Eigen::VectorXd &times_init, const Eigen::MatrixXd &waypoints_init, 
//                   NUBSTrajectory<3> &out_traj)
//     {
//         T_fixed = times_init; 
        
//         Eigen::MatrixXd P_full;
//         nubs.generateInitialControlPoints(waypoints_init, headPVA, tailPVA, T_fixed, P_full);
        
//         spatialDim = (P_full.rows() - 2 * s_order) * 3; 
//         Eigen::VectorXd x(spatialDim);
//         Eigen::Map<Eigen::MatrixXd> P_inner(x.data(), spatialDim / 3, 3);
//         for(int i = 0; i < spatialDim / 3; i++) P_inner.row(i) = P_full.row(i + s_order);

//         double minCost;
//         int ret = lbfgs::lbfgs_optimize(x, minCost, &NUBSOptimizer::costFunctional, nullptr, nullptr, this, lbfgs_params);

//         if (ret >= 0 || ret == lbfgs::LBFGSERR_MAXIMUMITERATION) {
//             for(int i = 0; i < P_inner.rows(); i++) P_full.row(i + s_order) = P_inner.row(i);
//             nubs.resolveBoundaryCtrlPts(headPVA, tailPVA, T_fixed, P_full); 
            
//             out_traj.setTrajectory(T_fixed, P_full);
//             return true;
//         }
        
//         std::cerr << "Optimization Failed: " << lbfgs::lbfgs_strerror(ret) << std::endl;
//         return false;
//     }
// };

// } // namespace nubs
// #endif

#ifndef NUBS_OPTIMIZER_HPP
#define NUBS_OPTIMIZER_HPP

#include "gcopter/BSplineTrajectory.hpp"
#include "gcopter/lbfgs.hpp"
#include <eigen3/Eigen/Eigen>
#include <vector>
#include <iostream>

namespace nubs
{

class NUBSOptimizer
{
public:
    typedef Eigen::Matrix3Xd PolyhedronV;
    typedef Eigen::MatrixX4d PolyhedronH;
    typedef std::vector<PolyhedronV> PolyhedraV;
    typedef std::vector<PolyhedronH> PolyhedraH;

private:
    int s_order;
    NUBSTrajectory<3> nubs;
    
    int temporalDim;
    int spatialDim;
    int integralRes;

    Eigen::Matrix3d headPVA;
    Eigen::Matrix3d tailPVA;
    
    PolyhedraV vPolytopes;
    PolyhedraH hPolytopes;
    Eigen::VectorXi vPolyIdx;
    Eigen::VectorXi hPolyIdx;

    double weightT;
    double weightDyn;
    double weightSFC;
    double weightEnergy;
    double smoothEps;
    double max_v, max_a;

    lbfgs::lbfgs_parameter_t lbfgs_params;

    static inline void forwardT(const Eigen::Ref<const Eigen::VectorXd> &tau, std::vector<double> &T_nodes) {
        int sz = tau.size(); T_nodes.resize(sz + 1, 0.0);
        for (int i = 0; i < sz; i++) {
            double T_i = tau(i) > 0.0 ? ((0.5 * tau(i) + 1.0) * tau(i) + 1.0) : 1.0 / ((0.5 * tau(i) - 1.0) * tau(i) + 1.0);
            T_nodes[i+1] = T_nodes[i] + T_i;
        }
    }
    
    static inline void backwardT(const Eigen::VectorXd &T, Eigen::Ref<Eigen::VectorXd> tau) {
        int sz = T.size(); 
        for (int i = 0; i < sz; i++) 
            tau(i) = T(i) > 1.0 ? (sqrt(2.0 * T(i) - 1.0) - 1.0) : (1.0 - sqrt(2.0 / T(i) - 1.0));
    }

    static inline void backwardGradT(const Eigen::Ref<const Eigen::VectorXd> &tau, const Eigen::VectorXd &gradT, Eigen::Ref<Eigen::VectorXd> gradTau) {
        int sz = tau.size(); 
        for (int i = 0; i < sz; i++) {
            if (tau(i) > 0) gradTau(i) = gradT(i) * (tau(i) + 1.0);
            else {
                double denSqrt = (0.5 * tau(i) - 1.0) * tau(i) + 1.0;
                gradTau(i) = gradT(i) * (1.0 - tau(i)) / (denSqrt * denSqrt);
            }
        }
    }

    static inline void forwardP(const Eigen::Ref<const Eigen::VectorXd> &xi, const Eigen::VectorXi &vIdx,
                                const PolyhedraV &vPolys, Eigen::MatrixXd &Q_inner) {
        int M_minus_1 = vIdx.size();
        Q_inner.resize(M_minus_1, 3);
        Eigen::VectorXd q;
        for (int i = 0, j = 0, k, l; i < M_minus_1; i++, j += k) {
            l = vIdx(i); k = vPolys[l].cols();
            q = xi.segment(j, k).normalized().head(k - 1);
            Q_inner.row(i) = (vPolys[l].rightCols(k - 1) * q.cwiseProduct(q) + vPolys[l].col(0)).transpose();
        }
    }

    static inline double costTinyNLS(void *ptr, const Eigen::VectorXd &xi, Eigen::VectorXd &gradXi) {
        const int n = xi.size();
        const Eigen::Matrix3Xd &ovPoly = *(Eigen::Matrix3Xd *)ptr;
        const double sqrNormXi = xi.squaredNorm();
        const double invNormXi = 1.0 / sqrt(sqrNormXi);
        const Eigen::VectorXd unitXi = xi * invNormXi;
        const Eigen::VectorXd r = unitXi.head(n - 1);
        const Eigen::Vector3d delta = ovPoly.rightCols(n - 1) * r.cwiseProduct(r) + ovPoly.col(1) - ovPoly.col(0);

        double cost = delta.squaredNorm();
        gradXi.head(n - 1) = (ovPoly.rightCols(n - 1).transpose() * (2 * delta)).array() * r.array() * 2.0;
        gradXi(n - 1) = 0.0;
        gradXi = (gradXi - unitXi.dot(gradXi) * unitXi).eval() * invNormXi;

        const double sqrNormViolation = sqrNormXi - 1.0;
        if (sqrNormViolation > 0.0) {
            double c = sqrNormViolation * sqrNormViolation;
            cost += c * sqrNormViolation;
            gradXi += 3.0 * c * 2.0 * xi;
        }
        return cost;
    }

    static inline void backwardP(const Eigen::MatrixXd &Q_inner, const Eigen::VectorXi &vIdx,
                                 const PolyhedraV &vPolys, Eigen::Ref<Eigen::VectorXd> xi) {
        int M_minus_1 = Q_inner.rows();
        double minSqrD;
        lbfgs::lbfgs_parameter_t tiny_params;
        tiny_params.past = 0; tiny_params.delta = 1e-5; tiny_params.g_epsilon = FLT_EPSILON; tiny_params.max_iterations = 128;
        Eigen::Matrix3Xd ovPoly;
        for (int i = 0, j = 0, k, l; i < M_minus_1; i++, j += k) {
            l = vIdx(i); k = vPolys[l].cols();
            ovPoly.resize(3, k + 1);
            ovPoly.col(0) = Q_inner.row(i).transpose();
            ovPoly.rightCols(k) = vPolys[l];
            Eigen::VectorXd x(k); x.setConstant(sqrt(1.0 / k));
            lbfgs::lbfgs_optimize(x, minSqrD, &NUBSOptimizer::costTinyNLS, nullptr, nullptr, &ovPoly, tiny_params);
            xi.segment(j, k) = x;
        }
    }

    static inline void backwardGradP(const Eigen::Ref<const Eigen::VectorXd> &xi, const Eigen::VectorXi &vIdx,
                                     const PolyhedraV &vPolys, const Eigen::MatrixXd &gradQ, Eigen::Ref<Eigen::VectorXd> gradXi) {
        int M_minus_1 = vIdx.size();
        double normInv;
        Eigen::VectorXd q, grad_q, unitQ;
        for (int i = 0, j = 0, k, l; i < M_minus_1; i++, j += k) {
            l = vIdx(i); k = vPolys[l].cols();
            q = xi.segment(j, k);
            normInv = 1.0 / q.norm();
            unitQ = q * normInv;
            grad_q.resize(k);
            grad_q.head(k - 1) = (vPolys[l].rightCols(k - 1).transpose() * gradQ.row(i).transpose()).array() * unitQ.head(k - 1).array() * 2.0;
            grad_q(k - 1) = 0.0;
            gradXi.segment(j, k) = (grad_q - unitQ * unitQ.dot(grad_q)) * normInv;
        }
    }

    static inline void normRetrictionLayer(const Eigen::Ref<const Eigen::VectorXd> &xi, const Eigen::VectorXi &vIdx,
                                           const PolyhedraV &vPolys, double &cost, Eigen::Ref<Eigen::VectorXd> gradXi) {
        int M_minus_1 = vIdx.size();
        double sqrNormQ, sqrNormViolation, c, dc;
        Eigen::VectorXd q;
        for (int i = 0, j = 0, k; i < M_minus_1; i++, j += k) {
            k = vPolys[vIdx(i)].cols();
            q = xi.segment(j, k);
            sqrNormQ = q.squaredNorm();
            sqrNormViolation = sqrNormQ - 1.0;
            if (sqrNormViolation > 0.0) {
                c = sqrNormViolation * sqrNormViolation; dc = 3.0 * c; c *= sqrNormViolation;
                cost += c; gradXi.segment(j, k) += dc * 2.0 * q;
            }
        }
    }

    static inline bool smoothedL1(const double &x, const double &mu, double &f, double &df) {
        if (x < 0.0) return false;
        else if (x > mu) { f = x - 0.5 * mu; df = 1.0; return true; }
        else {
            double xdmu = x / mu, sqrxdmu = xdmu * xdmu, mumxd2 = mu - 0.5 * x;
            f = mumxd2 * sqrxdmu * xdmu;
            df = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
            return true;
        }
    }

    inline void evaluateCostAndGrad(const std::vector<double>& T_nodes, const Eigen::MatrixXd& Q_inner,
                                    double& cost, Eigen::MatrixXd& gradQ_inner, bool need_grad = true)
    {
        int M = T_nodes.size() - 1;
        Eigen::MatrixXd Q_full(M + 1, 3);
        Q_full.row(0) = headPVA.col(0).transpose();
        for (int i = 0; i < M - 1; i++) Q_full.row(i + 1) = Q_inner.row(i);
        Q_full.row(M) = tailPVA.col(0).transpose();

        Eigen::MatrixXd start_d = headPVA.rightCols(s_order - 1).transpose().eval();
        Eigen::MatrixXd end_d   = tailPVA.rightCols(s_order - 1).transpose().eval();
        

        nubs.generate(T_nodes, Q_full, start_d, end_d);
        
        int N_c = nubs.getCtrlPtNum(M);
        Eigen::MatrixXd P_full = nubs.getControlPoints();
        Eigen::VectorXd knots = nubs.getKnots();
        int p = nubs.getPDeg();

        cost = 0.0;
        Eigen::MatrixXd gradP_full = Eigen::MatrixXd::Zero(N_c, 3);
        
        double max_v2 = max_v * max_v, max_a2 = max_a * max_a;
        Eigen::Matrix<double, 8, 8> ders; 

        for (int i = p; i < knots.size() - p - 1; i++) {
            double t_start = knots(i), t_end = knots(i + 1);
            if (t_end - t_start < 1e-9) continue;
            
            int seg_idx = std::min(std::max(0, i - p), M - 1);
            int poly_idx = hPolyIdx(seg_idx);
            const auto& polyH = hPolytopes[poly_idx];
            
            double step = (t_end - t_start) / integralRes;
            for (int j = 0; j <= integralRes; j++) {
                double t = t_start + j * step;
                if (t >= knots(knots.size() - p - 1)) t -= 1e-9;
                
                int span = nubs.findSpan(t, N_c, knots);
                nubs.dersBasisFuns(3, span, t, knots, ders);
                
                Eigen::Vector3d pos = Eigen::Vector3d::Zero();
                Eigen::Vector3d vel = Eigen::Vector3d::Zero();
                Eigen::Vector3d acc = Eigen::Vector3d::Zero();
                Eigen::Vector3d jer = Eigen::Vector3d::Zero();
                
                for (int k = 0; k <= p; k++) {
                    pos += ders(0, k) * P_full.row(span - p + k).transpose();
                    vel += ders(1, k) * P_full.row(span - p + k).transpose();
                    acc += ders(2, k) * P_full.row(span - p + k).transpose();
                    jer += ders(3, k) * P_full.row(span - p + k).transpose();
                }

                double node_w = (j == 0 || j == integralRes) ? 0.5 : 1.0;
                double w = node_w * step;

                Eigen::Vector3d grad_pos = Eigen::Vector3d::Zero();
                Eigen::Vector3d grad_vel = Eigen::Vector3d::Zero();
                Eigen::Vector3d grad_acc = Eigen::Vector3d::Zero();
                Eigen::Vector3d grad_jer = Eigen::Vector3d::Zero();

                cost += w * weightEnergy * jer.squaredNorm();
                if(need_grad) grad_jer += w * weightEnergy * 2.0 * jer;

                for (int m = 0; m < polyH.rows(); m++) {
                    Eigen::Vector3d n = polyH.block<1, 3>(m, 0).transpose();
                    double dist = n.dot(pos) + polyH(m, 3) + smoothEps; 
                    double f, df;
                    if (smoothedL1(dist, 0.1, f, df)) {
                        cost += w * weightSFC * f;
                        if(need_grad) grad_pos += w * weightSFC * df * n;
                    }
                }

                double f, df;
                if (smoothedL1(vel.squaredNorm() - max_v2, smoothEps, f, df)) {
                    cost += w * weightDyn * f;
                    if(need_grad) grad_vel += w * weightDyn * df * 2.0 * vel;
                }
                if (smoothedL1(acc.squaredNorm() - max_a2, smoothEps, f, df)) {
                    cost += w * weightDyn * f;
                    if(need_grad) grad_acc += w * weightDyn * df * 2.0 * acc;
                }

                if(need_grad) {
                    for (int k = 0; k <= p; k++) {
                        gradP_full.row(span - p + k) += ders(0, k) * grad_pos.transpose() +
                                                        ders(1, k) * grad_vel.transpose() +
                                                        ders(2, k) * grad_acc.transpose() +
                                                        ders(3, k) * grad_jer.transpose();
                    }
                }
            }
        }

        // ★ 显式传入 M 参数！
        if(need_grad) nubs.propogateGrad(gradP_full, gradQ_inner, M);
    }

    static inline double costFunctional(void *ptr, const Eigen::VectorXd &x, Eigen::VectorXd &g)
    {
        NUBSOptimizer &obj = *(NUBSOptimizer *)ptr;
        int dimTau = obj.temporalDim;    
        int dimP = obj.spatialDim; 

        Eigen::Map<const Eigen::VectorXd> tau(x.data(), dimTau);
        Eigen::Map<const Eigen::VectorXd> xi(x.data() + dimTau, dimP);
        Eigen::Map<Eigen::VectorXd> gradTau(g.data(), dimTau);
        Eigen::Map<Eigen::VectorXd> gradXi(g.data() + dimTau, dimP);

        std::vector<double> T_nodes; 
        forwardT(tau, T_nodes); 
        
        Eigen::MatrixXd Q_inner;
        forwardP(xi, obj.vPolyIdx, obj.vPolytopes, Q_inner);

        double total_cost = 0.0;
        Eigen::MatrixXd gradQ_inner;
        obj.evaluateCostAndGrad(T_nodes, Q_inner, total_cost, gradQ_inner, true);

        double eps = 1e-4;
        Eigen::VectorXd gradT = Eigen::VectorXd::Zero(dimTau);
        Eigen::MatrixXd dummy_g; 
        
        for (int i = 0; i < dimTau; i++) {
            std::vector<double> T_p = T_nodes, T_m = T_nodes;
            for(int j = i + 1; j <= dimTau; j++) { T_p[j] += eps; T_m[j] -= eps; }
            
            double c_p = 0.0, c_m = 0.0;
            obj.evaluateCostAndGrad(T_p, Q_inner, c_p, dummy_g, false);
            obj.evaluateCostAndGrad(T_m, Q_inner, c_m, dummy_g, false);
            gradT(i) = (c_p - c_m) / (2.0 * eps);
        }

        Eigen::VectorXd T(dimTau); 
        for(int i=0; i<dimTau; i++) T(i) = T_nodes[i+1] - T_nodes[i];
        total_cost += obj.weightT * T.sum();
        gradT.array() += obj.weightT;

        backwardGradT(tau, gradT, gradTau);
        backwardGradP(xi, obj.vPolyIdx, obj.vPolytopes, gradQ_inner, gradXi);
        normRetrictionLayer(xi, obj.vPolyIdx, obj.vPolytopes, total_cost, gradXi);

        return total_cost;
    }

public:
    NUBSOptimizer(int order = 3) : s_order(order), nubs(order) 
    {
        weightT = 10.0;
        weightEnergy = 0.1;
        weightSFC = 1000.0;
        weightDyn = 500.0;
        smoothEps = 0.1; 
        integralRes = 8; 

        lbfgs_params.mem_size = 256;
        lbfgs_params.past = 3;
        lbfgs_params.delta = 1e-4;
        lbfgs_params.g_epsilon = 1e-4;
        lbfgs_params.min_step = 1e-32;
    }

    bool setup(const Eigen::Matrix3d &initialPVA, const Eigen::Matrix3d &terminalPVA,
               const PolyhedraV &vPolys, const PolyhedraH &hPolys, 
               const Eigen::VectorXi &vIdx, const Eigen::VectorXi &hIdx,
               double vmax, double amax)
    {
        headPVA = initialPVA;
        tailPVA = terminalPVA;
        vPolytopes = vPolys;
        hPolytopes = hPolys;
        vPolyIdx = vIdx;
        hPolyIdx = hIdx;
        max_v = vmax; max_a = amax;

        for (size_t i = 0; i < hPolytopes.size(); i++) {
            const Eigen::ArrayXd norms = hPolytopes[i].leftCols<3>().rowwise().norm();
            hPolytopes[i].array().colwise() /= norms;
        }
        
        temporalDim = hIdx.size();
        spatialDim = 0;
        for (int i = 0; i < vPolyIdx.size(); i++) spatialDim += vPolytopes[vPolyIdx(i)].cols();
        return true;
    }

    bool optimize(const Eigen::VectorXd &times_init, const Eigen::MatrixXd &waypoints_init, 
                  NUBSTrajectory<3> &out_traj)
    {
        Eigen::VectorXd x(temporalDim + spatialDim);
        Eigen::Map<Eigen::VectorXd> tau(x.data(), temporalDim);
        Eigen::Map<Eigen::VectorXd> xi(x.data() + temporalDim, spatialDim);

        backwardT(times_init, tau);
        backwardP(waypoints_init, vPolyIdx, vPolytopes, xi);

        double minCost;
        int ret = lbfgs::lbfgs_optimize(x, minCost, &NUBSOptimizer::costFunctional, nullptr, nullptr, this, lbfgs_params);

        if (ret >= 0 || ret == lbfgs::LBFGSERR_MAXIMUMITERATION) {
            std::vector<double> T_opt; forwardT(tau, T_opt);
            Eigen::MatrixXd Q_opt; forwardP(xi, vPolyIdx, vPolytopes, Q_opt);
            
            Eigen::MatrixXd Q_full(Q_opt.rows() + 2, 3);
            Q_full.row(0) = headPVA.col(0).transpose();
            for (int i = 0; i < Q_opt.rows(); i++) Q_full.row(i + 1) = Q_opt.row(i);
            Q_full.row(Q_full.rows() - 1) = tailPVA.col(0).transpose();
            
            Eigen::MatrixXd start_d = headPVA.rightCols(s_order - 1).transpose().eval();
            Eigen::MatrixXd end_d   = tailPVA.rightCols(s_order - 1).transpose().eval();

            nubs.generate(T_opt, Q_full, start_d, end_d);
            out_traj = nubs;
            return true;
        }
        
        std::cerr << "Optimization Failed: " << lbfgs::lbfgs_strerror(ret) << std::endl;
        return false;
    }
};

} 
#endif