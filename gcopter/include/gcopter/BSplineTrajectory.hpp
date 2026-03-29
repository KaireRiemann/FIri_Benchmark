#ifndef BSPLINE_TRAJECTORY_HPP
#define BSPLINE_TRAJECTORY_HPP

#include <eigen3/Eigen/Eigen>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace nubs
{

class BandedSystem 
{
public:
    inline void create(const int &n, const int &p, const int &q) {
        destroy();
        N = n; lowerBw = p; upperBw = q;
        int actualSize = N * (lowerBw + upperBw + 1);
        ptrData = new double[actualSize];
        std::fill_n(ptrData, actualSize, 0.0);
    }
    inline void destroy() {
        if (ptrData != nullptr) { delete[] ptrData; ptrData = nullptr; }
    }
private:
    int N, lowerBw, upperBw;
    double *ptrData = nullptr;
public:
    inline void reset(void) { std::fill_n(ptrData, N * (lowerBw + upperBw + 1), 0.0); }
    inline const double &operator()(const int &i, const int &j) const { return ptrData[(i - j + upperBw) * N + j]; }
    inline double &operator()(const int &i, const int &j) { return ptrData[(i - j + upperBw) * N + j]; }
    
    inline void factorizeLU() {
        int iM, jM; double cVl;
        for (int k = 0; k <= N - 2; k++) {
            iM = std::min(k + lowerBw, N - 1);
            cVl = operator()(k, k);
            for (int i = k + 1; i <= iM; i++) { if (operator()(i, k) != 0.0) operator()(i, k) /= cVl; }
            jM = std::min(k + upperBw, N - 1);
            for (int j = k + 1; j <= jM; j++) {
                cVl = operator()(k, j);
                if (cVl != 0.0) {
                    for (int i = k + 1; i <= iM; i++) {
                        if (operator()(i, k) != 0.0) operator()(i, j) -= operator()(i, k) * cVl;
                    }
                }
            }
        }
    }

    template <typename EIGENMAT>
    inline void solve(EIGENMAT &b) const {
        int iM;
        for (int j = 0; j <= N - 1; j++) {
            iM = std::min(j + lowerBw, N - 1);
            for (int i = j + 1; i <= iM; i++) {
                if (operator()(i, j) != 0.0) b.row(i) -= operator()(i, j) * b.row(j);
            }
        }
        for (int j = N - 1; j >= 0; j--) {
            b.row(j) /= operator()(j, j);
            iM = std::max(0, j - upperBw);
            for (int i = iM; i <= j - 1; i++) {
                if (operator()(i, j) != 0.0) b.row(i) -= operator()(i, j) * b.row(j);
            }
        }
    }
};

template<int Dim>
class NUBSTrajectory 
{
private:
    int s;         
    int p;         
    
    Eigen::VectorXd knots;
    Eigen::Matrix<double, Eigen::Dynamic, Dim> control_points;

    std::vector<std::vector<double>> gauss_nodes = {
        {}, {0.0}, 
        {-0.577350269, 0.577350269}, 
        {-0.774596669, 0.0, 0.774596669}, 
        {-0.861136311, -0.339981043, 0.339981043, 0.861136311},
        {-0.906179845, -0.538469310, 0.0, 0.538469310, 0.906179845},
        {-0.932469514, -0.661209386, -0.238619186, 0.238619186, 0.661209386, 0.932469514}
    };
    std::vector<std::vector<double>> gauss_weights = {
        {}, {2.0}, 
        {1.0, 1.0}, 
        {0.555555555, 0.888888888, 0.555555555}, 
        {0.347854845, 0.652145154, 0.652145154, 0.347854845},
        {0.236926885, 0.478628670, 0.568888888, 0.478628670, 0.236926885},
        {0.171324492, 0.360761573, 0.467913934, 0.467913934, 0.360761573, 0.171324492}
    };

    inline int findSpan(double t, int num_ctrl_pts, const Eigen::VectorXd& u) const {
        if (t >= u(num_ctrl_pts)) return num_ctrl_pts - 1;
        if (t <= u(p)) return p;
        int low = p, high = num_ctrl_pts, mid;
        while (low < high) {
            mid = (low + high) / 2;
            if (t < u(mid)) high = mid;
            else low = mid + 1;
        }
        return low - 1;
    }

    inline void dersBasisFuns(int n, int span, double t, const Eigen::VectorXd& u, Eigen::MatrixXd& ders) const {
        n = std::min(n, p); 
        ders.setZero(n + 1, p + 1);
        Eigen::MatrixXd ndu(p + 1, p + 1);
        ndu(0, 0) = 1.0;
        Eigen::VectorXd left(p + 1), right(p + 1);

        for (int j = 1; j <= p; j++) {
            left(j) = t - u(span + 1 - j);
            right(j) = u(span + j) - t;
            double saved = 0.0;
            for (int r = 0; r < j; r++) {
                ndu(j, r) = right(r + 1) + left(j - r);
                double temp = (ndu(j, r) == 0.0) ? 0.0 : ndu(r, j - 1) / ndu(j, r);
                ndu(r, j) = saved + right(r + 1) * temp;
                saved = left(j - r) * temp;
            }
            ndu(j, j) = saved;
        }
        for (int j = 0; j <= p; j++) ders(0, j) = ndu(j, p);
        if (n == 0) return;

        Eigen::MatrixXd a(2, p + 1);
        for (int r = 0; r <= p; r++) {
            int s1 = 0, s2 = 1;
            a(0, 0) = 1.0;
            for (int k = 1; k <= n; k++) {
                double d = 0.0;
                int rk = r - k, pk = p - k;
                if (r >= k) {
                    double den = ndu(pk + 1, rk);
                    a(s2, 0) = (den == 0.0) ? 0.0 : a(s1, 0) / den;
                    d = a(s2, 0) * ndu(rk, pk);
                }
                int j1 = (rk >= -1) ? 1 : -rk;
                int j2 = (r - 1 <= pk) ? k - 1 : p - r;
                for (int j = j1; j <= j2; j++) {
                    double den = ndu(pk + 1, rk + j);
                    a(s2, j) = (den == 0.0) ? 0.0 : (a(s1, j) - a(s1, j - 1)) / den;
                    d += a(s2, j) * ndu(rk + j, pk);
                }
                if (r <= pk) {
                    double den = ndu(pk + 1, r);
                    a(s2, k) = (den == 0.0) ? 0.0 : -a(s1, k - 1) / den;
                    d += a(s2, k) * ndu(r, pk);
                }
                ders(k, r) = d;
                std::swap(s1, s2);
            }
        }
        double fac = p;
        for (int k = 1; k <= n; k++) {
            for (int j = 0; j <= p; j++) ders(k, j) *= fac;
            fac *= (p - k);
        }
    }

public:
    NUBSTrajectory(int sys_order = 3) : s(sys_order), p(2 * sys_order - 1) {}
    ~NUBSTrajectory() {}

    inline int getS() const { return s; }
    inline int getCtrlPtNum(int M) const { return M + 2 * s - 1; }
    inline double getTotalDuration() const { return knots(knots.size() - 1); }

    // 将最终优化好的 P 和 T 灌入对象中，供外部进行 evaluate 求值
    inline void setTrajectory(const Eigen::VectorXd& T, const Eigen::MatrixXd& P) {
        control_points = P;
        knots = generateKnots(T, P.rows());
    }

    // =================================================================================
    // ★ 极速 B 样条求值：利用局部支撑，O(p^2) 的速度求出位置、速度、加速度！
    // =================================================================================
    Eigen::Matrix<double, Dim, 1> evaluate(double t, int d_ord = 0) const 
    {
        if (t <= knots(p)) t = knots(p);
        if (t >= knots(knots.size() - p - 1)) t = knots(knots.size() - p - 1) - 1e-9;
        
        int N_c = control_points.rows();
        int span = findSpan(t, N_c, knots);
        
        Eigen::MatrixXd ders;
        dersBasisFuns(d_ord, span, t, knots, ders);
        
        Eigen::Matrix<double, Dim, 1> res = Eigen::Matrix<double, Dim, 1>::Zero();
        // 局部求和：仅由相关的 p+1 个控制点决定
        for (int j = 0; j <= p; j++) {
            res += ders(d_ord, j) * control_points.row(span - p + j).transpose();
        }
        return res;
    }

    inline Eigen::VectorXd generateKnots(const Eigen::VectorXd& T, int N_c) const {
        int num_knots = N_c + p + 1;
        Eigen::VectorXd u = Eigen::VectorXd::Zero(num_knots);
        for(int i = 0; i <= p; i++) u(i) = 0.0;
        double current_t = 0.0;
        for(int i = 0; i < T.size(); i++) {
            current_t += T(i);
            u(p + 1 + i) = current_t;
        }
        for(int i = p + 1 + T.size(); i < num_knots; i++) u(i) = current_t;
        return u;
    }

    inline void generateInitialControlPoints(const Eigen::MatrixXd& waypoints, 
                                             const Eigen::MatrixXd& headState, 
                                             const Eigen::MatrixXd& tailState, 
                                             const Eigen::VectorXd& T, 
                                             Eigen::MatrixXd& P_full) const
    {
        int M = T.size();
        int N_c = getCtrlPtNum(M);
        P_full.resize(N_c, Dim);
        Eigen::VectorXd u = generateKnots(T, N_c);
        
        BandedSystem A;
        A.create(N_c, p, p); 
        Eigen::Matrix<double, Eigen::Dynamic, Dim> b = Eigen::Matrix<double, Eigen::Dynamic, Dim>::Zero(N_c, Dim);
        
        int row = 0;
        for (int d = 0; d < s; d++) {
            Eigen::MatrixXd ders;
            dersBasisFuns(d, p, u(p), u, ders);
            for (int j = 0; j <= p; j++) A(row, j) = ders(d, j);
            b.row(row++) = headState.col(d).transpose();
        }
        
        for (int i = 1; i < M; i++) {
            double t = u(p + i); 
            int span = p + i;
            Eigen::MatrixXd ders;
            dersBasisFuns(0, span, t, u, ders);
            for (int j = 0; j <= p; j++) A(row, span - p + j) = ders(0, j);
            b.row(row++) = waypoints.row(i);
        }
        
        double t_end = u(N_c);
        for (int d = s - 1; d >= 0; d--) {
            Eigen::MatrixXd ders;
            dersBasisFuns(d, N_c - 1, t_end, u, ders);
            for (int j = 0; j <= p; j++) A(row, N_c - 1 - p + j) = ders(d, j);
            b.row(row++) = tailState.col(d).transpose();
        }
        
        A.factorizeLU();
        A.solve(b);
        P_full = b;
        A.destroy();
    }

    inline void resolveBoundaryCtrlPts(const Eigen::MatrixXd& headState, 
                                       const Eigen::MatrixXd& tailState, 
                                       const Eigen::VectorXd& T, 
                                       Eigen::MatrixXd& P_full) const 
    {
        int N_c = P_full.rows();
        Eigen::VectorXd u = generateKnots(T, N_c);
        Eigen::MatrixXd ders;

        dersBasisFuns(s - 1, p, 0.0, u, ders);
        for (int k = 0; k < s; k++) {
            Eigen::RowVectorXd val = headState.col(k).transpose();
            for (int j = 0; j < k; j++) val -= ders(k, j) * P_full.row(j);
            P_full.row(k) = val / ders(k, k);
        }

        Eigen::VectorXd T_rev(T.size());
        for(int i=0; i<T.size(); ++i) T_rev(i) = T(T.size()-1-i);
        Eigen::VectorXd u_rev = generateKnots(T_rev, N_c);
        Eigen::MatrixXd ders_rev;
        dersBasisFuns(s - 1, p, 0.0, u_rev, ders_rev); 

        for (int k = 0; k < s; k++) {
            Eigen::RowVectorXd val = tailState.col(k).transpose() * ((k % 2 != 0) ? -1.0 : 1.0);
            for (int j = 0; j < k; j++) val -= ders_rev(k, j) * P_full.row(N_c - 1 - j);
            P_full.row(N_c - 1 - k) = val / ders_rev(k, k);
        }
    }

    void getEnergyAndGradP(const Eigen::VectorXd& T, 
                           const Eigen::MatrixXd& P_full, 
                           const double& max_v, const double& max_a, const double& weight_kinematic,
                           double& cost, Eigen::MatrixXd& gradP) const 
    {
        int N_c = P_full.rows();
        Eigen::VectorXd u = generateKnots(T, N_c);
        cost = 0.0; gradP.setZero(N_c, Dim);
        
        int n_idx = std::min(s, 6);
        const auto& nodes = gauss_nodes[n_idx];
        const auto& weights = gauss_weights[n_idx];
        double max_v2 = max_v * max_v;
        double max_a2 = max_a * max_a;

        for (int i = p; i < u.size() - p - 1; i++) {
            double t_start = u(i), t_end = u(i + 1);
            if (t_end - t_start < 1e-9) continue;
            double len = t_end - t_start;
            double mid = (t_end + t_start) / 2.0;

            for (size_t k = 0; k < nodes.size(); k++) {
                double t = mid + (len / 2.0) * nodes[k];
                double w = weights[k] * (len / 2.0);

                int span = findSpan(t, N_c, u);
                Eigen::MatrixXd ders;
                dersBasisFuns(s, span, t, u, ders);

                Eigen::Matrix<double, 1, Dim> val_s = Eigen::Matrix<double, 1, Dim>::Zero();
                Eigen::Matrix<double, 1, Dim> val_v = Eigen::Matrix<double, 1, Dim>::Zero();
                Eigen::Matrix<double, 1, Dim> val_a = Eigen::Matrix<double, 1, Dim>::Zero();
                
                for (int j = 0; j <= p; j++) {
                    val_s += ders(s, j) * P_full.row(span - p + j);
                    if(s >= 2) {
                        val_v += ders(1, j) * P_full.row(span - p + j);
                        val_a += ders(2, j) * P_full.row(span - p + j);
                    }
                }

                cost += w * val_s.squaredNorm();
                for (int j = 0; j <= p; j++) gradP.row(span - p + j) += 2.0 * w * ders(s, j) * val_s;

                if (s >= 2) {
                    double viola_v = val_v.squaredNorm() - max_v2;
                    if (viola_v > 0) {
                        cost += w * weight_kinematic * viola_v * viola_v;
                        for (int j = 0; j <= p; j++) 
                            gradP.row(span - p + j) += 4.0 * w * weight_kinematic * viola_v * ders(1, j) * val_v;
                    }
                    double viola_a = val_a.squaredNorm() - max_a2;
                    if (viola_a > 0) {
                        cost += w * weight_kinematic * viola_a * viola_a;
                        for (int j = 0; j <= p; j++) 
                            gradP.row(span - p + j) += 4.0 * w * weight_kinematic * viola_a * ders(2, j) * val_a;
                    }
                }
            }
        }
    }

    void getGradT_FD(const Eigen::MatrixXd& headState, const Eigen::MatrixXd& tailState, 
                     const Eigen::VectorXd& T, const Eigen::MatrixXd& P_inner, 
                     const double& max_v, const double& max_a, const double& weight_kinematic,
                     Eigen::VectorXd& gradT) const 
    {
        int M = T.size();
        gradT.setZero(M);
        double eps = 1e-6;
        int N_c = getCtrlPtNum(M); 
        Eigen::MatrixXd P_full(N_c, Dim);

        for (int k = 0; k < P_inner.rows(); k++) P_full.row(s + k) = P_inner.row(k);

        for (int i = 0; i < M; i++) {
            Eigen::VectorXd T_plus = T, T_minus = T;
            T_plus(i) += eps;
            T_minus(i) = std::max(T_minus(i) - eps, 1e-8); 
            double actual_eps = T_plus(i) - T_minus(i);

            resolveBoundaryCtrlPts(headState, tailState, T_plus, P_full);
            double cost_plus; Eigen::MatrixXd dummy_gradP;
            getEnergyAndGradP(T_plus, P_full, max_v, max_a, weight_kinematic, cost_plus, dummy_gradP);

            resolveBoundaryCtrlPts(headState, tailState, T_minus, P_full);
            double cost_minus;
            getEnergyAndGradP(T_minus, P_full, max_v, max_a, weight_kinematic, cost_minus, dummy_gradP);

            gradT(i) = (cost_plus - cost_minus) / actual_eps;
        }
    }
};

// } 
// #endif

// #ifndef BSPLINE_TRAJECTORY_HPP
// #define BSPLINE_TRAJECTORY_HPP

// #include <eigen3/Eigen/Eigen>
// #include <vector>
// #include <cmath>
// #include <algorithm>
// #include <iostream>

// namespace nubs
// {

// class BandedSystem 
// {
// public:
//     inline void create(const int &n, const int &p, const int &q) {
//         destroy();
//         N = n; lowerBw = p; upperBw = q;
//         int actualSize = N * (lowerBw + upperBw + 1);
//         ptrData = new double[actualSize];
//         std::fill_n(ptrData, actualSize, 0.0);
//     }
//     inline void destroy() {
//         if (ptrData != nullptr) { delete[] ptrData; ptrData = nullptr; }
//     }
// private:
//     int N, lowerBw, upperBw;
//     double *ptrData = nullptr;
// public:
//     inline void reset(void) { std::fill_n(ptrData, N * (lowerBw + upperBw + 1), 0.0); }
//     inline const double &operator()(const int &i, const int &j) const { return ptrData[(i - j + upperBw) * N + j]; }
//     inline double &operator()(const int &i, const int &j) { return ptrData[(i - j + upperBw) * N + j]; }
    
//     inline void factorizeLU() {
//         int iM, jM; double cVl;
//         for (int k = 0; k <= N - 2; k++) {
//             iM = std::min(k + lowerBw, N - 1);
//             cVl = operator()(k, k);
//             for (int i = k + 1; i <= iM; i++) { if (operator()(i, k) != 0.0) operator()(i, k) /= cVl; }
//             jM = std::min(k + upperBw, N - 1);
//             for (int j = k + 1; j <= jM; j++) {
//                 cVl = operator()(k, j);
//                 if (cVl != 0.0) {
//                     for (int i = k + 1; i <= iM; i++) {
//                         if (operator()(i, k) != 0.0) operator()(i, j) -= operator()(i, k) * cVl;
//                     }
//                 }
//             }
//         }
//     }

//     template <typename EIGENMAT>
//     inline void solve(EIGENMAT &b) const {
//         int iM;
//         for (int j = 0; j <= N - 1; j++) {
//             iM = std::min(j + lowerBw, N - 1);
//             for (int i = j + 1; i <= iM; i++) {
//                 if (operator()(i, j) != 0.0) b.row(i) -= operator()(i, j) * b.row(j);
//             }
//         }
//         for (int j = N - 1; j >= 0; j--) {
//             b.row(j) /= operator()(j, j);
//             iM = std::max(0, j - upperBw);
//             for (int i = iM; i <= j - 1; i++) {
//                 if (operator()(i, j) != 0.0) b.row(i) -= operator()(i, j) * b.row(j);
//             }
//         }
//     }
// };

// template<int Dim>
// class NUBSTrajectory 
// {
// private:
//     int s;         
//     int p;         
    
//     Eigen::VectorXd knots;
//     Eigen::Matrix<double, Eigen::Dynamic, Dim> control_points;

//     std::vector<std::vector<double>> gauss_nodes = {
//         {}, {0.0}, 
//         {-0.577350269, 0.577350269}, 
//         {-0.774596669, 0.0, 0.774596669}, 
//         {-0.861136311, -0.339981043, 0.339981043, 0.861136311},
//         {-0.906179845, -0.538469310, 0.0, 0.538469310, 0.906179845},
//         {-0.932469514, -0.661209386, -0.238619186, 0.238619186, 0.661209386, 0.932469514}
//     };
//     std::vector<std::vector<double>> gauss_weights = {
//         {}, {2.0}, 
//         {1.0, 1.0}, 
//         {0.555555555, 0.888888888, 0.555555555}, 
//         {0.347854845, 0.652145154, 0.652145154, 0.347854845},
//         {0.236926885, 0.478628670, 0.568888888, 0.478628670, 0.236926885},
//         {0.171324492, 0.360761573, 0.467913934, 0.467913934, 0.360761573, 0.171324492}
//     };

// public:
//     NUBSTrajectory(int sys_order = 3) : s(sys_order), p(2 * sys_order - 1) {}
//     ~NUBSTrajectory() {}

//     inline int getS() const { return s; }
//     inline int getPDeg() const { return p; }
//     inline int getCtrlPtNum(int M) const { return M + 2 * s - 1; }
//     inline double getTotalDuration() const { return knots(knots.size() - 1); }

//     // 公开暴露供优化器采样查表使用
//     inline int findSpan(double t, int num_ctrl_pts, const Eigen::VectorXd& u) const {
//         if (t >= u(num_ctrl_pts)) return num_ctrl_pts - 1;
//         if (t <= u(p)) return p;
//         int low = p, high = num_ctrl_pts, mid;
//         while (low < high) {
//             mid = (low + high) / 2;
//             if (t < u(mid)) high = mid;
//             else low = mid + 1;
//         }
//         return low - 1;
//     }

//     // 公开暴露，De Boor-Cox 获取任意阶导数
//     inline void dersBasisFuns(int n, int span, double t, const Eigen::VectorXd& u, Eigen::MatrixXd& ders) const {
//         n = std::min(n, p); 
//         ders.setZero(n + 1, p + 1);
//         Eigen::MatrixXd ndu(p + 1, p + 1);
//         ndu(0, 0) = 1.0;
//         Eigen::VectorXd left(p + 1), right(p + 1);

//         for (int j = 1; j <= p; j++) {
//             left(j) = t - u(span + 1 - j);
//             right(j) = u(span + j) - t;
//             double saved = 0.0;
//             for (int r = 0; r < j; r++) {
//                 ndu(j, r) = right(r + 1) + left(j - r);
//                 double temp = (ndu(j, r) == 0.0) ? 0.0 : ndu(r, j - 1) / ndu(j, r);
//                 ndu(r, j) = saved + right(r + 1) * temp;
//                 saved = left(j - r) * temp;
//             }
//             ndu(j, j) = saved;
//         }
//         for (int j = 0; j <= p; j++) ders(0, j) = ndu(j, p);
//         if (n == 0) return;

//         Eigen::MatrixXd a(2, p + 1);
//         for (int r = 0; r <= p; r++) {
//             int s1 = 0, s2 = 1;
//             a(0, 0) = 1.0;
//             for (int k = 1; k <= n; k++) {
//                 double d = 0.0;
//                 int rk = r - k, pk = p - k;
//                 if (r >= k) {
//                     double den = ndu(pk + 1, rk);
//                     a(s2, 0) = (den == 0.0) ? 0.0 : a(s1, 0) / den;
//                     d = a(s2, 0) * ndu(rk, pk);
//                 }
//                 int j1 = (rk >= -1) ? 1 : -rk;
//                 int j2 = (r - 1 <= pk) ? k - 1 : p - r;
//                 for (int j = j1; j <= j2; j++) {
//                     double den = ndu(pk + 1, rk + j);
//                     a(s2, j) = (den == 0.0) ? 0.0 : (a(s1, j) - a(s1, j - 1)) / den;
//                     d += a(s2, j) * ndu(rk + j, pk);
//                 }
//                 if (r <= pk) {
//                     double den = ndu(pk + 1, r);
//                     a(s2, k) = (den == 0.0) ? 0.0 : -a(s1, k - 1) / den;
//                     d += a(s2, k) * ndu(r, pk);
//                 }
//                 ders(k, r) = d;
//                 std::swap(s1, s2);
//             }
//         }
//         double fac = p;
//         for (int k = 1; k <= n; k++) {
//             for (int j = 0; j <= p; j++) ders(k, j) *= fac;
//             fac *= (p - k);
//         }
//     }

//     inline Eigen::VectorXd generateKnots(const Eigen::VectorXd& T, int N_c) const {
//         int num_knots = N_c + p + 1;
//         Eigen::VectorXd u = Eigen::VectorXd::Zero(num_knots);
//         for(int i = 0; i <= p; i++) u(i) = 0.0;
//         double current_t = 0.0;
//         for(int i = 0; i < T.size(); i++) {
//             current_t += T(i);
//             u(p + 1 + i) = current_t;
//         }
//         for(int i = p + 1 + T.size(); i < num_knots; i++) u(i) = current_t;
//         return u;
//     }

//     inline void setTrajectory(const Eigen::VectorXd& T, const Eigen::MatrixXd& P) {
//         control_points = P;
//         knots = generateKnots(T, P.rows());
//     }

//     Eigen::Matrix<double, Dim, 1> evaluate(double t, int d_ord = 0) const 
//     {
//         if (t <= knots(p)) t = knots(p);
//         if (t >= knots(knots.size() - p - 1)) t = knots(knots.size() - p - 1) - 1e-9;
        
//         int N_c = control_points.rows();
//         int span = findSpan(t, N_c, knots);
//         Eigen::MatrixXd ders;
//         dersBasisFuns(d_ord, span, t, knots, ders);
        
//         Eigen::Matrix<double, Dim, 1> res = Eigen::Matrix<double, Dim, 1>::Zero();
//         for (int j = 0; j <= p; j++) {
//             res += ders(d_ord, j) * control_points.row(span - p + j).transpose();
//         }
//         return res;
//     }

//     inline void generateInitialControlPoints(const Eigen::MatrixXd& waypoints, 
//                                              const Eigen::MatrixXd& headState, 
//                                              const Eigen::MatrixXd& tailState, 
//                                              const Eigen::VectorXd& T, 
//                                              Eigen::MatrixXd& P_full) const
//     {
//         int M = T.size();
//         int N_c = getCtrlPtNum(M);
//         P_full.resize(N_c, Dim);
//         Eigen::VectorXd u = generateKnots(T, N_c);
        
//         BandedSystem A;
//         A.create(N_c, p, p); 
//         Eigen::Matrix<double, Eigen::Dynamic, Dim> b = Eigen::Matrix<double, Eigen::Dynamic, Dim>::Zero(N_c, Dim);
        
//         int row = 0;
//         for (int d = 0; d < s; d++) {
//             Eigen::MatrixXd ders;
//             dersBasisFuns(d, p, u(p), u, ders);
//             for (int j = 0; j <= p; j++) A(row, j) = ders(d, j);
//             b.row(row++) = headState.col(d).transpose();
//         }
//         for (int i = 1; i < M; i++) {
//             double t = u(p + i); 
//             int span = p + i;
//             Eigen::MatrixXd ders;
//             dersBasisFuns(0, span, t, u, ders);
//             for (int j = 0; j <= p; j++) A(row, span - p + j) = ders(0, j);
//             b.row(row++) = waypoints.row(i);
//         }
//         double t_end = u(N_c);
//         for (int d = s - 1; d >= 0; d--) {
//             Eigen::MatrixXd ders;
//             dersBasisFuns(d, N_c - 1, t_end, u, ders);
//             for (int j = 0; j <= p; j++) A(row, N_c - 1 - p + j) = ders(d, j);
//             b.row(row++) = tailState.col(d).transpose();
//         }
        
//         A.factorizeLU();
//         A.solve(b);
//         P_full = b;
//         A.destroy();
//     }

//     inline void resolveBoundaryCtrlPts(const Eigen::MatrixXd& headState, 
//                                        const Eigen::MatrixXd& tailState, 
//                                        const Eigen::VectorXd& T, 
//                                        Eigen::MatrixXd& P_full) const 
//     {
//         int N_c = P_full.rows();
//         Eigen::VectorXd u = generateKnots(T, N_c);
//         Eigen::MatrixXd ders;

//         dersBasisFuns(s - 1, p, 0.0, u, ders);
//         for (int k = 0; k < s; k++) {
//             Eigen::RowVectorXd val = headState.col(k).transpose();
//             for (int j = 0; j < k; j++) val -= ders(k, j) * P_full.row(j);
//             P_full.row(k) = val / ders(k, k);
//         }

//         Eigen::VectorXd T_rev(T.size());
//         for(int i=0; i<T.size(); ++i) T_rev(i) = T(T.size()-1-i);
//         Eigen::VectorXd u_rev = generateKnots(T_rev, N_c);
//         Eigen::MatrixXd ders_rev;
//         dersBasisFuns(s - 1, p, 0.0, u_rev, ders_rev); 

//         for (int k = 0; k < s; k++) {
//             Eigen::RowVectorXd val = tailState.col(k).transpose() * ((k % 2 != 0) ? -1.0 : 1.0);
//             for (int j = 0; j < k; j++) val -= ders_rev(k, j) * P_full.row(N_c - 1 - j);
//             P_full.row(N_c - 1 - k) = val / ders_rev(k, k);
//         }
//     }

//     // ★ 纯净能量计算：仅保留平滑度代价的高斯积分
//     void getEnergyAndGradP(const Eigen::VectorXd& T, 
//                            const Eigen::MatrixXd& P_full, 
//                            double& cost, Eigen::MatrixXd& gradP) const 
//     {
//         int N_c = P_full.rows();
//         Eigen::VectorXd u = generateKnots(T, N_c);
//         cost = 0.0; gradP.setZero(N_c, Dim);
        
//         int n_idx = std::min(s, 6);
//         const auto& nodes = gauss_nodes[n_idx];
//         const auto& weights = gauss_weights[n_idx];

//         for (int i = p; i < u.size() - p - 1; i++) {
//             double t_start = u(i), t_end = u(i + 1);
//             if (t_end - t_start < 1e-9) continue;
//             double len = t_end - t_start;
//             double mid = (t_end + t_start) / 2.0;

//             for (size_t k = 0; k < nodes.size(); k++) {
//                 double t = mid + (len / 2.0) * nodes[k];
//                 double w = weights[k] * (len / 2.0);

//                 int span = findSpan(t, N_c, u);
//                 Eigen::MatrixXd ders;
//                 dersBasisFuns(s, span, t, u, ders);

//                 Eigen::Matrix<double, 1, Dim> val_s = Eigen::Matrix<double, 1, Dim>::Zero();
//                 for (int j = 0; j <= p; j++) {
//                     val_s += ders(s, j) * P_full.row(span - p + j);
//                 }

//                 cost += w * val_s.squaredNorm();
//                 for (int j = 0; j <= p; j++) {
//                     gradP.row(span - p + j) += 2.0 * w * ders(s, j) * val_s;
//                 }
//             }
//         }
//     }
// };

// } // namespace nubs
// #endif


#ifndef BSPLINE_TRAJECTORY_HPP
#define BSPLINE_TRAJECTORY_HPP

#include <eigen3/Eigen/Eigen>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace nubs
{

class BandedSystem 
{
public:
    inline void create(const int &n, const int &p, const int &q) {
        destroy();
        N = n; lowerBw = p; upperBw = q;
        int actualSize = N * (lowerBw + upperBw + 1);
        ptrData = new double[actualSize];
        std::fill_n(ptrData, actualSize, 0.0);
    }
    inline void destroy() {
        if (ptrData != nullptr) { delete[] ptrData; ptrData = nullptr; }
    }
private:
    int N, lowerBw, upperBw;
    double *ptrData = nullptr;
public:
    inline void reset(void) { std::fill_n(ptrData, N * (lowerBw + upperBw + 1), 0.0); }
    inline const double &operator()(const int &i, const int &j) const { return ptrData[(i - j + upperBw) * N + j]; }
    inline double &operator()(const int &i, const int &j) { return ptrData[(i - j + upperBw) * N + j]; }
    
    inline void factorizeLU() {
        int iM, jM; double cVl;
        for (int k = 0; k <= N - 2; k++) {
            iM = std::min(k + lowerBw, N - 1);
            cVl = operator()(k, k);
            for (int i = k + 1; i <= iM; i++) { if (operator()(i, k) != 0.0) operator()(i, k) /= cVl; }
            jM = std::min(k + upperBw, N - 1);
            for (int j = k + 1; j <= jM; j++) {
                cVl = operator()(k, j);
                if (cVl != 0.0) {
                    for (int i = k + 1; i <= iM; i++) {
                        if (operator()(i, k) != 0.0) operator()(i, j) -= operator()(i, k) * cVl;
                    }
                }
            }
        }
    }

    template <typename EIGENMAT>
    inline void solve(EIGENMAT &b) const {
        int iM;
        for (int j = 0; j <= N - 1; j++) {
            iM = std::min(j + lowerBw, N - 1);
            for (int i = j + 1; i <= iM; i++) {
                if (operator()(i, j) != 0.0) b.row(i) -= operator()(i, j) * b.row(j);
            }
        }
        for (int j = N - 1; j >= 0; j--) {
            b.row(j) /= operator()(j, j);
            iM = std::max(0, j - upperBw);
            for (int i = iM; i <= j - 1; i++) {
                if (operator()(i, j) != 0.0) b.row(i) -= operator()(i, j) * b.row(j);
            }
        }
    }

    template <typename EIGENMAT>
    inline void solveAdj(EIGENMAT &b) const 
    {
        int iM;
        for (int j = 0; j <= N - 1; j++) {
            b.row(j) /= operator()(j, j);
            iM = std::min(j + upperBw, N - 1);
            for (int i = j + 1; i <= iM; i++) {
                if (operator()(j, i) != 0.0) b.row(i) -= operator()(j, i) * b.row(j);
            }
        }
        for (int j = N - 1; j >= 0; j--) {
            iM = std::max(0, j - lowerBw);
            for (int i = iM; i <= j - 1; i++) {
                if (operator()(j, i) != 0.0) b.row(i) -= operator()(j, i) * b.row(j);
            }
        }
    }
};

template<int Dim>
class NUBSTrajectory 
{
private:
    int s;             
    int p; 
    int N_c; 
    
    Eigen::VectorXd knots;
    Eigen::Matrix<double, Eigen::Dynamic, Dim> control_points;

public:
    BandedSystem A;  

    NUBSTrajectory(int sys_order = 3) : s(sys_order), p(2 * sys_order - 1) {}
    ~NUBSTrajectory() { A.destroy(); }

    inline int getS() const { return s; }
    inline int getPDeg() const { return p; }
    inline int getCtrlPtNum(int M) const { return M + 2 * s - 1; }
    inline double getTotalDuration() const { return knots(knots.size() - 1); }

    inline int findSpan(double t, int num_ctrl_pts, const Eigen::VectorXd& u) const {
        if (t >= u(num_ctrl_pts)) return num_ctrl_pts - 1;
        if (t <= u(p)) return p;
        int low = p, high = num_ctrl_pts, mid;
        while (low < high) {
            mid = (low + high) / 2;
            if (t < u(mid)) high = mid;
            else low = mid + 1;
        }
        return low - 1;
    }

    inline void dersBasisFuns(int n, int span, double t, const Eigen::VectorXd& u, Eigen::Matrix<double, 8, 8>& ders) const {
        n = std::min(n, p); 
        ders.setZero();
        double ndu[8][8] = {{0}}; 
        ndu[0][0] = 1.0;
        double left[8], right[8];

        for (int j = 1; j <= p; j++) {
            left[j] = t - u(span + 1 - j);
            right[j] = u(span + j) - t;
            double saved = 0.0;
            for (int r = 0; r < j; r++) {
                ndu[j][r] = right[r + 1] + left[j - r];
                double temp = (ndu[j][r] == 0.0) ? 0.0 : ndu[r][j - 1] / ndu[j][r];
                ndu[r][j] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            ndu[j][j] = saved;
        }
        for (int j = 0; j <= p; j++) ders(0, j) = ndu[j][p];
        if (n == 0) return;

        double a[2][8] = {{0}};
        for (int r = 0; r <= p; r++) {
            int s1 = 0, s2 = 1;
            a[0][0] = 1.0;
            for (int k = 1; k <= n; k++) {
                double d = 0.0;
                int rk = r - k, pk = p - k;
                if (r >= k) {
                    double den = ndu[pk + 1][rk];
                    a[s2][0] = (den == 0.0) ? 0.0 : a[s1][0] / den;
                    d = a[s2][0] * ndu[rk][pk];
                }
                int j1 = (rk >= -1) ? 1 : -rk;
                int j2 = (r - 1 <= pk) ? k - 1 : p - r;
                for (int j = j1; j <= j2; j++) {
                    double den = ndu[pk + 1][rk + j];
                    a[s2][j] = (den == 0.0) ? 0.0 : (a[s1][j] - a[s1][j - 1]) / den;
                    d += a[s2][j] * ndu[rk + j][pk];
                }
                if (r <= pk) {
                    double den = ndu[pk + 1][r];
                    a[s2][k] = (den == 0.0) ? 0.0 : -a[s1][k - 1] / den;
                    d += a[s2][k] * ndu[r][pk];
                }
                ders(k, r) = d;
                std::swap(s1, s2);
            }
        }
        double fac = p;
        for (int k = 1; k <= n; k++) {
            for (int j = 0; j <= p; j++) ders(k, j) *= fac;
            fac *= (p - k);
        }
    }

    inline Eigen::VectorXd generateKnots(const std::vector<double>& time_nodes, int num_c) const {
        int num_knots = num_c + p + 1;
        Eigen::VectorXd u = Eigen::VectorXd::Zero(num_knots);
        int idx = 0;
        for (int i = 0; i <= p; i++) u(idx++) = time_nodes.front();
        for (size_t i = 1; i < time_nodes.size() - 1; i++) u(idx++) = time_nodes[i];
        for (int i = 0; i <= p; i++) u(idx++) = time_nodes.back();
        return u;
    }

    void setTrajectory(const Eigen::VectorXd& T, const Eigen::MatrixXd& P) {
        control_points = P;
        knots = generateKnots(T, P.rows());
    }

    // 强行指定 MatrixXd 防止模板推导失败
    void generate(const std::vector<double> &time_nodes,
                  const Eigen::MatrixXd& waypoints,
                  const Eigen::MatrixXd& start_derivatives,
                  const Eigen::MatrixXd& end_derivatives)
    {
        int M = waypoints.rows() - 1; 
        if(M < 1) return;

        N_c = M + 2 * s - 1;
        knots = generateKnots(time_nodes, N_c);
        
        A.create(N_c, p, p); 
        Eigen::MatrixXd b = Eigen::MatrixXd::Zero(N_c, Dim);

        int row = 0;
        Eigen::Matrix<double, 8, 8> ders;

        auto assign_A = [&](int r, double t, int d) {
            int span = findSpan(t, N_c, knots);
            dersBasisFuns(d, span, t, knots, ders);
            for (int j = 0; j <= p; j++) {
                int col = span - p + j;
                if (col >= 0 && col < N_c) A(r, col) = ders(d, j);
            }
        };

        // 拆分赋值语句，避免 C++ Sequence Point 导致维度崩溃
        assign_A(row, time_nodes.front(), 0); 
        b.row(row) = waypoints.row(0); row++;

        for (int d = 1; d < s; d++) { 
            assign_A(row, time_nodes.front(), d); 
            b.row(row) = start_derivatives.row(d - 1); row++; 
        }
        for (int i = 1; i < M; i++) { 
            assign_A(row, time_nodes[i], 0); 
            b.row(row) = waypoints.row(i); row++; 
        }
        for (int d = s - 1; d >= 1; d--) { 
            assign_A(row, time_nodes.back(), d); 
            b.row(row) = end_derivatives.row(d - 1); row++; 
        }
        assign_A(row, time_nodes.back(), 0); 
        b.row(row) = waypoints.row(M); row++;

        A.factorizeLU();
        A.solve(b);
        control_points = b;
    }

    Eigen::Matrix<double,Dim,1> evaluate(double t, int d_ord = 0) const 
    {
        if (t <= knots(p)) t = knots(p);
        if (t >= knots(knots.size() - p - 1)) t = knots(knots.size() - p - 1) - 1e-9;
        
        int span = findSpan(t, N_c, knots);
        Eigen::Matrix<double, 8, 8> ders;
        dersBasisFuns(d_ord, span, t, knots, ders);
        
        Eigen::Matrix<double, Dim, 1> res = Eigen::Matrix<double, Dim, 1>::Zero();
        for (int j = 0; j <= p; j++) {
            res += ders(d_ord, j) * control_points.row(span - p + j).transpose();
        }
        return res;
    }

    const Eigen::VectorXd& getKnots() const { return knots; }
    const Eigen::Matrix<double, Eigen::Dynamic, Dim>& getControlPoints() const { return control_points; }

    // ★ 显式传入 M 参数，彻底解决 gradByPoints.rows() = 0 的越界 bug
    inline void propogateGrad(const Eigen::MatrixXd &gradByP, Eigen::MatrixXd &gradByPoints, int M)
    {
        gradByPoints.resize(M - 1, Dim); 
        Eigen::MatrixXd G = gradByP;
        A.solveAdj(G); 
        for (int i = 0; i < M - 1; i++) {
            gradByPoints.row(i) = G.row(s + i);
        }
    }
};

} 
#endif