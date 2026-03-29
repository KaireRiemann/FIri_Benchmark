/*
    sphere_opt.hpp
    Header-only library for unconstrained optimization on a Sphere Manifold.
    Uses Exact Geodesic Update with Non-monotone Line Search and BB Step Size.
*/

#ifndef SPHERE_OPT_HPP
#define SPHERE_OPT_HPP

#include <Eigen/Eigen>
#include <cmath>
#include <tuple>
#include <algorithm>

namespace sphere_opt
{
    // ========================================================================
    // 球面精确测地线更新 (Exact Geodesic Update on Sphere)
    // ========================================================================
    inline Eigen::VectorXd geodesicUpdate(const Eigen::VectorXd &y, const Eigen::VectorXd &g, double alpha)
    {
        double norm_g = g.norm();
        if (norm_g < 1e-12)
        {
            return y;
        }
        // 球面测地线闭式解析解: y_next = y*cos(||g||*alpha) - (g/||g||)*sin(||g||*alpha)
        Eigen::VectorXd y_next = y * std::cos(alpha * norm_g) - (g / norm_g) * std::sin(alpha * norm_g);
        
        // 消除浮点数累积误差，严格拉回标准圆
        return y_next.normalized(); 
    }

    // ========================================================================
    // 带 BB 步长的非单调黎曼线搜索求解器
    // ========================================================================
    template <typename Evaluator>
    inline int optimizeOnSphere(Eigen::VectorXd &y, Evaluator &&evaluator, 
                                  double tol = 1e-4, int max_iter = 50)
    {
        const double eta = 0.85;       
        const double delta = 0.5;      
        const double rho1 = 1e-4;      
        const double alpha_min = 1e-8; 
        const double alpha_max = 1e3;  

        auto [f, grad] = evaluator(y);
        
        Eigen::VectorXd g = grad - y.dot(grad) * y; 

        double C = f;
        double Q = 1.0;

        Eigen::VectorXd y_prev = y;
        Eigen::VectorXd g_prev = g;
        double alpha = 1.0; 

        for (int iter = 0; iter < max_iter; ++iter)
        {
            if (g.norm() < tol)
            {
                return iter; 
            }

    
            if (iter > 0)
            {
                Eigen::VectorXd S = y - y_prev;
                Eigen::VectorXd Y = g - g_prev;
                double s_sq = S.squaredNorm();
                double sy = std::abs(S.dot(Y));
                double y_sq = Y.squaredNorm();

                double alpha_BB1 = s_sq / (sy + 1e-12);
                double alpha_BB2 = sy / (y_sq + 1e-12);
                alpha = std::min(alpha_BB1, alpha_BB2);
                alpha = std::max(alpha_min, std::min(alpha, alpha_max));
            }

            double f_prime_initial = -g.squaredNorm();
            double alpha_search = alpha;
            
            Eigen::VectorXd y_candidate = geodesicUpdate(y, g, alpha_search);
            auto [f_candidate, grad_candidate] = evaluator(y_candidate);

   
            while (f_candidate > C + rho1 * alpha_search * f_prime_initial)
            {
                alpha_search *= delta;
                if (alpha_search < alpha_min)
                {
                    break; 
                }
                y_candidate = geodesicUpdate(y, g, alpha_search);
                auto res = evaluator(y_candidate);
                f_candidate = res.first;
                grad_candidate = res.second;
            }

            y_prev = y;
            g_prev = g;
            y = y_candidate;
            f = f_candidate;
            grad = grad_candidate;
            
            g = grad - y.dot(grad) * y; 

            Q = eta * Q + 1.0;
            C = (eta * Q * C + f_candidate) / Q;
        }
        return max_iter;
    }
}

#endif // SPHERE_OPT_HPP