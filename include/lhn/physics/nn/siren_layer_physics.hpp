#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <lhn/physics/nn/node.hpp>

namespace lhn::physics::nn {

struct SirenLayerPhysics {
    int in, out;
    double w0;
    
    std::vector<double> W, B;
    std::vector<double> gW, gB; 
    std::vector<double> mW, vW;
    std::vector<double> mB, vB;
    int t; 

    std::vector<Node> in_cache;
    std::vector<Node> y_cache;
    std::vector<Grad> gin_cache;
    
    std::vector<double> z_cache;
    std::vector<double> s_cache; 
    std::vector<double> c_cache;
    
    std::vector<double> dx_prev_cache, dy_prev_cache;
    std::vector<double> lap_prev_cache;

    SirenLayerPhysics(int i, int o, double w, bool is_first = false)
        : in(i), out(o), w0(w), t(0),
          W(i * o), B(o),
          gW(i * o, 0.0), gB(o, 0.0),
          mW(i * o, 0.0), vW(i * o, 0.0),
          mB(o, 0.0), vB(o, 0.0),
          y_cache(o), 
          gin_cache(i),
          z_cache(o), 
          s_cache(o), c_cache(o),
          dx_prev_cache(o), dy_prev_cache(o), lap_prev_cache(o) 
    {
        std::mt19937 rng(42); 
        double limit = is_first ? (1.0 / in) : (std::sqrt(6.0 / in) / w0);
        std::uniform_real_distribution<double> dist(-limit, limit);
        
        for (auto& weight : W) weight = dist(rng);
        for (auto& bias : B) bias = 0.0;
    }

    SirenLayerPhysics(const SirenLayerPhysics& other) = default;

    const std::vector<Node>& forward(const std::vector<Node>& x) {
        in_cache = x;
        const Node* p_x = x.data();
        const double* p_W = W.data();
        
        for (int j = 0; j < out; j++) {
            double z_val = 0.0;
            double dx_val = 0.0;
            double dy_val = 0.0;
            double lap_val = 0.0;

            int offset = j * in;
            
            for (int i = 0; i < in; i++) {
                double w = p_W[offset + i];
                const Node& n = p_x[i];
                
                z_val   += w * n.v;
                dx_val  += w * n.dx;
                dy_val  += w * n.dy;
                lap_val += w * n.lap;
            }

            z_val += B[j];
            
            z_val   *= w0;
            dx_val  *= w0;
            dy_val  *= w0;
            lap_val *= w0;

            z_cache[j] = z_val;
            dx_prev_cache[j] = dx_val;
            dy_prev_cache[j] = dy_val;
            lap_prev_cache[j] = lap_val;

            double s = std::sin(z_val);
            double c = std::cos(z_val);
            
            s_cache[j] = s;
            c_cache[j] = c;

            y_cache[j].v   = s;
            y_cache[j].dx  = c * dx_val;
            y_cache[j].dy  = c * dy_val;
            y_cache[j].lap = -s * (dx_val * dx_val + dy_val * dy_val) + c * lap_val;
        }
        return y_cache;
    }

    const std::vector<Grad>& backward(const std::vector<Grad>& g) {
        std::fill(gin_cache.begin(), gin_cache.end(), Grad{0,0,0,0});

        const double* p_W = W.data();
        
        for (int j = 0; j < out; j++) {
            double s = s_cache[j];
            double c = c_cache[j];
            double dx_pre = dx_prev_cache[j];
            double dy_pre = dy_prev_cache[j];
            double lap_pre = lap_prev_cache[j];
            
            const Grad& gj = g[j];

            double dL_dz = gj.dv * c 
                         - gj.ddx * s * dx_pre 
                         - gj.ddy * s * dy_pre 
                         + gj.dlap * (-c * (dx_pre * dx_pre + dy_pre * dy_pre) - s * lap_pre); 
            
            double dL_ddx_pre = gj.ddx * c + gj.dlap * (-2.0 * s * dx_pre);
            double dL_ddy_pre = gj.ddy * c + gj.dlap * (-2.0 * s * dy_pre);
            double dL_dlap_pre = gj.dlap * c;

            gB[j] += w0 * dL_dz;

            int offset = j * in;
            
            for (int i = 0; i < in; i++) {
                double w = p_W[offset + i];
                const Node& xi = in_cache[i];

                double grad_w_local = dL_dz * xi.v 
                                    + dL_ddx_pre * xi.dx 
                                    + dL_ddy_pre * xi.dy 
                                    + dL_dlap_pre * xi.lap;
                
                gW[offset + i] += w0 * grad_w_local;

                gin_cache[i].dv   += w0 * dL_dz       * w;
                gin_cache[i].ddx  += w0 * dL_ddx_pre  * w;
                gin_cache[i].ddy  += w0 * dL_ddy_pre  * w;
                gin_cache[i].dlap += w0 * dL_dlap_pre * w;
            }
        }
        return gin_cache;
    }

    void update_weights(double lr) {
        t++; 
        double beta1 = 0.9;
        double beta2 = 0.999;
        double eps = 1e-8;
        
        double corr1 = 1.0 / (1.0 - std::pow(beta1, t));
        double corr2 = 1.0 / (1.0 - std::pow(beta2, t));

        for (size_t i = 0; i < W.size(); i++) {
            double grad = gW[i];
            if (grad > 1.0) grad = 1.0;
            if (grad < -1.0) grad = -1.0;

            mW[i] = beta1 * mW[i] + (1.0 - beta1) * grad;
            vW[i] = beta2 * vW[i] + (1.0 - beta2) * grad * grad;
            
            double m_hat = mW[i] * corr1;
            double v_hat = vW[i] * corr2;
            
            W[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            gW[i] = 0.0; 
        }

        for (size_t i = 0; i < B.size(); i++) {
            double grad = gB[i];
            if (grad > 1.0) grad = 1.0;
            if (grad < -1.0) grad = -1.0;

            mB[i] = beta1 * mB[i] + (1.0 - beta1) * grad;
            vB[i] = beta2 * vB[i] + (1.0 - beta2) * grad * grad;
            
            double m_hat = mB[i] * corr1;
            double v_hat = vB[i] * corr2;

            B[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            gB[i] = 0.0;
        }
    }

    void clear_gradients() {
        std::fill(gW.begin(), gW.end(), 0.0);
        std::fill(gB.begin(), gB.end(), 0.0);
    }

    void sync_weights_from(const SirenLayerPhysics& other) {
        std::copy(other.W.begin(), other.W.end(), W.begin());
        std::copy(other.B.begin(), other.B.end(), B.begin());
    }
};

}