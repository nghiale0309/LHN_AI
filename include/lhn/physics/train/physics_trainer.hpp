#pragma once
#include <vector>
#include <array>
#include <lhn/physics/nn/siren_physics_net.hpp>

namespace lhn::physics::training {

struct PhysicsTrainer {
    lhn::physics::nn::SirenPhysicsNet& net;
    double lambda_lens, lambda_poisson, lr;
    
    std::vector<lhn::physics::nn::Node> input_cache;
    std::vector<lhn::physics::nn::Grad> back_grad_cache;

    PhysicsTrainer(lhn::physics::nn::SirenPhysicsNet& n, double l1, double l2, double lr_)
        : net(n), lambda_lens(l1), lambda_poisson(l2), lr(lr_),
          input_cache(2),
          back_grad_cache(1)
    {
        input_cache[0] = {0.0, 0.0, 0.0, 0.0};
        input_cache[1] = {0.0, 0.0, 0.0, 0.0};
        
        back_grad_cache[0] = {0,0,0,0};
    }

    void accumulate_step(double x, double y, double kappa) {
        input_cache[0] = {x, 1.0, 0.0, 0.0};
        input_cache[1] = {y, 0.0, 1.0, 0.0};

        const auto& out = net.forward_nodes(input_cache);

        back_grad_cache[0].dv   = lambda_lens; 
        back_grad_cache[0].ddx  = 0.0;
        back_grad_cache[0].ddy  = 0.0;
        back_grad_cache[0].dlap = 2.0 * lambda_poisson * (out[0].lap - 2.0 * kappa);

        net.backward(back_grad_cache);
    }
    
    void step(const std::vector<std::array<double, 2>>& X, const std::vector<double>& kappa) {
        net.clear_gradients();
        for (size_t i = 0; i < X.size(); i++) {
            accumulate_step(X[i][0], X[i][1], kappa[i]);
        }
        net.update_weights(lr);
    }

    void step(double x, double y, double kappa) {
        net.clear_gradients();
        accumulate_step(x, y, kappa);
        net.update_weights(lr);
    }
};

}