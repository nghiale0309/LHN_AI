#pragma once
#include <vector>
#include <omp.h>
#include <lhn/physics/nn/siren_layer_physics.hpp>

namespace lhn::physics::nn {

struct SirenPhysicsNet {
    std::vector<SirenLayerPhysics> layers;

    SirenPhysicsNet(const std::vector<int>& dims, double w0) {
        for (size_t i = 0; i + 1 < dims.size(); i++) {
            bool is_first = (i == 0);
            layers.emplace_back(dims[i], dims[i + 1], w0, is_first);
        }
    }

    SirenPhysicsNet(const SirenPhysicsNet&) = default;

    const std::vector<Node>& forward_nodes(const std::vector<Node>& x) {
        const std::vector<Node>* h = &x; 
        for (auto& l : layers) {
            h = &l.forward(*h); 
        }
        return *h;
    }

    std::vector<double> forward(double x, double y) {
        Node nx = {x, 1.0, 0.0, 0.0}; Node ny = {y, 0.0, 1.0, 0.0};
        auto out = forward_nodes({nx, ny}); 
        if (out.empty()) return {0,0,0,0};
        return {out[0].v, out[0].dx, out[0].dy, out[0].lap};
    }

    std::vector<double> laplacian_batch(const std::vector<double>& flat_X) {
        size_t N = flat_X.size() / 2;
        std::vector<double> results(N);
        for (size_t i = 0; i < N; ++i) {
             Node nx = {flat_X[2*i], 1.0, 0.0, 0.0};
             Node ny = {flat_X[2*i+1], 0.0, 1.0, 0.0};
             std::vector<Node> h = {nx, ny};
             for(auto& l : layers) h = l.forward(h);
             results[i] = h[0].lap;
        }
        return results;
    }

    void backward(const std::vector<Grad>& g) {
        const std::vector<Grad>* grad = &g;
        for (int k = (int)layers.size() - 1; k >= 0; k--) {
            grad = &layers[k].backward(*grad);
        }
    }

    void update_weights(double lr) {
        for (auto& l : layers) l.update_weights(lr);
    }

    void clear_gradients() {
        for (auto& l : layers) l.clear_gradients();
    }

    void accumulate_gradients_from(const SirenPhysicsNet& other) {
        for (size_t k = 0; k < layers.size(); ++k) {
            for (size_t i = 0; i < layers[k].gW.size(); ++i) {
                layers[k].gW[i] += other.layers[k].gW[i];
            }
            for (size_t i = 0; i < layers[k].gB.size(); ++i) {
                layers[k].gB[i] += other.layers[k].gB[i];
            }
        }
    }

    void sync_weights_from(const SirenPhysicsNet& other) {
        for (size_t i = 0; i < layers.size(); i++) {
            layers[i].sync_weights_from(other.layers[i]);
        }
    }
};

}