#pragma once
#include <vector>
#include <array>
#include <lhn/physics/train/physics_trainer.hpp>
#include <lhn/physics/sampling/lensing_sampler.hpp>
#include <lhn/physics/lensing/kappa_models.hpp>

namespace lhn::physics::training {

inline void train_batch(PhysicsTrainer& trainer,
                        sampling::LensingSampler& sampler,
                        int batch_size,
                        int step,
                        int total_steps){

    lensing::KappaModel kappa_model(lensing::KappaModelType::SIS);

    int n_core = int(0.4 * batch_size);
    int n_ring = int(0.4 * batch_size);
    int n_far  = batch_size - n_core - n_ring;

    std::vector<std::array<double, 2>> X_batch;
    std::vector<double> kappa_batch;

    X_batch.reserve(batch_size);
    kappa_batch.reserve(batch_size);

    for(int i = 0; i < n_core; i++){
        double x, y;
        sampler.sample(x, y, 0);
        double kappa = kappa_model(x, y);
        X_batch.push_back({x, y});
        kappa_batch.push_back(kappa);
    }

    if(step > 0.2 * total_steps){
        for(int i = 0; i < n_ring; i++){
            double x, y;
            sampler.sample(x, y, 1);
            double kappa = kappa_model(x, y);
            X_batch.push_back({x, y});
            kappa_batch.push_back(kappa);
        }
    }

    for(int i = 0; i < n_far; i++){
        double x, y;
        sampler.sample(x, y, 2);
        X_batch.push_back({x, y});
        kappa_batch.push_back(0.0);
    }

    if (!X_batch.empty()) {
        trainer.step(X_batch, kappa_batch);
    }
}
}