#include <vector>
#include <omp.h>
#include <lhn/physics/train/physics_trainer.hpp>
#include <lhn/physics/train/train_batch_poisson.hpp>

namespace lhn::physics::training {

void train_batch_poisson(
    PhysicsTrainer& trainer,
    const double* X_flat,
    const double* kappa_ptr,
    size_t n_samples,
    int epochs
) {
    int num_threads = omp_get_max_threads();

    std::vector<PhysicsTrainer> local_trainers;
    std::vector<lhn::physics::nn::SirenPhysicsNet> local_nets;
    
    local_nets.reserve(num_threads);
    local_trainers.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        local_nets.push_back(trainer.net); 
        local_trainers.emplace_back(local_nets.back(), trainer.lambda_lens, trainer.lambda_poisson, trainer.lr);
    }

    for (int e = 0; e < epochs; ++e) {
        
        for (int t = 0; t < num_threads; ++t) {
            local_nets[t].clear_gradients();
        }

        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int i = 0; i < static_cast<int>(n_samples); ++i) {
            int tid = omp_get_thread_num();
            local_trainers[tid].accumulate_step(
                X_flat[i * 2],
                X_flat[i * 2 + 1],
                kappa_ptr[i]
            );
        }

        trainer.net.clear_gradients();
        
        for (int t = 0; t < num_threads; ++t) {
            trainer.net.accumulate_gradients_from(local_nets[t]);
        }

        trainer.net.update_weights(trainer.lr);

        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int t = 0; t < num_threads; ++t) {
            local_nets[t].sync_weights_from(trainer.net); 
        }
    }
}

}