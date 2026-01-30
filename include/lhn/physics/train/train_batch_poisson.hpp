#pragma once
#include <cstddef>

namespace lhn::physics::training {

struct PhysicsTrainer; 

void train_batch_poisson(
    PhysicsTrainer& trainer,
    const double* X_flat,
    const double* kappa_ptr,
    size_t n_samples,
    int epochs
);

}