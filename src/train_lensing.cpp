#include <lhn/physics/nn/siren_physics_net.hpp>
#include <lhn/physics/train/physics_trainer.hpp>
#include <lhn/physics/train/batch_train.hpp>
#include <lhn/physics/sampling/lensing_sampler.hpp>

using namespace lhn::physics;

int main(){
    nn::SirenPhysicsNet net({2,64,64,1},30.0);
    training::PhysicsTrainer trainer(net,1.0,1.0,1e-4);
    sampling::LensingSampler sampler;

    int total_steps=200000;
    int batch_size=256;

    for(int step=0;step<total_steps;step++)
        training::train_batch(trainer,sampler,batch_size,step,total_steps);
}
