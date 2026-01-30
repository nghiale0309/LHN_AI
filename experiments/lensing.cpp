#include <iostream>
#include "physics_ai/nn/siren_mlp.hpp"
#include "physics_ai/train/trainer.hpp"

using namespace lhn::physics;

int main(){
    nn::SirenMLP net;
    train::Trainer trainer(net);

    for(int i=0;i<1000;i++){
        double L=trainer.step(128);
        std::cout<<i<<" "<<L<<"\n";
    }
}
