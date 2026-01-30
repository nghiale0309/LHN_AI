#pragma once
#include <cstdlib>
#include "lhn/physics/lensing/loss.hpp"

namespace lhn::physics::train {

struct Trainer {
    nn::SirenMLP& net;

    Trainer(nn::SirenMLP&n):net(n){}

    double step(int batch){
        double L=0.0;
        for(int i=0;i<batch;i++){
            double x=((double)rand()/RAND_MAX-0.5);
            double y=((double)rand()/RAND_MAX-0.5);
            L+=lensing::lens_loss(net,x,y);
        }
        return L;
    }
};

}
