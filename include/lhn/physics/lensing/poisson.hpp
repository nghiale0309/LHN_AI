#pragma once
#include "laplacian.hpp"

namespace lhn::physics::lensing {

struct PoissonLoss {
    nn::SirenMLP& net;

    PoissonLoss(nn::SirenMLP&n):net(n){}

    double operator()(double x,double y,double kappa){
        double lap=laplacian(net,x,y);
        double r=lap-2.0*kappa;
        return r*r;
    }
};

}
