#pragma once
#include "lhn/physics/nn/siren_mlp.hpp"

namespace lhn::physics::lensing {

struct Potential {
    nn::SirenMLP& net;

    Potential(nn::SirenMLP&n):net(n){}

    double operator()(double x,double y){
        autodiff::Dual dx(x,0.0);
        autodiff::Dual dy(y,0.0);
        return net(dx,dy).v;
    }
};

}
