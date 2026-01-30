#pragma once
#include "lhn/physics/nn/siren_mlp.hpp"

namespace lhn::physics::lensing {

using autodiff::Dual;

inline std::pair<double,double>
gradient(nn::SirenMLP&net,double x,double y){
    Dual dx(x,1.0),dy(y,0.0);
    double gx=net(dx,dy).d;
    dx=Dual(x,0.0); dy=Dual(y,1.0);
    double gy=net(dx,dy).d;
    return {gx,gy};
}

}
