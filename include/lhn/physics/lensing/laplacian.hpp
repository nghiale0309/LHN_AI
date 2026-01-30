#pragma once
#include "lhn/physics/nn/siren_mlp.hpp"

namespace lhn::physics::lensing {

using autodiff::Dual;

inline double laplacian(nn::SirenMLP&net,double x,double y){
    Dual dx(x,1.0),dy(y,0.0);
    Dual gx=net(dx,dy);
    double dxx=gx.d;

    dx=Dual(x,0.0); dy=Dual(y,1.0);
    Dual gy=net(dx,dy);
    double dyy=gy.d;

    return dxx+dyy;
}

}
