#pragma once
#include "gradient.hpp"

namespace lhn::physics::lensing {

inline double lens_loss(nn::SirenMLP&net,double x,double y){
    auto[gx,gy]=gradient(net,x,y);
    double bx=x-gx;
    double by=y-gy;
    return bx*bx+by*by;
}

}
