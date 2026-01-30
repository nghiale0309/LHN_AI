#pragma once
#include <lhn/physics/lensing/loss.hpp>
#include <lhn/physics/lensing/poisson.hpp>

namespace lhn::physics::lensing {

struct FullLoss {
    nn::SirenMLP& net;
    double lambda_lens;
    double lambda_poisson;

    FullLoss(nn::SirenMLP&n,double l1,double l2)
        :net(n),lambda_lens(l1),lambda_poisson(l2){}

    double operator()(double x,double y,double kappa){
        double Ll=lens_loss(net,x,y);
        double Lp=PoissonLoss(net)(x,y,kappa);
        return lambda_lens*Ll+lambda_poisson*Lp;
    }
};

}
