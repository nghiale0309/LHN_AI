#pragma once
#include <vector>
#include <cstdlib>
#include <lhn/core/optimizer.hpp>
#include <lhn/physics/lensing/full_loss.hpp>

namespace lhn::physics::train {

struct PINNTrainer {
    nn::SirenMLP& net;
    lensing::FullLoss& loss;
    core::Adam& opt;
    std::vector<double*> params;
    std::vector<double> grads;

    PINNTrainer(nn::SirenMLP&n,
                lensing::FullLoss&l,
                core::Adam&o)
        :net(n),loss(l),opt(o){
        collect_params();
        grads.resize(params.size());
    }

    void collect_params(){
        for(auto&layer:net.layers){
            for(auto&v:layer.W) params.push_back(&v);
            for(auto&v:layer.B) params.push_back(&v);
        }
    }

    double step(int batch){
        for(auto&g:grads) g=0.0;
        double L=0.0;

        for(int i=0;i<batch;i++){
            double x=((double)rand()/RAND_MAX-0.5);
            double y=((double)rand()/RAND_MAX-0.5);
            double kappa=0.5;

            double base=loss(x,y,kappa);
            L+=base;

            for(size_t p=0;p<params.size();p++){
                double eps=1e-6;
                double old=*params[p];

                *params[p]=old+eps;
                double Lp=loss(x,y,kappa);

                *params[p]=old-eps;
                double Lm=loss(x,y,kappa);

                *params[p]=old;
                grads[p]+=(Lp-Lm)/(2.0*eps);
            }
        }

        for(auto&g:grads) g/=batch;

        std::vector<double> w(params.size());
        for(size_t i=0;i<params.size();i++) w[i]=*params[i];

        opt.step(w,grads);

        for(size_t i=0;i<params.size();i++) *params[i]=w[i];

        return L/batch;
    }
};

}
