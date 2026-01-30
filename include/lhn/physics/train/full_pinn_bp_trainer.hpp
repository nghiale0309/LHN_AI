#pragma once
#include <vector>
#include <cmath>
#include <cstdlib>
#include <lhn/core/optimizer.hpp>
#include <lhn/physics/nn/siren_mlp_bp.hpp>

namespace lhn::physics::train {

struct FullPINNBPTrainer {
    nn::SirenMLPBP& net;
    core::Adam& opt;
    double lambda_lens;
    double lambda_poisson;

    std::vector<std::vector<double>> gW;
    std::vector<std::vector<double>> gB;

    FullPINNBPTrainer(nn::SirenMLPBP&n,
                      core::Adam&o,
                      double l1,
                      double l2)
        :net(n),opt(o),lambda_lens(l1),lambda_poisson(l2){
        for(auto&l:net.layers){
            gW.emplace_back(l.W.size());
            gB.emplace_back(l.B.size());
        }
    }

    double psi(double x,double y){
        return net.forward(x,y);
    }

    double dpsi_dx(double x,double y){
        double h=1e-4;
        return (psi(x+h,y)-psi(x-h,y))/(2*h);
    }

    double dpsi_dy(double x,double y){
        double h=1e-4;
        return (psi(x,y+h)-psi(x,y-h))/(2*h);
    }

    double laplacian(double x,double y){
        double h=1e-4;
        return (psi(x+h,y)+psi(x-h,y)+psi(x,y+h)+psi(x,y-h)-4*psi(x,y))/(h*h);
    }

    double step(int batch){
        for(auto&v:gW) for(auto&x:v) x=0.0;
        for(auto&v:gB) for(auto&x:v) x=0.0;

        double L=0.0;

        for(int i=0;i<batch;i++){
            double x=((double)rand()/RAND_MAX-0.5);
            double y=((double)rand()/RAND_MAX-0.5);
            double kappa=0.5;

            double gx=dpsi_dx(x,y);
            double gy=dpsi_dy(x,y);
            double bx=x-gx;
            double by=y-gy;
            double Ll=bx*bx+by*by;

            double lap=laplacian(x,y);
            double r=lap-2.0*kappa;
            double Lp=r*r;

            double loss=lambda_lens*Ll+lambda_poisson*Lp;
            L+=loss;

            double grad_psi=0.0;

            grad_psi+=lambda_lens*(-2*bx*(dpsi_dx(x,y)-dpsi_dx(x+1e-4,y)));
            grad_psi+=lambda_lens*(-2*by*(dpsi_dy(x,y)-dpsi_dy(x,y+1e-4)));
            grad_psi+=lambda_poisson*2*r;

            net.forward(x,y);
            net.backward(grad_psi,gW,gB);
        }

        for(auto&v:gW) for(auto&x:v) x/=batch;
        for(auto&v:gB) for(auto&x:v) x/=batch;

        std::vector<double> w,gw;
        for(size_t k=0;k<net.layers.size();k++){
            for(auto&v:net.layers[k].W) w.push_back(v);
            for(auto&v:net.layers[k].B) w.push_back(v);
            for(auto&v:gW[k]) gw.push_back(v);
            for(auto&v:gB[k]) gw.push_back(v);
        }

        opt.step(w,gw);

        int idx=0;
        for(auto&l:net.layers){
            for(auto&v:l.W) v=w[idx++];
            for(auto&v:l.B) v=w[idx++];
        }

        return L/batch;
    }
};

}