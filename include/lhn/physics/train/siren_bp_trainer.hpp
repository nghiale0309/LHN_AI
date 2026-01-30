#pragma once
#include <lhn/core/optimizer.hpp>
#include <lhn/physics/nn/siren_mlp_bp.hpp>

namespace lhn::physics::train {

struct SirenBPTrainer {
    nn::SirenMLPBP& net;
    core::Adam& opt;
    std::vector<std::vector<double>> gW;
    std::vector<std::vector<double>> gB;

    SirenBPTrainer(nn::SirenMLPBP&n,core::Adam&o):net(n),opt(o){
        for(auto&l:net.layers){
            gW.emplace_back(l.W.size());
            gB.emplace_back(l.B.size());
        }
    }

    double step(){
        for(auto&v:gW) for(auto&x:v) x=0.0;
        for(auto&v:gB) for(auto&x:v) x=0.0;

        double x=((double)rand()/RAND_MAX-0.5);
        double y=((double)rand()/RAND_MAX-0.5);

        double psi=net.forward(x,y);
        double L=psi*psi;
        net.backward(2.0*psi,gW,gB);

        std::vector<double> w,gw;
        for(size_t k=0;k<net.layers.size();k++){
            for(auto&v:net.layers[k].W){ w.push_back(v); }
            for(auto&v:net.layers[k].B){ w.push_back(v); }
            for(auto&v:gW[k]) gw.push_back(v);
            for(auto&v:gB[k]) gw.push_back(v);
        }

        opt.step(w,gw);

        int idx=0;
        for(auto&l:net.layers){
            for(auto&v:l.W) v=w[idx++];
            for(auto&v:l.B) v=w[idx++];
        }

        return L;
    }
};

}
