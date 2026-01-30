#pragma once
#include "siren_layer_bp.hpp"

namespace lhn::physics::nn {

struct SirenMLPBP {
    std::vector<SirenLayerBP> layers;

    SirenMLPBP(){
        layers.emplace_back(2,64,30.0);
        layers.emplace_back(64,64,30.0);
        layers.emplace_back(64,1,30.0);
    }

    double forward(double x,double y){
        std::vector<double> v={x,y};
        for(auto&l:layers) v=l.forward(v);
        return v[0];
    }

    void backward(double grad,
                  std::vector<std::vector<double>>&gW,
                  std::vector<std::vector<double>>&gB){
        std::vector<double> g={grad};
        for(int k=layers.size()-1;k>=0;k--){
            g=layers[k].backward(g,gW[k],gB[k]);
        }
    }
};

}
