#pragma once
#include "siren_layer.hpp"

namespace lhn::physics::nn {

struct SirenMLP {
    std::vector<SirenLayer> layers;

    SirenMLP(){
        layers.emplace_back(2,64,30.0);
        layers.emplace_back(64,64,30.0);
        layers.emplace_back(64,1,30.0);
    }

    autodiff::Dual operator()(const autodiff::Dual&x,const autodiff::Dual&y){
        std::vector<autodiff::Dual> v={x,y};
        for(auto&l:layers) v=l.forward(v);
        return v[0];
    }
};

}
