#pragma once
#include <vector>
#include <random>
#include "lhn/physics/autodiff/dual.hpp"

namespace lhn::physics::nn {

using autodiff::Dual;

struct SirenLayer {
    int in,out;
    double w0;
    std::vector<double> W,B;

    SirenLayer(int i,int o,double w):in(i),out(o),w0(w),W(i*o),B(o){
        std::mt19937 g(42);
        std::uniform_real_distribution<double>d(-1.0,1.0);
        for(auto&x:W)x=d(g);
        for(auto&x:B)x=d(g);
    }

    std::vector<Dual> forward(const std::vector<Dual>&x){
        std::vector<Dual> y(out);
        for(int j=0;j<out;j++){
            Dual s(B[j],0.0);
            for(int i=0;i<in;i++) s=s+Dual(W[j*in+i],0.0)*x[i];
            y[j]=autodiff::sin(Dual(w0,0.0)*s);
        }
        return y;
    }
};

}
