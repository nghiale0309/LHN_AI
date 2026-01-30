#pragma once
#include <vector>
#include <cmath>

namespace lhn::physics::nn {

struct SirenLayerBP {
    int in,out;
    double w0;
    std::vector<double> W,B;
    std::vector<double> x_cache;
    std::vector<double> z_cache;

    SirenLayerBP(int i,int o,double w):in(i),out(o),w0(w),W(i*o),B(o){
        for(auto&v:W) v=((double)rand()/RAND_MAX-0.5);
        for(auto&v:B) v=((double)rand()/RAND_MAX-0.5);
    }

    std::vector<double> forward(const std::vector<double>&x){
        x_cache=x;
        z_cache.assign(out,0.0);
        std::vector<double> y(out);
        for(int j=0;j<out;j++){
            double s=B[j];
            for(int i=0;i<in;i++) s+=W[j*in+i]*x[i];
            z_cache[j]=w0*s;
            y[j]=std::sin(z_cache[j]);
        }
        return y;
    }

    std::vector<double> backward(const std::vector<double>&grad_out,
                                 std::vector<double>&gW,
                                 std::vector<double>&gB){
        std::vector<double> grad_in(in,0.0);
        for(int j=0;j<out;j++){
            double dz=grad_out[j]*std::cos(z_cache[j])*w0;
            gB[j]+=dz;
            for(int i=0;i<in;i++){
                gW[j*in+i]+=dz*x_cache[i];
                grad_in[i]+=dz*W[j*in+i];
            }
        }
        return grad_in;
    }
};

}