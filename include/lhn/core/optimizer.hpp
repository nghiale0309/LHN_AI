#pragma once
#include <vector>
#include <cmath>

namespace lhn::core {

struct Adam {
    double lr;
    double b1;
    double b2;
    double eps;
    int t;

    std::vector<double> m;
    std::vector<double> v;

    Adam(double lr_=1e-4,double b1_=0.9,double b2_=0.999,double eps_=1e-8)
        :lr(lr_),b1(b1_),b2(b2_),eps(eps_),t(0){}

    void init(int n){
        m.assign(n,0.0);
        v.assign(n,0.0);
    }

    void step(std::vector<double>&w,const std::vector<double>&g){
        if(m.empty()) init(w.size());
        t++;
        for(size_t i=0;i<w.size();i++){
            m[i]=b1*m[i]+(1.0-b1)*g[i];
            v[i]=b2*v[i]+(1.0-b2)*g[i]*g[i];
            double mh=m[i]/(1.0-std::pow(b1,t));
            double vh=v[i]/(1.0-std::pow(b2,t));
            w[i]-=lr*mh/(std::sqrt(vh)+eps);
        }
    }
};

}
