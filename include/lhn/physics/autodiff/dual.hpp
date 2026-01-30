#pragma once
#include <cmath>

namespace lhn::physics::autodiff {

struct Dual {
    double v;
    double d;
    Dual(double v_=0.0,double d_=0.0):v(v_),d(d_){}
};

inline Dual operator+(const Dual&a,const Dual&b){return {a.v+b.v,a.d+b.d};}
inline Dual operator-(const Dual&a,const Dual&b){return {a.v-b.v,a.d-b.d};}
inline Dual operator*(const Dual&a,const Dual&b){return {a.v*b.v,a.v*b.d+a.d*b.v};}
inline Dual operator/(const Dual&a,const Dual&b){return {a.v/b.v,(a.d*b.v-a.v*b.d)/(b.v*b.v)};}
inline Dual sin(const Dual&a){return {std::sin(a.v),std::cos(a.v)*a.d};}

}
