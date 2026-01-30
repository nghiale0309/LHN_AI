#pragma once
#include <immintrin.h>
#include <vector>
#include <cmath>
#include <lhn/physics/nn/node.hpp>

namespace lhn::physics::nn {

struct SirenLayerPhysicsAVX {
    int in,out;
    double w0;
    std::vector<double> W,B;
    std::vector<Node> in_cache;
    std::vector<double> z_cache,dx_cache,dy_cache;

    SirenLayerPhysicsAVX(int i,int o,double w)
        :in(i),out(o),w0(w),
         W(i*o),B(o),
         z_cache(o),dx_cache(o),dy_cache(o){}

    std::vector<Node> forward(const std::vector<Node>&x){
        in_cache=x;
        std::vector<Node> y(out);

        __m256d vw0=_mm256_set1_pd(w0);

        for(int j=0;j<out;j++){
            __m256d vz=_mm256_setzero_pd();
            __m256d vdx=_mm256_setzero_pd();
            __m256d vdy=_mm256_setzero_pd();
            __m256d vlap=_mm256_setzero_pd();

            int i=0;
            for(;i+3<in;i+=4){
                __m256d vw=_mm256_loadu_pd(&W[j*in+i]);

                __m256d vv=_mm256_set_pd(
                    x[i+3].v,x[i+2].v,x[i+1].v,x[i].v);
                __m256d vdxi=_mm256_set_pd(
                    x[i+3].dx,x[i+2].dx,x[i+1].dx,x[i].dx);
                __m256d vdyi=_mm256_set_pd(
                    x[i+3].dy,x[i+2].dy,x[i+1].dy,x[i].dy);
                __m256d vlapi=_mm256_set_pd(
                    x[i+3].lap,x[i+2].lap,x[i+1].lap,x[i].lap);

                vz=_mm256_fmadd_pd(vw,vv,vz);
                vdx=_mm256_fmadd_pd(vw,vdxi,vdx);
                vdy=_mm256_fmadd_pd(vw,vdyi,vdy);
                vlap=_mm256_fmadd_pd(vw,vlapi,vlap);
            }

            double z=B[j],dx=0,dy=0,lap=0;
            double buf[4];

            _mm256_storeu_pd(buf,vz);   z+=buf[0]+buf[1]+buf[2]+buf[3];
            _mm256_storeu_pd(buf,vdx);  dx+=buf[0]+buf[1]+buf[2]+buf[3];
            _mm256_storeu_pd(buf,vdy);  dy+=buf[0]+buf[1]+buf[2]+buf[3];
            _mm256_storeu_pd(buf,vlap); lap+=buf[0]+buf[1]+buf[2]+buf[3];

            for(;i<in;i++){
                double w=W[j*in+i];
                z+=w*x[i].v;
                dx+=w*x[i].dx;
                dy+=w*x[i].dy;
                lap+=w*x[i].lap;
            }

            z*=w0; dx*=w0; dy*=w0; lap*=w0;

            z_cache[j]=z;
            dx_cache[j]=dx;
            dy_cache[j]=dy;

            double s=std::sin(z);
            double c=std::cos(z);

            y[j].v=s;
            y[j].dx=c*dx;
            y[j].dy=c*dy;
            y[j].lap=-s*(dx*dx+dy*dy)+c*lap;
        }
        return y;
    }