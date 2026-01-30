#pragma once
#include <random>
#include <cmath>

namespace lhn::physics::sampling {

constexpr double PI = 3.14159265358979323846;

struct LensingSampler {
    double r_core, r_E, dr, r_far;
    std::mt19937 rng;
    std::uniform_real_distribution<double> uni;

    LensingSampler()
        : r_core(0.2), r_E(0.6), dr(0.05), r_far(1.2),
          uni(-1.0, 1.0) {}

    void sample(double& x, double& y, int region) {
        double r, t;
        if (region == 0) {
            r = std::sqrt(std::abs(uni(rng))) * r_core;
        } else if (region == 1) {
            r = r_E + dr * uni(rng);
        } else {
            r = r_far + std::abs(uni(rng));
        }
        t = 2.0 * PI * uni(rng);   
        x = r * std::cos(t);
        y = r * std::sin(t);
    }
};

}
