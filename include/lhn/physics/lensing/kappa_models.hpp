#pragma once
#include <cmath>

namespace lhn::physics::lensing {

enum class KappaModelType {
    PointMass,
    SIS,
    NFW
};

inline double kappa_point_mass(double x, double y, double eps = 1e-6) {
    double r = std::sqrt(x * x + y * y) + eps;
    return 0.5 / r;
}

inline double kappa_sis(double x, double y, double eps = 1e-6) {
    double r = std::sqrt(x * x + y * y) + eps;
    return 0.5 / r;
}

inline double kappa_nfw(double x, double y,
                        double rs = 0.5,
                        double k0 = 1.0,
                        double eps = 1e-6) {
    double r = std::sqrt(x * x + y * y) + eps;
    double xrs = r / rs;

    if (xrs < 1.0) {
        return k0 * (1.0 - std::atanh(std::sqrt(1.0 - xrs * xrs)) /
                               std::sqrt(1.0 - xrs * xrs)) /
               (xrs * xrs - 1.0);
    } else if (xrs > 1.0) {
        return k0 * (1.0 - std::atan(std::sqrt(xrs * xrs - 1.0)) /
                               std::sqrt(xrs * xrs - 1.0)) /
               (xrs * xrs - 1.0);
    } else {
        return k0 / 3.0;
    }
}

struct KappaModel {
    KappaModelType type;

    double rs;
    double k0;

    KappaModel(KappaModelType t = KappaModelType::PointMass)
        : type(t), rs(0.5), k0(1.0) {}

    inline double operator()(double x, double y) const {
        switch (type) {
            case KappaModelType::PointMass:
                return kappa_point_mass(x, y);
            case KappaModelType::SIS:
                return kappa_sis(x, y);
            case KappaModelType::NFW:
                return kappa_nfw(x, y, rs, k0);
            default:
                return 0.0;
        }
    }
};

} // namespace lhn::physics::lensing
