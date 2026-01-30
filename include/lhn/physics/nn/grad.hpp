#pragma once

namespace lhn::physics::nn {

struct Grad {
    double dv,ddx,ddy,dlap;
};

}
