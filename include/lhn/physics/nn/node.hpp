#pragma once

namespace lhn::physics::nn {

struct Node {
    double v, dx, dy, lap;
};

struct Grad {
    double dv, ddx, ddy, dlap;
};

}