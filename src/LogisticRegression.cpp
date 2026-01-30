#include "lhn/LogisticRegression.h"
#include <cmath>

LogisticRegression::LogisticRegression(double lr_, int max_iter_)
    : lr(lr_), max_iter(max_iter_), b(0.0) {}

void LogisticRegression::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    int n = X.rows();
    int d = X.cols();

    w = Eigen::VectorXd::Zero(d);
    b = 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        Eigen::VectorXd z = X * w;
        z.array() += b;

        Eigen::VectorXd p = z.unaryExpr([this](double v) {
            return sigmoid(v);
        });

        Eigen::VectorXd diff = p - y;

        Eigen::VectorXd grad_w = (X.transpose() * diff) / n;
        double grad_b = diff.mean();

        w -= lr * grad_w;
        b -= lr * grad_b;
    }
}

Eigen::VectorXd LogisticRegression::predict_proba(const Eigen::MatrixXd& X) const {
    Eigen::VectorXd z = X * w;
    z.array() += b;

    return z.unaryExpr([this](double v) {
        return sigmoid(v);
    });
}

Eigen::VectorXi LogisticRegression::predict(const Eigen::MatrixXd& X) const {
    Eigen::VectorXd p = predict_proba(X);
    Eigen::VectorXi y = (p.array() >= 0.5).cast<int>();
    return y;
}

Eigen::VectorXd LogisticRegression::get_weight() const {
    return w;
}

double LogisticRegression::get_bias() const {
    return b;
}
