#pragma once
#include <Eigen/Dense>

class LogisticRegression {
private:
    Eigen::VectorXd w;
    double b;
    double lr;
    int max_iter;

    inline double sigmoid(double x) const {
        return 1.0 / (1.0 + std::exp(-x));
    }

public:
    LogisticRegression(double lr_=0.1, int max_iter_=1000);

    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);

    Eigen::VectorXd predict_proba(const Eigen::MatrixXd& X) const;

    Eigen::VectorXi predict(const Eigen::MatrixXd& X) const;

    Eigen::VectorXd get_weight() const;
    double get_bias() const;
};
