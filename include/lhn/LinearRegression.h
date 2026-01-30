#pragma once
#include <Eigen/Dense>
class LinearRegression {
public:
    LinearRegression();
        void fit(const Eigen::VectorXd& X, const Eigen::VectorXd& y);
    
    Eigen::VectorXd predict(const Eigen::VectorXd& X);
    
    double get_weight() const;
    double get_bias() const;

private:
    Eigen::VectorXd m_coeffs; 
};