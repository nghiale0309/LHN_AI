#include "lhn/LinearRegression.h"
#include <iostream>

LinearRegression::LinearRegression() {}

void LinearRegression::fit(const Eigen::VectorXd& X, const Eigen::VectorXd& y) {
    long n = X.size();
    double sum_x = X.sum();           
    double sum_xx = X.dot(X);         
    double sum_y = y.sum();         
    double sum_xy = X.dot(y);        

    Eigen::Matrix2d A;
    A << n,      sum_x,
         sum_x,  sum_xx;
         
    Eigen::Vector2d b;
    b << sum_y, sum_xy;

    m_coeffs = A.ldlt().solve(b);
}

Eigen::VectorXd LinearRegression::predict(const Eigen::VectorXd& X) {
    double bias = m_coeffs(0);
    double weight = m_coeffs(1);
    return (X * weight).array() + bias;
}

double LinearRegression::get_weight() const { return (m_coeffs.size() > 1) ? m_coeffs(1) : 0.0; }
double LinearRegression::get_bias() const { return (m_coeffs.size() > 0) ? m_coeffs(0) : 0.0; }