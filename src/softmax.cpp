//
// Created by night-gale on 2020/4/4.
//

#include "cnn_layers/softmax.hpp"

using namespace cnn_forward;
using namespace std;
using namespace Eigen;

Softmax::Softmax(vector<int> &shape) {
    m_shape = vector<int>(shape);
}

void Softmax::forward(MatrixXf &result, MatrixXf &x) {
    MatrixXf exp_pre = MatrixXf::Zero(1, m_shape[0]);
    result = MatrixXf::Zero(1, m_shape[0]);
    x = x - x.maxCoeff() * MatrixXf::Ones(x.rows(), x.cols());
    exp_pre << x.array().exp();
    result = exp_pre/exp_pre.sum();
}

Softmax::Softmax() {}
