//
// Created by night-gale on 2020/3/21.
//

#include "cnn_layers/relu.hpp"

using namespace cnn_forward;

Relu::Relu(const vector<int>& shape) {
    this->m_shape = vector<int>(shape);
}

void Relu::forward(vector<MatrixXf> &output, vector<MatrixXf> x) {
    for(int i = 0; i < x.size(); i++) {
        output.emplace_back(MatrixXf(x[i].rows(),x[i].cols()));
        output[i] << x[i].array().max(MatrixXf::Zero(x[i].rows(), x[i].cols()).array());
    }
}

Relu::Relu() {}
