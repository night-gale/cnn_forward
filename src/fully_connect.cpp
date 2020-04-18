//
// Created by night-gale on 2020/3/21.
//

#include "cnn_layers/fully_connect.hpp"
#include <iostream>

using namespace cnn_forward;

FullyConnect::FullyConnect(vector<int>& shape, int output_num) {
    m_input_shape = vector<int>(shape);
    m_output_num = output_num;
    m_output_shape = {output_num};
}

void FullyConnect::forward(MatrixXf &output, vector<MatrixXf> &x) {
    MatrixXf fc_in(1, x.size()*x[0].cols()*x[0].rows());
    std::cout << x[0].cols() << endl;
    std::cout << x[0].rows() << endl;
    for(int i = 0; i < x.size(); i++) {
        MatrixXf temp(x[i]);
        temp.resize(1, m_input_shape[2]*m_input_shape[3]);
        fc_in.block(0, i*m_input_shape[2]*m_input_shape[3], 1, m_input_shape[2]*m_input_shape[3]) << temp;
    }
    output = fc_in * m_weights + m_bias;
}

void FullyConnect::loadWeights(const string &path) {
    int input_len = 1;
    for(int i = 1; i < m_input_shape.size(); i++) {
        input_len *= m_input_shape[i];
    }
    m_weights = MatrixXf::Ones(input_len, m_output_num);
    m_bias = MatrixXf::Ones(1, m_output_num);
}