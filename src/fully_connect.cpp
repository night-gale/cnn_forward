//
// Created by night-gale on 2020/3/21.
//

#include "cnn_layers/fully_connect.hpp"

#include <iostream>
#include <fstream>

using namespace cnn_forward;

FullyConnect::FullyConnect(vector<int>& shape, int output_num) {
    m_input_shape = vector<int>(shape);
    m_output_num = output_num;
    m_output_shape = {output_num};
}

void FullyConnect::forward(MatrixXf &output, vector<MatrixXf> &x) {
    MatrixXf fc_in(1, x.size()*x[0].cols()*x[0].rows());

    for(int i = 0; i < x.size(); i++) {
        MatrixXf temp(x[i]);
        temp.transposeInPlace();
        temp.resize(1, m_input_shape[2]*m_input_shape[3]);
        fc_in.block(0, i*m_input_shape[2]*m_input_shape[3],
                1, m_input_shape[2]*m_input_shape[3]) << temp;
    }
    output = fc_in * m_weights + m_bias;
}

void FullyConnect::loadWeights(const string &path_weights, const string &path_bias) {
    std::fstream fin;
    fin.open(path_weights);
    int input_len = 1;
    for(int i = 1; i < m_input_shape.size(); i++) {
        input_len *= m_input_shape[i];
    }
    m_weights = MatrixXf::Ones(input_len, m_output_num);
    for(int i = 0; i < m_weights.rows(); i++) {
        for(int j = 0; j < m_weights.cols(); j++) {
            fin >> m_weights(i, j);
        }
    }

    fin.close();
    fin.open(path_bias);
    m_bias = MatrixXf::Ones(1, m_output_num);
    for(int i = 0; i < m_bias.cols(); i++) {
        fin >> m_bias(0, i);
    }
    fin.close();
}

FullyConnect::FullyConnect() {}
