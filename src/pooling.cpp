//
// Created by night-gale on 2020/3/21.
//

#include "cnn_layers/pooling.hpp"

using namespace cnn_forward;

MaxPooling::MaxPooling(vector<int>& shape, int ksize, int stride) {
    m_input_shape = vector<int>(shape);
    m_k_size = ksize;
    m_stride = stride;
    m_output_channels = shape[3];
    m_output_shape = {shape[0], shape[1]/m_stride, shape[2]/m_stride, m_output_channels};
}

void MaxPooling::forward(vector<MatrixXf> &output, vector<MatrixXf> &x) {
    for(int i = 0; i < m_output_shape[1]; i++) {
        output.emplace_back(MatrixXf(m_output_shape[2], m_output_channels));
    }

    for(int c = 0; c < m_output_channels; c++) {
        for(int i = 0; i < m_output_shape[1]; i++) {
            for(int j = 0; j < m_output_shape[2];j++) {
                float _max = 0;
                for(int a = 0; a < m_k_size; a++) {
                    for(int b = 0; b < m_k_size; b++) {
                        _max = std::max(x[i * 2 + a].coeff(j*2+b, c), _max);
                    }
                }
                output[i].block(j,c, 1,1) << _max;
            }
        }
    }
}

MaxPooling::MaxPooling() {}
