//
// Created by night-gale on 2020/3/20.
//

#include "cnn_layers/conv.hpp"
#include <iostream>

using namespace cnn_forward;

Conv2d::Conv2d(const vector<int>& shape, int output_channels, int ksize, int stride, const string &method) {
    m_input_shape = vector<int>(shape);
    m_input_channels = shape[3];
    m_output_channels = output_channels;
    m_batch_size = shape[0];
    m_stride = stride;
    m_k_size = ksize;
    m_method = method;
    if(m_method == "VALID") {
        m_output_shape = {1, (shape[1]-m_k_size+1)/m_stride, (shape[2]-m_k_size+1)/m_stride, m_output_channels};
    }else {
        m_output_shape = {1, shape[1]/m_stride, shape[2]/m_stride , output_channels};
    }
}

void Conv2d::forward(vector<MatrixXf>& output, vector<MatrixXf>& x) {
    // reshape the filters

    MatrixXf padded_x;
    if(m_method == "SAME") {
        // padding
    }
    MatrixXf img_col;
    im2col(img_col, x, m_k_size, m_stride);
    cout << img_col.cols() << endl;
    cout << m_weights.rows() << endl;
    MatrixXf conv_out = img_col * m_weights;


    for(int i = 0; i < m_output_shape[1]; i++) {
        output.emplace_back(
                MatrixXf(m_output_shape[2], m_output_shape[3]));
        for(int j = 0; j < m_output_shape[2]; j++) {
            output[i].row(j) << conv_out.row(i * m_output_shape[2] + j);
        }
    }
//    cout << output[1] << endl;
}

void Conv2d::load_weights(const string &path) {
    m_bias = MatrixXf::Ones(1, m_output_channels);
    m_weights = MatrixXf::Ones(m_k_size*m_k_size*m_input_channels, m_output_channels);
        // read weights file from path
        // load weights to m_weights;
}

bool cnn_forward::im2col(MatrixXf& output, vector<MatrixXf>& x, int ksize, int stride) {
    output = MatrixXf(((x.size() - ksize + 1)/stride)*((x[0].rows() - ksize + 1)/stride), ksize * ksize * x[0].cols());
    for(int i = 0; i < (x.size() - ksize + 1)/stride; i++) {
//        cout << x[i] << endl;
//        cout << x[i].rows() << endl;
//        cout << x[i].cols() << endl;
        for(int j = 0; j < (x[i].rows() - ksize + 1)/stride; j++) {
            for(int a = 0; a < ksize; a++) {
                MatrixXf temp = x[a].block(j, 0, ksize, x[i].cols());
                temp.resize(1, temp.cols() * temp.rows());
                output.block((i*(x.size()-ksize+1)+j), a * ksize * x[i].cols(), 1, ksize*x[i].cols()) << temp;
            }
        }
    }
    return true;
}