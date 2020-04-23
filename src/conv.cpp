//
// Created by night-gale on 2020/3/20.
//

#include "cnn_layers/conv.hpp"
#include <iostream>
#include <fstream>

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

    MatrixXf conv_out = img_col * m_weights;

    for(int i = 0; i < conv_out.cols(); i++) {
        conv_out.block(0, i, conv_out.rows(), 1) <<
                conv_out.block(0, i, conv_out.rows(), 1) + m_bias(0, i)*MatrixXf::Ones(conv_out.rows(), 1);
    }


    for(int i = 0; i < m_output_shape[1]; i++) {
        output.emplace_back(
                MatrixXf(m_output_shape[2], m_output_shape[3]));
        for(int j = 0; j < m_output_shape[2]; j++) {
            output[i].row(j) << conv_out.row(i * m_output_shape[2] + j);
        }

    }

}

void Conv2d::load_weights(const string &path_weights, const string&path_bias) {
    m_bias = MatrixXf::Zero(1, m_output_channels);
    m_weights = MatrixXf::Ones(m_k_size*m_k_size*m_input_channels, m_output_channels);
    fstream fin;
    fin.open(path_weights);

//    string first_line;
//    getline(fin, first_line);
    for(int i = 0; i < m_weights.rows(); i++) {
        for(int j = 0; j < m_weights.cols(); j++) {
            fin >> m_weights(i, j);
        }
    }

    fin.close();
    fin.open(path_bias);
    for(int i = 0; i < m_bias.cols(); i++) {
        fin >> m_bias(0, i);
    }
    fin.close();
}

Conv2d::Conv2d() {}

bool cnn_forward::im2col(MatrixXf& output, vector<MatrixXf>& x, int ksize, int stride) {
    output = MatrixXf(((x.size() - ksize + 1)/stride)*((x[0].rows() - ksize + 1)/stride), ksize * ksize * x[0].cols());
    for(int i = 0; i < (x.size() - ksize + 1)/stride; i++) {
        // i for row selection
        for(int j = 0; j < (x[i].rows() - ksize + 1)/stride; j++) {
            // j for col selection
            for(int a = 0; a < ksize; a++) {
                // x[a, j:j+ksize, :]
                MatrixXf temp = x[i+a].block(j, 0, ksize, x[i].cols());
                temp.transposeInPlace();
                temp.resize(1, temp.cols() * temp.rows());
                output.block((i*(x.size()-ksize+1)+j), a * ksize * x[i].cols(), 1, ksize*x[i].cols()) << temp;
            }
        }
    }
    return true;
}
