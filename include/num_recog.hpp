//
// Created by night-gale on 2020/4/19.
//

#ifndef NUM_RECOG_HPP_
#define NUM_RECOG_HPP_

#include "cnn_layers/conv.hpp"
#include "cnn_layers/fully_connect.hpp"
#include "cnn_layers/pooling.hpp"
#include "cnn_layers/relu.hpp"
#include "cnn_layers/softmax.hpp"

#include <vector>

using namespace cnn_forward;
using namespace std;

class NumRecog {
private:
    Conv2d conv1;
    Relu relu1;
    MaxPooling pool1;
    Conv2d conv2;
    Relu relu2;
    MaxPooling pool2;
    FullyConnect fc;
    Softmax sm;
    vector<MatrixXf> conv1_out;
    vector<MatrixXf> relu1_out;
    vector<MatrixXf> pool1_out;
    vector<MatrixXf> conv2_out;
    vector<MatrixXf> relu2_out;
    vector<MatrixXf> pool2_out;
    MatrixXf fc_out;
    const string k_conv1_weights_path = "../weights/conv1.weights";
    const string k_conv1_bias_path = "../weights/conv1.bias";
    const string k_conv2_weights_path = "../weights/conv2.weights";
    const string k_conv2_bias_path = "../weights/conv2.bias";
    const string k_fc_weights_path = "../weights/fc.weights";
    const string k_fc_bias_path = "../weights/fc.bias";

public:
    NumRecog(const vector<int> &input_shape, int k1_size=5, int k1_num=12,
            int s1=1, int k2_size=3, int k2_num=24, int s2=1, int out_channels=10) {
        conv1 = Conv2d(input_shape, k1_num, k1_size, s1);
        relu1 = Relu(conv1.getOutputShape());
        pool1 = MaxPooling(relu1.getOutputShape());
        conv2 = Conv2d(pool1.getOutputShape(), k2_num, k2_size, s2);
        relu2 = Relu(conv2.getOutputShape());
        pool2 = MaxPooling(relu2.getOutputShape());
        fc = FullyConnect(pool2.getOutputShape(), out_channels);
        sm = Softmax(fc.getOutputShape());

        conv1.load_weights(k_conv1_weights_path, k_conv1_bias_path);
        conv2.load_weights(k_conv2_weights_path, k_conv2_bias_path);
        fc.loadWeights(k_fc_weights_path, k_fc_bias_path);
    }

    int predict(MatrixXf& output, vector<MatrixXf>& x) {
        conv1.forward(conv1_out, x);
        relu1.forward(relu1_out, conv1_out);
        pool1.forward(pool1_out, relu1_out);
        conv2.forward(conv2_out, pool1_out);
        relu2.forward(relu2_out, conv2_out);
        pool2.forward(pool2_out, relu2_out);
        fc.forward(fc_out, pool2_out);
        sm.forward(output, fc_out);
        float _max = output.maxCoeff();
//        int ii;
        for(int i = 0; i < output.cols(); i++) {
            if(_max == output(0, i)) {
                return i;
            }
        }
        return -1;
    }
};
#endif