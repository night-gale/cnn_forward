#include "cnn_layers/conv.hpp"
#include "cnn_layers/pooling.hpp"
#include "cnn_layers/relu.hpp"
#include "cnn_layers/fully_connect.hpp"
#include "cnn_layers/softmax.hpp"

#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;
using namespace cnn_forward;

int main() {
    vector<int> shape = {1, 4, 4, 3};
    vector<MatrixXf> input;
    for (int i = 0; i < 28; i++) {
        input.emplace_back(MatrixXf(28, 3));
        input[i] = MatrixXf::Ones(28, 3);
    }
    vector<int> conv1_shape = {1, 28, 28, 1};
    Conv2d conv1(conv1_shape, 12, 5, 1);
    conv1.load_weights("../weights/conv1.weights");
    Relu relu1(conv1.getOutputShape());
    MaxPooling pool1(relu1.getOutputShape());
    Conv2d conv2(pool1.getOutputShape(), 24, 3, 1);
    conv2.load_weights("../weights/conv2.weights");
    Relu relu2(conv2.getOutputShape());
    MaxPooling pool2(relu2.getOutputShape());
    FullyConnect fc(pool2.getOutputShape(), 2);
    fc.loadWeights("../weights/fc.weights");
    Softmax sm(fc.getOutputShape());
    vector<MatrixXf> conv1_out;
    vector<MatrixXf> relu1_out;
    vector<MatrixXf> pool1_out;
    vector<MatrixXf> conv2_out;
    vector<MatrixXf> relu2_out;
    vector<MatrixXf> pool2_out;
    MatrixXf fc_out;
    MatrixXf result;
    conv1.forward(conv1_out, input);
    relu1.forward(relu1_out, conv1_out);
    pool1.forward(pool1_out, relu1_out);
    conv2.forward(conv2_out, pool1_out);
    relu2.forward(relu2_out, conv2_out);
    pool2.forward(pool2_out, relu2_out);
    fc.forward(fc_out, pool2_out);
    sm.forward(result, fc_out);
    cout << result << endl;
}

