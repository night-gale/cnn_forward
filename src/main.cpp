#include "cnn_layers/conv.hpp"
#include "cnn_layers/pooling.hpp"
#include "cnn_layers/relu.hpp"
#include "cnn_layers/fully_connect.hpp"
#include "cnn_layers/softmax.hpp"
#include "num_recog.hpp"

#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;
using namespace cnn_forward;

int main() {
    vector<MatrixXf> input;
    for (int i = 0; i < 28; i++) {
        input.emplace_back(MatrixXf(28, 1));
        input[i] = MatrixXf::Ones(28, 1);
    }
    for(int i = 0; i < 28; i++) {
        for(int j = 0; j < 28; j++) {
            cin >> input[i](j, 0);
        }
    }
    vector<int> conv1_shape = {1, 28, 28, 1};
    NumRecog nr(conv1_shape);
    MatrixXf output;
    cout << nr.predict(output, input) << endl;
    cout << output << endl;
}

