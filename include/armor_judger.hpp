//
// Created by night-gale on 2020/4/19.
//

#ifndef ARMOR_JUDGER_HPP_
#define ARMOR_JUDGER_HPP_

#include "cnn_layers/conv.hpp"
#include "cnn_layers/fully_connect.hpp"
#include "cnn_layers/pooling.hpp"
#include "cnn_layers/relu.hpp"
#include "cnn_layers/softmax.hpp"

#include <vector>

using namespace cnn_forward;
using namespace std;

class ArmorJudger {
private:
    Conv2d conv1;
    Relu relu1;
    MaxPooling pool1;
    Conv2d conv2;
    Relu relu2;
    MaxPooling pool2;

};
#endif ARMOR_JUDGER_HPP_