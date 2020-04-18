#ifndef CNN_LAYERS_SOFTMAX_HPP_
#define CNN_LAYERS_SOFTMAX_HPP_

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>

namespace cnn_forward {
    class Softmax{
    private:
        std::vector<int> m_shape;

    public:
        Softmax(std::vector<int>& shape);

        void forward(Eigen::MatrixXf &result, Eigen::MatrixXf &x);
    };
}

#endif