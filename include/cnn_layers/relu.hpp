#ifndef CNN_LAYERS_RELU_HPP_
#define CNN_LAYERS_RELU_HPP_

#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
namespace cnn_forward {
    class Relu {
    private:
        vector<int> m_shape;
    public:
        Relu(const vector<int>& shape);

        void forward(vector<MatrixXf>& output, vector<MatrixXf> x);

        vector<int>& getOutputShape() {
            return m_shape;
        }
    };

}

#endif

