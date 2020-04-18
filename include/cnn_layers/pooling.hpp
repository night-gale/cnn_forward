#ifndef CNN_LAYERS_POOLING_HPP_
#define CNN_LAYERS_POOLING_HPP_

#include <Eigen/Dense>
#include <vector>

using namespace std;
using namespace Eigen;

namespace cnn_forward {
    class MaxPooling {
    private:
        vector<int> m_input_shape;
        int m_k_size;
        int m_stride;
        int m_output_channels;
        vector<int> m_output_shape;

    public:
        MaxPooling(vector<int>& shape, int ksize=2, int stride=2);

        void forward(vector<MatrixXf> &output, vector<MatrixXf> &x);

        vector<int>& getOutputShape() {
            return m_output_shape;
        }
    };
}
#endif