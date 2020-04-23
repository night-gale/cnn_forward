#ifndef CNN_LAYERS_FULLY_CONNECT_HPP_
#define CNN_LAYERS_FULLY_CONNECT_HPP_

#include <Eigen/Dense>

#include <vector>

using namespace std;
using namespace Eigen;

namespace cnn_forward {
    class FullyConnect {
    private:
        vector<int> m_input_shape;
        int m_output_num;
        MatrixXf m_weights;
        MatrixXf m_bias;
        vector<int> m_output_shape;

    public:
        FullyConnect(vector<int>& shape, int output_num);

        FullyConnect();

        void forward(MatrixXf& output, vector<MatrixXf>& x);
        void loadWeights(const string & path_weights, const string &path_bias);
        vector<int>& getOutputShape() {
            return m_output_shape;
        };
    };
}
#endif