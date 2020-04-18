#ifndef CNN_LAYERS_CONV_HPP_
#define CNN_LAYERS_CONV_HPP_

#include <Eigen/Dense>
#include <vector>
using namespace std;
using namespace Eigen;

namespace cnn_forward {
    class Conv2d {
    private:
        vector<int> m_input_shape;
        vector<int> m_output_shape;
        int m_output_channels;
        int m_input_channels;
        int m_batch_size;
        int m_stride;
        int m_k_size;
        string m_method;
        MatrixXf m_weights;
        MatrixXf m_bias;

    public:
        Conv2d(const vector<int>& shape, int output_channels, int ksize = 3, int stride = 1, const string & method = "VALID");

        void forward(vector<MatrixXf>& output,  vector<MatrixXf>& x);

        void load_weights(const string & path);

        vector<int>& getOutputShape() {
            return m_output_shape;
        }
    };
    bool im2col(MatrixXf& output, vector<MatrixXf>& x, int ksize, int stride);
}

#endif