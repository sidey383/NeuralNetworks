#pragma once
#include "MatrixTypes.h"

namespace TPNN {

    class ConvolutionalLayer {
    public:
        ConvolutionalLayer();
        ConvolutionalLayer(size_t size, size_t channel, Dimension<3> inputSize, bool hasPadding);
        void resize(size_t size, size_t channel, Dimension<3> inputSize, bool hasPadding);
        void apply(const Tensor3<float>& input, Tensor3<float>& output);
        Dimension<3> inputDimension();
        Dimension<3> outputDimensions();


    private:
        Tensor4<float> kernel;
        Tensor4<float> bias;
        bool hasPadding;
    };

}