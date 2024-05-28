#include "ConvolutionalLayer.h"
#include <random>

static std::default_random_engine re;


TPNN::ConvolutionalLayer::ConvolutionalLayer() = default;

TPNN::ConvolutionalLayer::ConvolutionalLayer(size_t size, size_t channel, Dimension<3> inputSize, bool hasPadding)
        : hasPadding(hasPadding) {
    bias = Tensor4<float>(inputSize);
    std::uniform_real_distribution<float> kernel_distribution(0, 0.5f / (float) (size * size));
    std::uniform_real_distribution<float> bias_distribution(0, 0.5f / (float) (size * size));
    kernel = Tensor4<float>(Dimension<4>{
                                    size * 2 + 1, // x size of kernel
                                    size * 2 + 1, // y size of kernel
                                    inputSize[2],
                                    channel
                            }
    )
    for (size_t i = 0; i < kernel.getDimension().total(); i++)
        kernel.getValues()[i] = kernel_distribution(re);
    bias.resize(outputSize);
}

void TPNN::ConvolutionalLayer::resize(size_t size, Dimension outputSize) {
    std::uniform_real_distribution<float> kernel_distribution(0, 0.5f / (float) (size * size));
    kernel = Tensor3<float>({size, size, size});
    for (size_t i = 0; i < kernel.getDimension().total(); i++)
        kernel.getValues()[i] = kernel_distribution(re);
}


void TPNN::ConvolutionalLayer::apply(const Tensor3<float> &input, Tensor3<float> &output) {
    size_t xs = input.getDimension().x - kernel.getDimension().x - 1;
    size_t ys = input.getDimension().y - kernel.getDimension().y - 1;
    size_t zs = input.getDimension().z;
    for (size_t z = 0; z < kernel.getDimension().z; z++) {
        for (size_t y = 0; y < ys; y++) {
            for (size_t x = 0; x < xs; x++) {
                float sum = bias.get(x, y, z);
                for (size_t iz = 0; iz < input.getDimension().z; iz++) {
                    for (size_t dy = 0; dy < kernel.getDimension().y; dy++) {
                        for (size_t dx = 0; dx < kernel.getDimension().x; dx++) {
                            sum += input.get(x + dx, y + dy, iz) * kernel.get(dx, dy, z);
                        }
                    }
                }
                output.set(x, y, z, sum);
            }
        }
    }
}

TPNN::Dimension TPNN::ConvolutionalLayer::inputDimension() {
    return TPNN::Dimension();
}

TPNN::Dimension TPNN::ConvolutionalLayer::outputDimensions() {
    return TPNN::Dimension();
}
