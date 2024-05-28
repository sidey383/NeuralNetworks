#include <random>
#include <iostream>
#include "DenseLayer.h"

static std::default_random_engine re;

TPNN::DenseLayer::DenseLayer() = default;

TPNN::DenseLayer::DenseLayer(size_t input, size_t output) {
    std::uniform_real_distribution<float> weights_distribution(0, 1.0f / (float) input);
    std::uniform_real_distribution<float> bias_distribution(0, 0.5f / (float) output);
    _bias.resize(output);
    _weights.resize(output);
    for (size_t i = 0; i < output; i++) {
        _bias[i] = bias_distribution(re);
        _weights[i].resize(input);
        for (size_t j = 0; j < input; j++) {
            _weights[i][j] = weights_distribution(re);
        }
    }
}

void TPNN::DenseLayer::resize(size_t input, size_t output) {
    std::uniform_real_distribution<float> weights_distribution(0, 1.0f / (float) input);
    std::uniform_real_distribution<float> bias_distribution(0, 0.5f / (float) output);
    _bias.resize(output);
    _weights.resize(output);
    for (size_t i = 0; i < output; i++) {
        _bias[i] = bias_distribution(re);
        _weights[i].resize(input);
        for (size_t j = 0; j < input; j++) {
            _weights[i][j] = weights_distribution(re);
        }
    }
}

TPNN::DenseLayer::~DenseLayer() = default;

size_t TPNN::DenseLayer::inputSize() const {
    return _weights.getDimensions()[0];
}

size_t TPNN::DenseLayer::outputSize() const {
    return _weights.getDimensions()[1];
}

void TPNN::DenseLayer::apply(const Tensor<1, float> &input, Tensor<1, float> &output) {
    if (input.size() != _weights.getDimensions()[0])
        throw std::invalid_argument("Wrong input size");
    if (output.size() != _weights.getDimensions()[1])
        throw std::invalid_argument("Wrong output size");
    for (size_t o = 0; o < _weights.getDimensions()[1]; o++) {
        output[o] = _bias[o];
        for (size_t i = 0; i < _weights.getDimensions()[0]; i++) {
            output[o] += _weights.val(Pose<2>{i, o}) * input[i];
        }
    }
}

void TPNN::DenseLayer::applyError(const Tensor<1, float> &error,
                                  std::vector<float> &propagatedError, float weight) {
    size_t inputSize = this->inputSize();
    size_t outputSize = this->outputSize();
    for (size_t i = 0; i < outputSize; i++) {
        _bias[i] -= error[i] * weight;
    }
    for (size_t i = 0; i < inputSize; i++) {
        propagatedError[i] = 0;
        for (size_t j = 0; j < outputSize; j++) {
            propagatedError[i] += _weights[j][i] * error[j];
        }
    }
    for (size_t i = 0; i < outputSize; i++) {
        for (size_t j = 0; j < inputSize; j++) {
            _weights[i][j] -= error[i] * input[j] * weight;
        }
    }
}

void TPNN::DenseLayer::debugPrint() {
    for (size_t i = 0; i < _weights.size(); i++) {
        std::cout << "[" << i << "]=";
        bool isFirst = true;
        for (size_t j = 0; j < _weights[i].size(); j++) {
            if (isFirst) {
                std::cout << _weights[i][j]
                          << "*[" << j << "]";
            } else {
                std::cout << (_weights[i][j] < 0 ? "-" : "+") << (_weights[i][j] < 0 ? -_weights[i][j] : _weights[i][j])
                          << "*[" << j << "]";
            }
            isFirst = false;
        }
        std::cout << (_bias[i] < 0 ? "-" : "+") << (_bias[i] < 0 ? -_bias[i] : _bias[i]);
        std::cout << std::endl;
    }

}

void TPNN::DenseLayer::save(std::ostream &s) const {
    size_t size = inputSize();
    s.write((char*) &size, sizeof(size_t));
    size = outputSize();
    s.write((char*) &size, sizeof(size_t));
    for (size_t i = 0; i < outputSize(); i++) {
        s.write((char*)_weights[i].data(), (std::streamsize) (sizeof(float)*inputSize()));
    }
    s.write((char*)_bias.data(),  (std::streamsize)(sizeof(float)*outputSize()));
}

TPNN::DenseLayer::DenseLayer(std::istream &s) {
    size_t inputSize;
    size_t outputSize;
    s.read((char*)&inputSize, sizeof(size_t));
    s.read((char*)&outputSize, sizeof(size_t));
    _weights.resize(outputSize);
    for (size_t i = 0; i < outputSize; i++) {
        _weights[i].resize(inputSize);
        s.read((char*)_weights[i].data(), (std::streamsize)(sizeof(float)*inputSize));
    }
    _bias.resize(outputSize);
    s.read((char*)_bias.data(),  (std::streamsize)(sizeof(float)*outputSize));
}


