#include <cstring>
#include <iostream>
#include <stdexcept>
#include "Perceptron.h"
#include "components/DenseLayer.h"

TPNN::Perceptron::Perceptron(size_t layerCount, size_t* layerSizes, ActivationFunction af)  {
    _layers = std::vector<NeuronLayer>(layerCount);
    for (size_t i = 0; i < layerCount; i++) {
        _layers[i].init(layerSizes[i], af);
    }
    _transformations.resize(layerCount - 1);
    for (size_t i = 0; i < layerCount - 1; i++) {
        _transformations[i].resize(layerSizes[i], layerSizes[i+1]);
    }
}

TPNN::Perceptron::~Perceptron() {}

const std::vector<float>& TPNN::Perceptron::calculate(const std::vector<float>& input) {
    if (_layers[0].input.size() != input.size())
        throw std::range_error("Wrong input size");
    _layers[0].input = input;
    _layers[0].calculateOutput();
    for (int i = 0; i < _transformations.size(); i++) {
        _transformations[i].apply(_layers[i].output, _layers[i+1].input);
        _layers[i+1].calculateOutput();
    }
    return _layers[_layers.size() - 1].input;
}

void TPNN::Perceptron::applyError(const std::vector<float>& error, float weight) {
    if (_layers[_layers.size() - 1].outputError.size() != error.size())
        throw std::range_error("Wrong error size");
    _layers[_layers.size() - 1].outputError = error;
    for (size_t i = 1; i <= _transformations.size(); i++) {
        NeuronLayer& prevLayer = _layers[_layers.size() - i - 1];
        NeuronLayer& nextLayer = _layers[_layers.size() - i];
        _transformations[_transformations.size() - i].applyError(prevLayer.output, nextLayer.outputError, prevLayer.inputError, weight);
        prevLayer.calculatePropagation();
    }
}

size_t TPNN::Perceptron::inputSize() const {
    return _transformations[0].inputSize();
}

size_t TPNN::Perceptron::outputSize() const {
    return _transformations[_transformations.size() - 1].outputSize();
}

void TPNN::Perceptron::debugPrint() {
    for (auto & _transformation : _transformations) {
        std::cout << "-----" << std::endl;
        _transformation.debugPrint();
    }
    std::cout << "-----" << std::endl;

}

void TPNN::Perceptron::save(std::ostream &s) {
    size_t size = _transformations.size();
    s.write((char*)&size, sizeof(size_t));
    for (auto & _transformation : _transformations) {
        _transformation.save(s);
    }
}

TPNN::Perceptron::Perceptron(std::istream &s, ActivationFunction function) {
    size_t size;
    s.read((char*)&size, sizeof(size_t));
    _transformations.resize(size);
    _layers.resize(size + 1);
    for (size_t i = 0; i < _transformations.size(); i++) {
        _transformations[i] = DenseLayer(s);
        if (i == 0) {
            _layers[i].init(_transformations[i].inputSize(), function);
        } else {
            if (_layers[i].size != _transformations[i].inputSize())
                throw std::invalid_argument("Wrong sizes");
        }
        _layers[i+1].init(_transformations[i].outputSize(), function);
    }
}

void TPNN::Perceptron::NeuronLayer::calculatePropagation() {
    for (size_t i = 0; i < size; i++)
        outputError[i] = activationFunction.df(input[i]) * inputError[i];
}

void TPNN::Perceptron::NeuronLayer::calculateOutput() {
    for (size_t i = 0; i < size; i++)
        output[i] = activationFunction.f(input[i]);
}

TPNN::Perceptron::NeuronLayer::NeuronLayer() :
        size(0),
        activationFunction({nullptr, nullptr}) {
}

void TPNN::Perceptron::NeuronLayer::init(size_t _size, TPNN::ActivationFunction _function) {
    size = _size;
    activationFunction = _function;
    input.resize(size);
    inputError.resize(size);
    output.resize(size);
    outputError.resize(size);
}

TPNN::Perceptron::NeuronLayer::~NeuronLayer() = default;
