#pragma once
#include <vector>
#include "components/DenseLayer.h"
#include <istream>

namespace TPNN {

    typedef float(*FloatFunction)(float);

    typedef struct {
        FloatFunction f;
        FloatFunction df;
    } ActivationFunction;

    class Perceptron {
    public:
        Perceptron(size_t layerCount, size_t* layerSizes, ActivationFunction activationFunction);
        Perceptron(std::istream& s, ActivationFunction function);
        ~Perceptron();
        const std::vector<float>& calculate(const std::vector<float>& input);
        void applyError(const std::vector<float>& error, float weight = 0.0001);
        [[nodiscard]] size_t inputSize() const;
        [[nodiscard]] size_t outputSize() const;
        void debugPrint();
        void save(std::ostream& s);

    protected:

        struct NeuronLayer {
            NeuronLayer();
            ~NeuronLayer();

            void init(size_t size, ActivationFunction function);
            void calculateOutput();
            void calculatePropagation();

            size_t size;
            std::vector<float> input{0};
            std::vector<float> output{0};
            std::vector<float> inputError{0};
            std::vector<float> outputError{0};
            ActivationFunction activationFunction;
        };

    private:

        std::vector<NeuronLayer> _layers;
        std::vector<DenseLayer> _transformations;
    };
}

