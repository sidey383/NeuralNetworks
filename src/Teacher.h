#pragma once
#include "Perceptron.h"

namespace TPNN {

    class Teacher {

    public:
        explicit Teacher(Perceptron& perceptron);
        virtual ~Teacher() = default;

        virtual float teach(
                const std::vector<std::vector<float>>& trainInput,
                const std::vector<std::vector<float>>& trainOutput,
                const std::vector<std::vector<float>>& testInput,
                const std::vector<std::vector<float>>& testOutput,
                float targetAccuracy,
                float learning_rate = 1e-3,
                size_t maxIteration = 10000
                );
        virtual float accuracy(
                const std::vector<std::vector<float>>& input,
                const std::vector<std::vector<float>>& output
                ) = 0;
        const std::vector<float>& getHistory();
    protected:
        Perceptron& perceptron;
        std::vector<float> teachHistory;
        virtual std::vector<float> error(
                const std::vector<float>& output,
                const std::vector<float>& expected
                ) = 0;

    };

}
