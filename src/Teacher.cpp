#include "Teacher.h"
#include <stdexcept>

using namespace TPNN;

Teacher::Teacher(TPNN::Perceptron &perceptron) : perceptron(perceptron) {}

float Teacher::teach(const std::vector<std::vector<float>> &trainInput, const std::vector<std::vector<float>> &trainOutput,
                      const std::vector<std::vector<float>> &testInput, const std::vector<std::vector<float>> &testOutput,
                      float targetAccuracy, float learning_rate, size_t maxIteration) {
    if (trainInput.size() != trainOutput.size())
        throw std::invalid_argument("trainInput.size() != trainOutput.size()");
    if (testInput.size() != testOutput.size())
        throw std::invalid_argument("testInput.size() != testOutput.size()");
    teachHistory.resize(0);
    float currentAccuracy = accuracy(testInput, testOutput);
    for (size_t i = 0; i < maxIteration; i++) {
        teachHistory.push_back(currentAccuracy);
        if (currentAccuracy >= targetAccuracy) {
            break;
        }
        for (size_t train = 0; train < trainInput.size(); train++) {
            const std::vector<float>& r = perceptron.calculate(trainInput[train]);
            std::vector<float> err = error(r, trainOutput[train]);
            perceptron.applyError(err, learning_rate);
        }
        currentAccuracy = accuracy(testInput, testOutput);
    }
    return currentAccuracy;
}

const std::vector<float> &Teacher::getHistory() {
    return teachHistory;
}

