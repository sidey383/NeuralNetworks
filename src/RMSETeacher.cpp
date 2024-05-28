#include "RMSETeacher.h"
#include <cmath>

using namespace TPNN;

float TPNN::RMSETeacher::accuracy(const std::vector<std::vector<float>> &input,
                                   const std::vector<std::vector<float>> &output) {
    float error = 0;
    for (size_t i = 0; i < input.size(); i++) {
        const std::vector<float>& r = perceptron.calculate(input[i]);
        error += (r[0] - output[i][0])*(r[0] - output[i][0]);
    }
    return 1 - sqrt(error / (float) input.size());
}

std::vector<float>
TPNN::RMSETeacher::error(const std::vector<float> &output, const std::vector<float> &expected) {
    return std::vector<float>(1, output[0] - expected[0]);
}
