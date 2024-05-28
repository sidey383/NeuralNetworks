#include "SoftMaxTeacher.h"
#include <cmath>

float TPNN::SoftMaxTeacher::accuracy(const std::vector<std::vector<float>> &input,
                                      const std::vector<std::vector<float>> &output) {
    unsigned int correct = 0;
    for (size_t n = 0; n < input.size(); n++) {
        const std::vector<float>& r = perceptron.calculate(input[n]);
        unsigned int expected = 0;
        unsigned int actual = 0;
        float max = r[0];
        for (size_t i = 1; i < r.size(); i++) {
            if (max < r[i]) {
                actual = i;
                max = r[i];
            }
        }
        max = output[n][0];
        for (size_t i = 0; i < r.size(); i++) {
            if (max < output[n][i]) {
                expected = i;
                max = output[n][i];
            }
        }
        correct += actual == expected;
    }
    return (float) correct / (float) input.size() ;
}

std::vector<float>
TPNN::SoftMaxTeacher::error(const std::vector<float> &output, const std::vector<float> &expected) {
    float div = 0;
    for (float v : output) {
        div += exp(v);
    }
    std::vector<float> err(output.size());
    for (int i = 0; i < output.size(); i++) {
        err[i] = (exp(output[i]) / div) - expected[i];
    }
    return err;
}

std::vector<float> TPNN::SoftMaxTeacher::softMax(const std::vector<float> &output) {
    float div = 0;
    for (float i : output) {
        div += exp(i);
    }
    std::vector<float> softMax(output.size());
    for (int i = 0; i < output.size(); i++) {
        softMax[i] = output[i] / div;
    }
    return softMax;
}
