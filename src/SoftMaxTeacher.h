#pragma once
#include "Teacher.h"

namespace TPNN {

    class SoftMaxTeacher : public Teacher {
    public:
        explicit SoftMaxTeacher(Perceptron& p) : Teacher(p) {}
        float accuracy(
                const std::vector<std::vector<float>>& input,
                const std::vector<std::vector<float>>& output
        ) override;
        std::vector<float> softMax(const std::vector<float>& output);
    protected:
        std::vector<float> error(
                const std::vector<float>& output,
                const std::vector<float>& expected
        ) override;

    };
}

