#pragma once
#include "Teacher.h"

namespace TPNN {

    class RMSETeacher : public Teacher {
    public:
        explicit RMSETeacher(Perceptron& p) : Teacher(p) {}
        float accuracy(
                const std::vector<std::vector<float>>& input,
                const std::vector<std::vector<float>>& output
        ) override;
    protected:
        std::vector<float> error(
                const std::vector<float>& output,
                const std::vector<float>& expected
        ) override;

    };
}


