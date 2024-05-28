#pragma once

namespace TPNN {

    typedef float(*FloatFunction)(float);

    typedef struct {
        FloatFunction f;
        FloatFunction df;
    } ActivationFunction;

    enum class ActivationFunctions {
        RELU,
        TH
    };

    ActivationFunction toFunction(ActivationFunctions f);
}
