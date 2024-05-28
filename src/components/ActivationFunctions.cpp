#include "ActivationFunctions.h"
#include <cmath>

using namespace TPNN;

float relu(float x) {
    return x < 0 ? 0 : x;
}

float reluD(float x) {
    return x < 0 ? 0 : 1;
}

float th(float x) {
    return std::atan(x);
}


float thD(float x) {
    float v = std::atan(x);
    return 1 - v*v;
}


ActivationFunction toFunction(ActivationFunctions f) {
    switch (f)  {
        case ActivationFunctions::RELU:
            return {relu, reluD};
        case ActivationFunctions::TH:
            return {th, thD};
    }
    return {nullptr, nullptr};
}

