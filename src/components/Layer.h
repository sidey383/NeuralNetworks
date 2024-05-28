#pragma once
#include "Tensor.h"

namespace TPNN {

    template<size_t InDim, size_t OutDim>
    class Layer {
        virtual ~Layer() = default;
        virtual void apply(const Tensor<InDim, float>& input, Tensor<OutDim, float>& output) = 0;
        virtual void applyError(const Tensor<OutDim, float>& error, Tensor<InDim, float>& propagatedError, float weight) = 0;

        virtual void debugPrint() = 0;

    };
}
