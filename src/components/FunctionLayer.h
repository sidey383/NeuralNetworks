#pragma once
#include "ActivationFunctions.h"
#include "Dimension.h"

namespace TPNN {

    class FunctionLayer {

    public:
        FunctionLayer(ActivationFunctions function, Dimension<1> dim);

    private:
        ActivationFunction function;
        size_t size;
    };

}
