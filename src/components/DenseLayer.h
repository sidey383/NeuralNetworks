#pragma once
#include <iostream>
#include "Layer.h"

namespace TPNN {

    class DenseLayer : public Layer<1, 1> {
    public:
        ~DenseLayer() override;
        DenseLayer();
        DenseLayer(size_t input, size_t output);
        explicit DenseLayer(std::istream& s);
        [[nodiscard]] size_t inputSize() const;
        [[nodiscard]] size_t outputSize() const;
        void resize(size_t input, size_t output);
        void apply(const Tensor<1, float>& input, Tensor<1, float>& output) override;
        void applyError(const Tensor<1, float>& error, Tensor<1, float>& propagatedError, float weight) override;
        void debugPrint() override;
        void save(std::ostream& s) const;

    private:
        Tensor<2, float> _weights{};
        Tensor<1, float> _bias{};
    };

}
