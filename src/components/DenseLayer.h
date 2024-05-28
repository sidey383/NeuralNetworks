#pragma once
#include <iostream>

namespace TPNN {

    class DenseLayer {
    public:
        ~DenseLayer();
        DenseLayer();
        DenseLayer(size_t input, size_t output);
        explicit DenseLayer(std::istream& s);
        [[nodiscard]] size_t inputSize() const;
        [[nodiscard]] size_t outputSize() const;
        void resize(size_t input, size_t output);
        void apply(const std::vector<float>& input, std::vector<float>& output);
        void applyError(const std::vector<float>& input, const std::vector<float>&error, std::vector<float>& propagatedError, float weight);
        void debugPrint();
        void save(std::ostream& s) const;

    private:
        std::vector<std::vector<float>> _weights{0};
        std::vector<float> _bias{0};
    };

}
