#pragma once

#include <cstddef>
#include <vector>
#include "Dimension.h"

namespace TPNN {

    template<typename T> class Matrix;
    template<typename T> class Tenser;

    template<typename T>
    class Matrix {
    public:

        Matrix<T>() = default;

        explicit Matrix<T>(Dimension<2> dim) {
            dimension = dim;
            values.resize(dim.total());
        }

        Matrix<T>(Dimension<2> dim, std::vector<T> part) {
            dimension = dim;
            values(part.data(), part.data() + part.size());
        }

        void resize(Dimension<2> dim) {
            dimension = dim;
            values.resize(dim.total());
        }

        T&& val(size_t x, size_t y) const {
            values[x + y * dimension[0]];
        }

        T& val(size_t x, size_t y) {
            values[x + y * dimension[0]];
        }

    private:

        std::vector<T> values;
        Dimension<2> dimension{0, 0};
    };

    template <typename T>
    class Tensor3 {
    public:

        Tensor3<T> () = default;

        explicit Tensor3<T>(Dimension<3> dim) {
            values.resize(dim.total());
        }

        void resize(Dimension<3> dim) {
            values.resize(dim.total());
        }

        T&& val(size_t x, size_t y, size_t z) const {
            values[x + y * dimension.total(1) + z * dimension.total(2)];
        }

        T& val(size_t x, size_t y, size_t z) {
            values[x + y * dimension.total(1) + z * dimension.total(2)];
        }

        [[nodiscard]] std::vector<T> &getValues() {
            return static_cast<std::vector<float> &>(values);
        }

        [[nodiscard]] const std::vector<T> &getValuesConst() const {
            return static_cast<const std::vector<T> &>(values);
        }

        [[nodiscard]] Dimension<3> getDimension() const {
            return dimension;
        }

    private:
        std::vector<T> values;
        Dimension<3> dimension{0, 0, 0};
    };

    template <typename T>
    class Tensor4 {
    public:

        Tensor4<T> () = default;

        explicit Tensor4<T>(Dimension<4> dim) {
            values.resize(dim.total());
        }

        void resize(Dimension<4> dim) {
            values.resize(dim.total());
        }

        T&& val(size_t x, size_t y, size_t z, size_t w) const {
            values[x + y * dimension.total(1) + z * dimension.total(2) + w * dimension.total(3)];
        }

        T& val(size_t x, size_t y, size_t z, size_t w) {
            values[x + y * dimension.total(1) + z * dimension.total(2) + w * dimension.total(3)];
        }

        [[nodiscard]] std::vector<T> &getValues() {
            return static_cast<std::vector<float> &>(values);
        }

        [[nodiscard]] const std::vector<T> &getValuesConst() const {
            return static_cast<const std::vector<T> &>(values);
        }

        [[nodiscard]] Dimension<4> getDimension() const {
            return dimension;
        }

    private:
        std::vector<T> values;
        Dimension<4> dimension{0, 0, 0, 0};
    };
}
