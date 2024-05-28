#pragma once

#include <cstddef>
#include <cstring>

namespace TPNN {

    template<int N>
    struct Dimension {

        size_t dims[N];

        [[nodiscard]] size_t total() const {
            size_t res = 1;
            for (size_t i = 0; i < N; i++) {
                res *= dims[i];
            }
            return res;
        }

        [[nodiscard]] size_t total(size_t subSize) const {
            size_t res = 1;
            for (size_t i = 0; i < std::min((size_t)N, subSize); i++) {
                res *= dims[i];
            }
            return res;
        }

        Dimension<N - 1> sub() const {
            Dimension<N - 1> res = Dimension<N - 1>();
            std::memcpy(res.dims, dims, (N-1) * sizeof (size_t));
            return res;
        }

        size_t &&operator[](size_t i) const {
            return dims[i];
        }

        size_t &operator[](size_t i) {
            return dims[i];
        }

    };

    template <>
    struct Dimension<1> {

        size_t dims[1];

        [[nodiscard]] size_t total() const {
            return dims[0];
        }

        size_t &operator[](size_t i) {
            return dims[i];
        }
    };

    template<int N, int M>
    Dimension<N> resizeDimensions(Dimension<M> i) {
        Dimension<N> res = Dimension<N>();
        std::memcpy(res.dims, i.dims, (N) * sizeof (size_t));
        return res;
    }

    template<int N>
    struct Pose {
        size_t pose[N];

        size_t &operator[](size_t i) {
            return pose[i];
        }
    };


    template<int N, int M>
    size_t toFlat(Dimension<N> d, Pose<M> p) {
        size_t res = p[0];
        size_t prevSize = d[0];
        for (size_t i = 1; i < N; i++) {
            res += p[i] * prevSize;
            prevSize *= d[i];
        }
        return res;
    }

}
