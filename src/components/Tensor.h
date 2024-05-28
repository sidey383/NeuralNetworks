#pragma once
#include "Dimension.h"
#include <vector>

namespace TPNN {

    template<size_t Dim, typename T>
    class Tensor {
    public:
        Tensor() = default;

        explicit Tensor(Dimension<Dim> dim) : dim(dim) {
            values.resize(dim.total());
        }

        void resize(Dimension<Dim> dimension) {
            dim = dimension;
            values.resize(dimension.total());
        }

        Dimension<Dim> getDimensions() const {
            return dim;
        }

        T&& val(Pose<Dim> pose) const {
            return values[flatPose(pose)];
        }

        T& val(Pose<Dim> pose) {
            return values[flatPose( pose)];
        }

        std::vector<T>& getFlat() {
            return values;
        }

        std::vector<T>&& getFlat() const {
            return values;
        }

    protected:
        size_t flatPose(Pose<Dim> pose) {
            size_t val = pose[0];
            size_t mult = dim[0];
            for (size_t i = 1; i < Dim; i++) {
                val += pose[i] * mult;
                mult *= dim[i];
            }
            return val;
        }

    private:
        std::vector<T> values;
        Dimension<Dim> dim;
    };

    template<typename T>
    class Tensor<1, T> {
    public:
        Tensor() = default;

        Tensor(std::vector<T> values) : values(values) {} // NOLINT(*-explicit-constructor)

        explicit Tensor(Dimension<1> dim) {
            values.resize(dim[0]);
        }

        explicit Tensor(size_t size) {
            values.resize(size);
        }

        void resize(Dimension<1> dimension) {
            values.resize(dimension[0]);
        }

        void resize(size_t size) {
            values.resize(size);
        }

        Dimension<1> getDimensions() const {
            return Dimension<1>{values.size()};
        }

        size_t getSize() {
            return values.size();
        }

        T&& val(Pose<1> pose) const {
            return values[pose[0]];
        }

        T& val(Pose<1> pose) {
            return values[pose[0]];
        }

        const T&& val(size_t pose) const {
            return values[pose];
        }

        T& val(size_t pose) {
            return values[pose];
        }

        std::vector<T>& getFlat() {
            return values;
        }

        std::vector<T>&& getFlat() const {
            return values;
        }

        T operator [] (size_t pose) const {
            return values[pose];
        }


        T& operator [] (size_t pose) {
            return values[pose];
        }

        [[nodiscard]] size_t size() const {
            return values.size();
        }

    protected:

    private:
        std::vector<T> values;
    };

}