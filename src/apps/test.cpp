#include <iostream>
#import "../components/Tensor.h"

int main() {
    TPNN::Tensor<4, int> t(TPNN::Dimension<4>{4, 2, 2, 2});
    int n = 0;
    for (size_t i = 0; i < t.getDimensions()[3]; i++) {
        for (size_t j = 0; j < t.getDimensions()[2]; j++) {
            for (size_t k = 0; k < t.getDimensions()[1]; k++) {
                for (size_t l = 0; l < t.getDimensions()[0]; l++) {
                    TPNN::Pose<4> p{l, k, j, i};
                    t.val(p) = n;
                    std::cout << "(" << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << "): " << t.val(p) << std::endl;
                    n++;
                }
            }
        }
    }
    std::cout << std::endl << std::endl;
    for (size_t i = 0; i < t.getDimensions()[3]; i++) {
        for (size_t j = 0; j < t.getDimensions()[2]; j++) {
            for (size_t k = 0; k < t.getDimensions()[1]; k++) {
                for (size_t l = 0; l < t.getDimensions()[0]; l++) {
                    TPNN::Pose<4> p {l, k, j, i};
                    std::cout << "(" << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << "): " << t.val(p) << std::endl;
                }
            }
        }
    }
}
