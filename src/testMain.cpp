#include "components/MatrixTypes2.h"
#include <iostream>
#include <tuple>

using namespace TPNN;


int main() {
    product(1, 2, 3, 4);
    std::vector<SubTensor<double, 1, 2>> v;
    Dimensions<2> dims2{{1, 2}};
    Dimensions<3> dims3{{1, 2, 3}};
    std::cout << "expect 1: " << dims2.sub()[0] << std::endl;
    std::cout << "expect 1: " << dims3.sub()[0] << std::endl;
    Pose<2> pose2{{0, 2}};
    Pose<3> pose3{{0, 1, 1}};
    std::cout << "expect 2: " << toFlat(dims2, pose2) << std::endl;
    std::cout << "expect 3: " << toFlat(dims3, pose3) << std::endl;
    TPNN::Tensor3<int, 4> t = Tensor<int, 4>({{2, 2, 2, 2}});
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            for (size_t k = 0; k < 2; k++) {
                for (size_t l = 0; l < 2; l++) {
                    t.set({{i, j, k, l}}, 10 + i + j*2 + k*4 + l*8);
                }
            }
        }
    }
    std::cout << t.get({{1, 0, 0, 1}}) << " " << t.get({{0, 0, 0, 0}})  << std::endl;
    TPNN::Tensor3<int, 2> t2 = Tensor<int, 2>({{4, 3}});
    int n = 0;
    for (size_t i = 0; i < t2.size()[0]; i++) {
        for (size_t j = 0; j < t2.size()[1]; i++) {
            t2[j][i] = n++;
        }
    }
    int i1 = t2.get({{0, 0}});
    int i2 = t2[0][2];
    int i3 = t2.get({{2, 0}});
    int i4 = t2[1][2];
    std::cout << i1 << " " << i2 << " " << i3 << " " << i4  << std::endl;
    std::tuple<int, int, int, int> tuple(10, 20, 30, 40);
}
