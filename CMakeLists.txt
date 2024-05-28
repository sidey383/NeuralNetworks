cmake_minimum_required(VERSION 3.28)
project(TPNN)

set(CMAKE_CXX_STANDARD 17)

add_executable(task2_Laptop
        src/apps/mainLaptop.cpp
        src/components/DenseLayer.cpp
        src/Perceptron.cpp
        src/DataSet.cpp
        src/Teacher.cpp
        src/RMSETeacher.cpp
        src/SoftMaxTeacher.cpp
        src/components/ActivationFunctions.cpp
        src/components/ConvolutionalLayer.cpp
        src/components/DenseLayer.h
        src/components/Layer.h
)

add_executable(task2_Mushroom
        src/apps/mainMushroom.cpp
        src/components/DenseLayer.cpp
        src/Perceptron.cpp
        src/DataSet.cpp
        src/Teacher.cpp
        src/RMSETeacher.cpp
        src/SoftMaxTeacher.cpp
        src/components/ActivationFunctions.cpp
        src/components/DenseLayer.h
        src/components/Layer.h
)

add_executable(task4_Test
        src/apps/test.cpp
        src/components/Dimension.h
        src/components/Tensor.h
        src/components/DenseLayer.h
        src/components/Layer.h
)
