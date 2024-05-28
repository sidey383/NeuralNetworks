cmake_minimum_required(VERSION 3.28)
project(Lab2)

set(CMAKE_CXX_STANDARD 17)

add_executable(Laptop 
	../src/apps/mainLaptop.cpp
        ../src/components/DenseLayer.cpp
        ../src/Perceptron.cpp
        ../src/DataSet.cpp
        ../src/Teacher.cpp
        ../src/RMSETeacher.cpp
        ../src/SoftMaxTeacher.cpp
        ../src/components/ActivationFunctions.cpp
        ../src/components/ConvolutionalLayer.cpp
)

add_executable(Mushroom 
	../src/apps/mainMushroom.cpp
        ../src/components/DenseLayer.cpp
        ../src/Perceptron.cpp
        ../src/DataSet.cpp
        ../src/Teacher.cpp
        ../src/RMSETeacher.cpp
        ../src/SoftMaxTeacher.cpp
        ../src/components/ActivationFunctions.cpp
)
