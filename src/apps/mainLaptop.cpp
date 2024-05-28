#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include "../Perceptron.h"
#include "../DataSet.h"
#include "../RMSETeacher.h"
#include <cfloat>

float func(float a) {
    return a < 0 ? 0 : a;
}

float dfunc(float a) {
    return a < 0 ? 0 : 1;
}

int main() {
#ifndef NDEBUG
    _clearfp();
    _controlfp(_controlfp(0, 0) & ~(_EM_INVALID | _EM_ZERODIVIDE | _EM_OVERFLOW),
               _MCW_EM);
#endif
    TPNN::DataSet data({
                               {"Brand",           TPNN::FieldType::TYPE,       false},
                               {"ProcessorSpeed",  TPNN::FieldType::FRACTIONAL, false},
                               {"RamSize",         TPNN::FieldType::INTEGER,    false},
                               {"StorageCapacity", TPNN::FieldType::INTEGER,    false},
                               {"ScreenSize",      TPNN::FieldType::FRACTIONAL, false},
                               {"Weight",          TPNN::FieldType::FRACTIONAL, false},
                               {"Price",           TPNN::FieldType::FRACTIONAL, true}
                       });
    std::ifstream dataStream;
    dataStream.open("..\\..\\task2\\Laptop_price.csv");
    if (!dataStream) {
        std::cerr << "Can't open file\n";
        return 0;
    }
    std::string buf;
    // skip header
    std::getline(dataStream, buf);
    while (!dataStream.eof()) {
        std::getline(dataStream, buf);
        std::istringstream l(buf);
        std::vector<std::string> values(7);
        for (size_t i = 0; i < 7; i++) {
            std::getline(l, values[i], ',');
        }
        data.addValue(values);
    }
    size_t dataSetSize = data.recordCount();
    size_t trainStart = 0;
    size_t trainEnd = dataSetSize * 3 / 4;
    size_t testStart = trainEnd;
    size_t testEnd = dataSetSize;

    std::vector<std::vector<float>> trainInput = data.input(trainStart, trainEnd);
    std::vector<std::vector<float>> trainOutput = data.output(trainStart, trainEnd);
    std::vector<std::vector<float>> testInput = data.input(testStart, testEnd);
    std::vector<std::vector<float>> testOutput = data.output(testStart, testEnd);

    size_t sizes[] = {data.inputLen(), 10, 10, data.outputLen()};
    TPNN::Perceptron perceptron(sizeof(sizes) / sizeof(sizes[0]), sizes, {func, dfunc});
    TPNN::RMSETeacher teacher(perceptron);


    float accuracy = teacher.teach(
            trainInput, trainOutput,
            testInput, testOutput,
            0.94);
    for (float d: teacher.getHistory()) {
        std::cout << d << std::endl;
    }

    for (size_t i = 0; i < testInput.size(); i++) {
        const std::vector<float> result = perceptron.calculate(testInput[i]);
        std::vector<std::string> inputStr = data.fromInput(testInput[i]);
        std::vector<std::string> outputExpectStr = data.fromOutput(testOutput[i]);
        std::vector<std::string> outputActualStr = data.fromOutput(result);
        std::cout << "-----------------\n";
        std::cout << "Row: \n\t";
        for (const std::string &s: inputStr) {
            std::cout << s << " ";
        }
        std::cout << "\nActual: \n\t";
        for (const std::string &s: outputActualStr) {
            std::cout << s << " ";
        }
        std::cout << "\nExpect: \n\t";
        for (const std::string &s: outputExpectStr) {
            std::cout << s << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\n\nAccuracy: " << accuracy << std::endl;
    std::ofstream perceptronOut;
    perceptronOut.open("LaptopPricePerceptron.dat");
    perceptron.save(perceptronOut);
}
