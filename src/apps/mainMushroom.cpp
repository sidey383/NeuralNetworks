#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include "../Perceptron.h"
#include "../DataSet.h"
#include "../SoftMaxTeacher.h"
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
    std::vector<TPNN::Field> fields = {
            {"poisonous",           TPNN::FieldType::TYPE, true},
            {"cap-shape",  TPNN::FieldType::TYPE, false},
            {"cap-surface",  TPNN::FieldType::TYPE, false},
            {"cap-color",  TPNN::FieldType::TYPE, false},
            {"bruises",  TPNN::FieldType::TYPE, false},
            {"odor",  TPNN::FieldType::TYPE, false},
            {"gill-attachment",  TPNN::FieldType::TYPE, false},
            {"gill-spacing",  TPNN::FieldType::TYPE, false},
            {"gill-size",  TPNN::FieldType::TYPE, false},
            {"gill-color",  TPNN::FieldType::TYPE, false},
            {"stalk-shape",  TPNN::FieldType::TYPE, false},
            {"stalk-root",  TPNN::FieldType::TYPE, false},
            {"stalk-surface-above-ring",  TPNN::FieldType::TYPE, false},
            {"stalk-surface-below-ring",  TPNN::FieldType::TYPE, false},
            {"stalk-color-above-ring",  TPNN::FieldType::TYPE, false},
            {"stalk-color-below-ring",  TPNN::FieldType::TYPE, false},
            {"veil-type",  TPNN::FieldType::TYPE, false},
            {"veil-color",  TPNN::FieldType::TYPE, false},
            {"ring-number",  TPNN::FieldType::TYPE, false},
            {"ring-type",  TPNN::FieldType::TYPE, false},
            {"spore-print-color",  TPNN::FieldType::TYPE, false},
            {"population",  TPNN::FieldType::TYPE, false},
            {"habitat",  TPNN::FieldType::TYPE, false}
    };
    TPNN::DataSet data(fields);
    std::ifstream dataStream;
    dataStream.open("..\\..\\task2\\agaricus-lepiota.data");
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
        std::vector<std::string> values(fields.size());
        for (size_t i = 0; i < fields.size(); i++) {
            std::getline(l, values[i], ',');
        }
        data.addValue(values);
    }
    size_t dataSetSize = data.recordCount();
    size_t trainStart = 0;
    size_t trainEnd = dataSetSize * 9 / 10;
    size_t testStart = trainEnd;
    size_t testEnd = dataSetSize;

    std::vector<std::vector<float>> trainInput = data.input(trainStart, trainEnd);
    std::vector<std::vector<float>> trainOutput = data.output(trainStart, trainEnd);
    std::vector<std::vector<float>> testInput = data.input(testStart, testEnd);
    std::vector<std::vector<float>> testOutput = data.output(testStart, testEnd);

    size_t sizes[] = {data.inputLen(), 16, 8, data.outputLen()};
    TPNN::Perceptron perceptron(sizeof(sizes) / sizeof(sizes[0]), sizes, {func, dfunc});
    TPNN::SoftMaxTeacher teacher(perceptron);


    float accuracy = teacher.teach(
            trainInput, trainOutput,
            testInput, testOutput,
            1,
            1e-4,
            200);
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
    perceptronOut.open("MushroomPerceptron.dat");
    perceptron.save(perceptronOut);
}
