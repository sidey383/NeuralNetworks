#include <stdexcept>
#include "DataSet.h"

using namespace TPNN;

class TPNN::ColumBuilder {
public:
    virtual ~ColumBuilder() = default;

    virtual void addValue(const std::string &v) = 0;

    [[nodiscard]] virtual std::string valueOf(const std::vector<float> &vals, size_t offset) const = 0;

    [[nodiscard]] virtual size_t valueCount() const = 0;

    virtual void writeValues(float *vals, size_t rowPose, size_t num) const = 0;

};

class TypeColumnBuilder : public TPNN::ColumBuilder {
public:
    TypeColumnBuilder() = default;

    void addValue(const std::string &v) override {
        size_t number = -1;
        for (size_t i = 0; i < available.size(); i++) {
            if (available[i] == v) {
                number = i;
                break;
            }
        }
        if (number == -1) {
            number = available.size();
            available.push_back(v);
        }
        values.push_back(number);
    }

    [[nodiscard]] size_t valueCount() const override {
        return available.size();
    }

    void writeValues(float *vals, size_t rowPose, size_t num) const override {
        if (values.empty())
            return;
        for (size_t type = 0; type < available.size(); type++) {
            vals[rowPose + type] = values[num] == type;
        }
    }

    [[nodiscard]] std::string valueOf(const std::vector<float> &vals, size_t offset) const override {
        size_t maxPose = 0;
        float maxValue = vals[offset];
        for (size_t i = 1; i < available.size(); i++) {
            if (vals[i + offset] > maxValue) {
                maxValue = vals[i + offset];
                maxPose = i;
            }
        }
        return available[maxPose];
    }

private:
    std::vector<size_t> values{};
    std::vector<std::string> available{};
};

class FractionalColumnBuilder : public TPNN::ColumBuilder {
public:
    FractionalColumnBuilder() = default;

    void addValue(const std::string &v) override {
        float d = 0;
        if (!v.empty()) {
            d = std::stof(v);
        }
        if (!hasFirst) {
            min = d;
            max = d;
            hasFirst = true;
        }
        if (d < min) {
            min = d;
        }
        if (d > max) {
            max = d;
        }
        values.push_back(d);
    }

    [[nodiscard]] size_t valueCount() const override {
        return 1;
    }

    void writeValues(float *vals, size_t rowPose, size_t num) const override {
        if (min != max) {
            vals[rowPose] = (values[num] - min) / (max - min);
        } else {
            vals[rowPose] = 0;
        }
    }

    [[nodiscard]] std::string valueOf(const std::vector<float> &vals, size_t offset) const override {
        return std::to_string(vals[offset] * (max - min) + min);
    }

private:
    std::vector<float> values{};
    float min = 0;
    float max = 0;
    bool hasFirst = false;
};

class IntegerColumnBuilder : public TPNN::ColumBuilder {
public:
    IntegerColumnBuilder() = default;

    void addValue(const std::string &v) override {

        int i = 0;
        if (!v.empty()) {
            i = std::stoi(v);
        }
        if (!hasFirst) {
            min = i;
            max = i;
            hasFirst = true;
        }
        if (i < min) {
            min = i;
        }
        if (i > max) {
            max = i;
        }
        values.push_back(i);

    }

    [[nodiscard]] size_t valueCount() const override {
        return 1;
    }

    void writeValues(float *vals, size_t rowPose, size_t num) const override {
        if (min != max) {
            vals[rowPose] = (float) (values[num] - min) / (float) (max - min);
        } else {
            vals[rowPose] = 0;
        }
    }

    [[nodiscard]] std::string valueOf(const std::vector<float> &vals, size_t offset) const override {
        return std::to_string(vals[offset] * (float) (max - min) + (float) min);
    }

private:
    std::vector<int> values{};
    int min = 0;
    int max = 0;
    bool hasFirst = false;
};

TPNN::DataSet::DataSet(const std::vector<Field> &fields) {
    for (const Field &f: fields) {
        ColumBuilder *column = nullptr;
        switch (f.type) {
            case FRACTIONAL:
                column = new FractionalColumnBuilder();
                break;
            case INTEGER:
                column = new IntegerColumnBuilder();
                break;
            case TYPE:
                column = new TypeColumnBuilder();
                break;
        }
        allColumns.push_back(column);
        if (f.isResult) {
            if (outputColumn != nullptr)
                throw std::invalid_argument("More than one output field");
            outputColumn = column;
        } else {
            inputColumns.push_back(column);
        }
    }
    if (outputColumn == nullptr)
        throw std::invalid_argument("No output field");

}

TPNN::DataSet::~DataSet() {
    for (ColumBuilder *c: allColumns)
        delete c;
}

void TPNN::DataSet::addValue(const std::vector<std::string> &values) {
    for (size_t i = 0; i < allColumns.size(); i++) {
        if (i < values.size()) {
            allColumns[i]->addValue(values[i]);
        } else {
            allColumns[i]->addValue("");
        }
    }
    size++;
}

std::vector<std::vector<float>> DataSet::input(size_t start, size_t end) const {
    std::vector<std::vector<float>> values(end - start);
    size_t inputLen = this->inputLen();
    for (int i = 0; i < end - start; i++) {
        size_t pose = 0;
        values[i].resize(inputLen);
        for (const ColumBuilder *b: inputColumns) {
            b->writeValues(values[i].data(), pose, i);
            pose += b->valueCount();
        }
    }
    return values;
}

std::vector<std::vector<float>> DataSet::output(size_t start, size_t end) const {
    std::vector<std::vector<float>> values(end - start);
    size_t outputLen = this->outputLen();
    for (int i = 0; i < end - start; i++) {
        size_t pose = 0;
        values[i].resize(outputLen);
        outputColumn->writeValues(values[i].data(), pose, i);
    }
    return values;
}

size_t DataSet::inputLen() const {
    size_t len = 0;
    for (const ColumBuilder *b: inputColumns) {
        len += b->valueCount();
    }
    return len;
}

size_t DataSet::outputLen() const {
    return outputColumn->valueCount();
}

size_t DataSet::recordCount() const {
    return size;
}

std::vector<std::string> DataSet::fromInput(const std::vector<float> &vals) const {
    std::vector<std::string> values;
    size_t pose = 0;
    for (const ColumBuilder *b: inputColumns) {
        values.push_back(b->valueOf(vals, pose));
        pose += b->valueCount();
    }
    return values;
}

std::vector<std::string> DataSet::fromOutput(const std::vector<float> &vals) const {
    std::vector<std::string> values;
    values.push_back(outputColumn->valueOf(vals, 0));
    return values;
}

