#pragma once
#include <string>
#include <vector>

namespace TPNN {

    enum FieldType {
        INTEGER,
        FRACTIONAL,
        TYPE
    };

    struct Field {
        std::string name;
        FieldType type;
        bool isResult;
    };

    class DataSet;
    class ColumBuilder;

    class DataSet {
    public:
        explicit DataSet(const std::vector<Field>& fields);
        ~DataSet();
        void addValue(const std::vector<std::string>& values);
        [[nodiscard]] std::vector<std::vector<float>> input(size_t start, size_t end) const;
        [[nodiscard]] std::vector<std::vector<float>> output(size_t start, size_t end) const;
        [[nodiscard]] size_t inputLen() const;
        [[nodiscard]] size_t outputLen() const;
        [[nodiscard]] size_t recordCount() const;
        [[nodiscard]] std::vector<std::string> fromInput(const std::vector<float>& vals) const;
        [[nodiscard]] std::vector<std::string> fromOutput(const std::vector<float>& vals) const;
    private:
        std::vector<ColumBuilder*> inputColumns{};
        ColumBuilder* outputColumn = nullptr;
        std::vector<ColumBuilder*> allColumns;
        size_t size = 0;
    };
}
