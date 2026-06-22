#ifndef GCOPTER_BENCHMARK_CSV_WRITER_HPP
#define GCOPTER_BENCHMARK_CSV_WRITER_HPP

#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace firi_benchmark
{
    inline bool fileExists(const std::string &path)
    {
        struct stat st;
        return ::stat(path.c_str(), &st) == 0;
    }

    inline std::string csvEscape(const std::string &value)
    {
        bool quote = false;
        for (const char c : value)
        {
            if (c == ',' || c == '"' || c == '\n' || c == '\r')
            {
                quote = true;
                break;
            }
        }
        if (!quote)
        {
            return value;
        }
        std::string out = "\"";
        for (const char c : value)
        {
            if (c == '"')
            {
                out += "\"\"";
            }
            else
            {
                out += c;
            }
        }
        out += "\"";
        return out;
    }

    template <typename T>
    inline std::string toCsvValue(const T &value)
    {
        std::ostringstream oss;
        oss.precision(17);
        oss << value;
        return oss.str();
    }

    inline std::string toCsvValue(const std::string &value)
    {
        return csvEscape(value);
    }

    inline std::string toCsvValue(const char *value)
    {
        return csvEscape(value == nullptr ? std::string() : std::string(value));
    }

    inline std::string toCsvValue(const bool value)
    {
        return value ? "1" : "0";
    }

    class CsvWriter
    {
    public:
        CsvWriter() = default;

        CsvWriter(const std::string &path,
                  const std::vector<std::string> &header)
        {
            open(path, header);
        }

        void open(const std::string &path,
                  const std::vector<std::string> &header)
        {
            const bool append = fileExists(path);
            stream_.open(path.c_str(), std::ios::out | std::ios::app);
            if (!append)
            {
                writeVector(header);
            }
        }

        template <typename... Args>
        void writeRow(const Args &...args)
        {
            std::vector<std::string> fields;
            fields.reserve(sizeof...(Args));
            appendFields(fields, args...);
            writeVector(fields);
        }

        bool good() const
        {
            return stream_.good();
        }

    private:
        std::ofstream stream_;

        void writeVector(const std::vector<std::string> &fields)
        {
            for (std::size_t i = 0; i < fields.size(); ++i)
            {
                if (i != 0)
                {
                    stream_ << ',';
                }
                stream_ << fields[i];
            }
            stream_ << '\n';
            stream_.flush();
        }

        inline void appendFields(std::vector<std::string> &) {}

        template <typename T, typename... Rest>
        void appendFields(std::vector<std::string> &fields,
                          const T &value,
                          const Rest &...rest)
        {
            fields.push_back(toCsvValue(value));
            appendFields(fields, rest...);
        }
    };
}

#endif
