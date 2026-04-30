#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "libsais.h"

namespace fs = std::filesystem;

static std::vector<uint8_t> read_file_bytes(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open input: " + path.string());
    }

    in.seekg(0, std::ios::end);
    std::streamsize sz = in.tellg();
    if (sz < 0) {
        throw std::runtime_error("failed to get size: " + path.string());
    }
    in.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(static_cast<size_t>(sz));
    if (!in.read(reinterpret_cast<char *>(data.data()), sz)) {
        throw std::runtime_error("failed to read input: " + path.string());
    }
    return data;
}

static void write_u32_binary(const fs::path &path, const std::vector<int32_t> &sa) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output: " + path.string());
    }

    for (size_t i = 0; i < sa.size(); ++i) {
        uint32_t v = static_cast<uint32_t>(sa[i]);
        out.write(reinterpret_cast<const char *>(&v), sizeof(uint32_t));
    }

    if (!out) {
        throw std::runtime_error("failed to write output: " + path.string());
    }
}

static void validate_sa_basic(const std::vector<int32_t> &sa, size_t n) {
    if (sa.size() != n) {
        throw std::runtime_error("SA size mismatch");
    }

    std::vector<uint8_t> seen(n, 0);
    for (size_t i = 0; i < n; ++i) {
        int32_t x = sa[i];
        if (x < 0 || static_cast<size_t>(x) >= n) {
            throw std::runtime_error("SA out of range at index " + std::to_string(i));
        }
        if (seen[static_cast<size_t>(x)]) {
            throw std::runtime_error("SA duplicate value at index " + std::to_string(i));
        }
        seen[static_cast<size_t>(x)] = 1;
    }
}

int main(int argc, char **argv) {
    try {
        if (argc != 3) {
            std::cerr << "Usage: build_sa <input_corpus.bin> <output_sa.bin>\n";
            return 1;
        }

        fs::path input_path = argv[1];
        fs::path output_path = argv[2];

        std::cout << "============================================\n";
        std::cout << " BUILD TRUE SA\n";
        std::cout << "============================================\n";
        std::cout << "input:  " << input_path << "\n";
        std::cout << "output: " << output_path << "\n";

        std::vector<uint8_t> text = read_file_bytes(input_path);
        const int64_t n64 = static_cast<int64_t>(text.size());

        if (n64 <= 0) {
            throw std::runtime_error("empty input corpus");
        }
        if (n64 > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("corpus too large for int32 SA");
        }

        const int32_t n = static_cast<int32_t>(n64);
        std::cout << "corpus_bytes: " << n << "\n";

        std::vector<int32_t> sa(static_cast<size_t>(n), 0);

        int32_t rc = libsais(
            reinterpret_cast<const uint8_t *>(text.data()),
            sa.data(),
            n,
            0,
            nullptr
        );

        if (rc < 0) {
            throw std::runtime_error("libsais failed with code " + std::to_string(rc));
        }

        std::cout << "libsais_ok\n";

        validate_sa_basic(sa, static_cast<size_t>(n));
        std::cout << "basic_validation_ok\n";

        fs::create_directories(output_path.parent_path());
        write_u32_binary(output_path, sa);

        std::cout << "saved: " << output_path << "\n";
        std::cout << "done\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 2;
    }
}