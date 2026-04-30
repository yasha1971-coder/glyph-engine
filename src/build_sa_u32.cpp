#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "libsais64.h"

namespace fs = std::filesystem;

static std::vector<uint8_t> read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open input");

    in.seekg(0, std::ios::end);
    size_t n = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(n);
    if (n) in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(n));
    return data;
}

static void write_sa_u32(const fs::path& path, const std::vector<int64_t>& sa) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open output");

    for (size_t i = 0; i < sa.size(); ++i) {
        int64_t v64 = sa[i];
        if (v64 < 0 || v64 > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::runtime_error("SA value out of uint32 range at i=" + std::to_string(i));
        }
        uint32_t v = static_cast<uint32_t>(v64);
        out.write(reinterpret_cast<const char*>(&v), sizeof(uint32_t));
    }
}

static void validate_sa_basic(const std::vector<int64_t>& sa, uint64_t n) {
    if (sa.size() != n) throw std::runtime_error("SA size mismatch");

    uint64_t minv = UINT64_MAX;
    uint64_t maxv = 0;

    for (size_t i = 0; i < sa.size(); ++i) {
        int64_t x = sa[i];
        if (x < 0 || static_cast<uint64_t>(x) >= n) {
            throw std::runtime_error("SA value out of range at i=" + std::to_string(i));
        }
        uint64_t ux = static_cast<uint64_t>(x);
        minv = std::min(minv, ux);
        maxv = std::max(maxv, ux);
    }

    if (minv != 0 || maxv != n - 1) {
        throw std::runtime_error("SA min/max validation failed");
    }
}

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "usage: build_sa_u32 <input.bin> <output_sa_u32.bin>\n";
            return 2;
        }

        fs::path input = argv[1];
        fs::path output = argv[2];

        std::cout << "============================================\n";
        std::cout << " BUILD SA U32 VIA LIBSAIS64\n";
        std::cout << "============================================\n";
        std::cout << "input:  \"" << input.string() << "\"\n";
        std::cout << "output: \"" << output.string() << "\"\n";

        std::vector<uint8_t> text = read_file(input);
        int64_t n = static_cast<int64_t>(text.size());

        std::cout << "corpus_bytes: " << n << "\n";

        if (n <= 0) throw std::runtime_error("empty corpus");
        if (static_cast<uint64_t>(n) > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::runtime_error("corpus too large for uint32 SA");
        }

        std::vector<int64_t> sa(static_cast<size_t>(n), 0);

        int64_t rc = libsais64(text.data(), sa.data(), n, 0, nullptr);
        if (rc != 0) {
            throw std::runtime_error("libsais64 failed with code " + std::to_string(rc));
        }

        std::cout << "libsais64_ok\n";

        validate_sa_basic(sa, static_cast<uint64_t>(n));
        std::cout << "basic_validation_ok\n";

        write_sa_u32(output, sa);
        std::cout << "saved: \"" << output.string() << "\"\n";
        std::cout << "done\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
