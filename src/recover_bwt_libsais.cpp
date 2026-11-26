// GLYPH raw+unique-zero BWT -> original bytes, using the already-vendored libsais.
// No input corpus is required. Primary index is the zero position in the full L.
#include <libsais.h>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: recover_bwt_libsais BWT OUTPUT");
        std::ifstream in(argv[1], std::ios::binary | std::ios::ate);
        auto size = in.tellg();
        if (!in || size < 2 || size > 128 * 1024 * 1024 + 1)
            throw std::runtime_error("unsupported BWT length");
        std::vector<uint8_t> text(static_cast<size_t>(size));
        in.seekg(0);
        if (!in.read(reinterpret_cast<char *>(text.data()), size) || in.peek() != EOF)
            throw std::runtime_error("BWT length changed");
        auto zero = std::find(text.begin(), text.end(), uint8_t(0));
        if (zero == text.end() || std::find(zero + 1, text.end(), uint8_t(0)) != text.end())
            throw std::runtime_error("BWT must contain exactly one sentinel zero");
        int32_t primary = static_cast<int32_t>(zero - text.begin());
        if (primary < 1) throw std::runtime_error("invalid BWT primary index");
        text.erase(zero);
        int32_t n = static_cast<int32_t>(text.size());
        std::vector<int32_t> workspace(static_cast<size_t>(n) + 1);
        if (libsais_unbwt(text.data(), text.data(), workspace.data(), n, nullptr, primary) != 0)
            throw std::runtime_error("libsais_unbwt rejected BWT");
        if (std::find(text.begin(), text.end(), uint8_t(0)) != text.end())
            throw std::runtime_error("recovered corpus contains sentinel");
        std::ofstream out(argv[2], std::ios::binary | std::ios::trunc);
        if (!out.write(reinterpret_cast<char *>(text.data()), n))
            throw std::runtime_error("cannot write recovered corpus");
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
