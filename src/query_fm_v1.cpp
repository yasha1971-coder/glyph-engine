#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct FMIndex {
    uint64_t n = 0;
    uint32_t checkpoint_step = 0;
    uint64_t num_blocks = 0;
    std::array<uint64_t, 256> C{};
    std::vector<uint32_t> checkpoints; // [num_blocks][256]
};

static std::vector<uint8_t> read_file_bytes(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open file: " + path.string());
    }

    in.seekg(0, std::ios::end);
    std::streamsize sz = in.tellg();
    if (sz < 0) {
        throw std::runtime_error("failed to get size: " + path.string());
    }
    in.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(static_cast<size_t>(sz));
    if (!in.read(reinterpret_cast<char *>(data.data()), sz)) {
        throw std::runtime_error("failed to read file: " + path.string());
    }
    return data;
}

template <typename T>
static void read_one(std::ifstream &in, T &value) {
    in.read(reinterpret_cast<char *>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("failed while reading binary value");
    }
}

static FMIndex load_fm(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open fm: " + path.string());
    }

    char magic[8];
    in.read(magic, 8);
    if (!in) {
        throw std::runtime_error("failed to read fm magic");
    }

    const std::string got(magic, magic + 8);
    const std::string want("FMBINv1\0", 8);
    if (got != want) {
        throw std::runtime_error("bad fm magic");
    }

    FMIndex fm;
    read_one(in, fm.n);
    read_one(in, fm.checkpoint_step);
    read_one(in, fm.num_blocks);

    for (size_t c = 0; c < 256; ++c) {
        read_one(in, fm.C[c]);
    }

    fm.checkpoints.resize(static_cast<size_t>(fm.num_blocks) * 256);
    in.read(reinterpret_cast<char *>(fm.checkpoints.data()),
            static_cast<std::streamsize>(fm.checkpoints.size() * sizeof(uint32_t)));
    if (!in) {
        throw std::runtime_error("failed to read checkpoints");
    }

    return fm;
}

static std::vector<uint8_t> parse_hex_pattern(const std::string &hex) {
    if (hex.size() % 2 != 0) {
        throw std::runtime_error("hex pattern length must be even");
    }

    std::vector<uint8_t> out;
    out.reserve(hex.size() / 2);

    for (size_t i = 0; i < hex.size(); i += 2) {
        std::string byte_str = hex.substr(i, 2);
        unsigned int v = 0;
        std::stringstream ss;
        ss << std::hex << byte_str;
        ss >> v;
        if (ss.fail()) {
            throw std::runtime_error("bad hex pattern");
        }
        out.push_back(static_cast<uint8_t>(v));
    }

    return out;
}

static uint64_t occ(const FMIndex &fm, const std::vector<uint8_t> &bwt, uint8_t c, uint64_t pos) {
    if (pos == 0) {
        return 0;
    }
    if (pos > fm.n) {
        pos = fm.n;
    }

    uint64_t block = pos / fm.checkpoint_step;
    uint64_t offset = pos % fm.checkpoint_step;

    if (block >= fm.num_blocks) {
        block = fm.num_blocks - 1;
        offset = pos - block * fm.checkpoint_step;
    }

    uint64_t base = fm.checkpoints[static_cast<size_t>(block) * 256 + c];
    uint64_t start = block * fm.checkpoint_step;
    uint64_t end = start + offset;

    uint64_t extra = 0;
    for (uint64_t i = start; i < end; ++i) {
        if (bwt[static_cast<size_t>(i)] == c) {
            extra++;
        }
    }

    return base + extra;
}

static std::pair<uint64_t, uint64_t> backward_search(
    const FMIndex &fm,
    const std::vector<uint8_t> &bwt,
    const std::vector<uint8_t> &pattern
) {
    uint64_t l = 0;
    uint64_t r = fm.n;

    for (auto it = pattern.rbegin(); it != pattern.rend(); ++it) {
        uint8_t c = *it;
        l = fm.C[c] + occ(fm, bwt, c, l);
        r = fm.C[c] + occ(fm, bwt, c, r);
        if (l >= r) {
            return {0, 0};
        }
    }

    return {l, r};
}

int main(int argc, char **argv) {
    try {
        if (argc != 4) {
            std::cerr << "Usage: query_fm_v1 <fm.bin> <bwt.bin> <pattern_hex>\n";
            return 1;
        }

        fs::path fm_path = argv[1];
        fs::path bwt_path = argv[2];
        std::string pattern_hex = argv[3];

        std::cout << "============================================\n";
        std::cout << " QUERY FM V1\n";
        std::cout << "============================================\n";
        std::cout << "fm:      " << fm_path << "\n";
        std::cout << "bwt:     " << bwt_path << "\n";
        std::cout << "pattern: " << pattern_hex << "\n";

        FMIndex fm = load_fm(fm_path);
        std::vector<uint8_t> bwt = read_file_bytes(bwt_path);
        if (bwt.size() != fm.n) {
            throw std::runtime_error("bwt size != fm.n");
        }

        std::vector<uint8_t> pattern = parse_hex_pattern(pattern_hex);
        auto [l, r] = backward_search(fm, bwt, pattern);

        std::cout << "interval: [" << l << ", " << r << ")\n";
        std::cout << "count:    " << (r - l) << "\n";
        std::cout << "done\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 2;
    }
}