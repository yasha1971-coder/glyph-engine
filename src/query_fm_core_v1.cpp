#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct FMCore {
    uint64_t bwt_bytes = 0;
    uint32_t checkpoint_step = 0;
    uint64_t num_checkpoints = 0;
    std::vector<uint64_t> C;
    std::vector<uint64_t> freq;
    std::vector<uint64_t> rank_checkpoints;
    std::vector<uint8_t> bwt;
};

static uint32_t read_u32(std::istream& in) {
    uint8_t b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) throw std::runtime_error("failed to read u32");
    return (uint32_t)b[0]
        | ((uint32_t)b[1] << 8)
        | ((uint32_t)b[2] << 16)
        | ((uint32_t)b[3] << 24);
}

static uint64_t read_u64(std::istream& in) {
    uint8_t b[8];
    in.read(reinterpret_cast<char*>(b), 8);
    if (!in) throw std::runtime_error("failed to read u64");
    return (uint64_t)b[0]
        | ((uint64_t)b[1] << 8)
        | ((uint64_t)b[2] << 16)
        | ((uint64_t)b[3] << 24)
        | ((uint64_t)b[4] << 32)
        | ((uint64_t)b[5] << 40)
        | ((uint64_t)b[6] << 48)
        | ((uint64_t)b[7] << 56);
}

static FMCore load_fm_core(const std::string& fm_core_path, const std::string& bwt_path) {
    FMCore fm;

    {
        std::ifstream in(fm_core_path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open fm_core: " + fm_core_path);

        char magic[4];
        in.read(magic, 4);
        if (!in || std::string(magic, 4) != "FMV1") {
            throw std::runtime_error("bad FMV1 magic");
        }

        fm.bwt_bytes = read_u64(in);
        fm.checkpoint_step = read_u32(in);
        fm.num_checkpoints = read_u64(in);

        if (fm.checkpoint_step == 0) {
            throw std::runtime_error("bad checkpoint_step=0");
        }
        if (fm.num_checkpoints == 0) {
            throw std::runtime_error("bad num_checkpoints=0");
        }

        fm.C.resize(256);
        fm.freq.resize(256);

        for (int i = 0; i < 256; ++i) fm.C[i] = read_u64(in);
        for (int i = 0; i < 256; ++i) fm.freq[i] = read_u64(in);

        const uint64_t total = fm.num_checkpoints * 256ULL;
        fm.rank_checkpoints.resize(static_cast<size_t>(total));
        for (uint64_t i = 0; i < total; ++i) {
            fm.rank_checkpoints[static_cast<size_t>(i)] = read_u64(in);
        }
    }

    {
        std::ifstream in(bwt_path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open bwt: " + bwt_path);

        in.seekg(0, std::ios::end);
        uint64_t sz = static_cast<uint64_t>(in.tellg());
        in.seekg(0, std::ios::beg);

        if (sz != fm.bwt_bytes) {
            throw std::runtime_error("bwt size mismatch vs fm_core");
        }

        fm.bwt.resize(static_cast<size_t>(sz));
        in.read(reinterpret_cast<char*>(fm.bwt.data()), static_cast<std::streamsize>(sz));
        if (!in) throw std::runtime_error("failed reading bwt");
    }

    return fm;
}

static int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

static std::vector<uint8_t> parse_hex_pattern(std::string hex) {
    if (hex.rfind("0x", 0) == 0 || hex.rfind("0X", 0) == 0) {
        hex = hex.substr(2);
    }

    std::string clean;
    clean.reserve(hex.size());
    for (char c : hex) {
        if (!std::isspace(static_cast<unsigned char>(c))) clean.push_back(c);
    }

    if (clean.empty()) throw std::runtime_error("empty hex pattern");
    if (clean.size() % 2 != 0) throw std::runtime_error("hex pattern length must be even");

    std::vector<uint8_t> out;
    out.reserve(clean.size() / 2);

    for (size_t i = 0; i < clean.size(); i += 2) {
        int hi = hex_value(clean[i]);
        int lo = hex_value(clean[i + 1]);
        if (hi < 0 || lo < 0) throw std::runtime_error("invalid hex character");
        out.push_back(static_cast<uint8_t>((hi << 4) | lo));
    }

    return out;
}

static uint64_t rank_symbol(const FMCore& fm, uint8_t c, uint64_t pos) {
    if (pos == 0) return 0;
    if (pos > fm.bwt_bytes) throw std::runtime_error("rank pos out of range");

    uint64_t block = pos / fm.checkpoint_step;
    uint64_t offset = pos % fm.checkpoint_step;

    if (block >= fm.num_checkpoints) {
        throw std::runtime_error("checkpoint block out of range");
    }

    uint64_t base = fm.rank_checkpoints[static_cast<size_t>(block * 256ULL + c)];
    uint64_t start = block * fm.checkpoint_step;

    uint64_t cnt = 0;
    for (uint64_t i = 0; i < offset; ++i) {
        if (fm.bwt[static_cast<size_t>(start + i)] == c) ++cnt;
    }

    return base + cnt;
}

static std::pair<uint64_t, uint64_t> backward_search(const FMCore& fm, const std::vector<uint8_t>& pattern) {
    uint64_t l = 0;
    uint64_t r = fm.bwt_bytes;

    for (auto it = pattern.rbegin(); it != pattern.rend(); ++it) {
        uint8_t c = *it;
        l = fm.C[c] + rank_symbol(fm, c, l);
        r = fm.C[c] + rank_symbol(fm, c, r);
        if (l >= r) return {l, r};
    }

    return {l, r};
}

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr << "usage: query_fm_core_v1 <fm_core.bin> <bwt.bin> <pattern_hex>\n";
            return 2;
        }

        const std::string fm_core_path = argv[1];
        const std::string bwt_path = argv[2];
        const std::string pattern_hex = argv[3];

        FMCore fm = load_fm_core(fm_core_path, bwt_path);
        std::vector<uint8_t> pattern = parse_hex_pattern(pattern_hex);

        auto [l, r] = backward_search(fm, pattern);

        std::cout << "bwt_bytes: " << fm.bwt_bytes << "\n";
        std::cout << "checkpoint_step: " << fm.checkpoint_step << "\n";
        std::cout << "num_checkpoints: " << fm.num_checkpoints << "\n";
        std::cout << "fm_interval: [" << l << ", " << r << "]\n";
        std::cout << "match_count: " << (r - l) << "\n";
        std::cout << "count:    " << (r - l) << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "query_fm_core_v1 error: " << e.what() << "\n";
        return 1;
    }
}
