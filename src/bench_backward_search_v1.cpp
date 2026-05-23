#include <array>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static constexpr uint64_t OCC_ADAPTIVE_THRESHOLD = 32;

struct FMIndex {
    uint64_t n = 0;
    uint32_t checkpoint_step = 0;
    uint64_t num_blocks = 0;
    std::array<uint64_t, 256> C{};
    std::vector<uint32_t> checkpoints;
};

static uint64_t pct(std::vector<uint64_t> x, double p) {
    std::sort(x.begin(), x.end());
    return x[size_t((p / 100.0) * (x.size() - 1))];
}

static std::vector<uint8_t> read_file_bytes(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open file");

    in.seekg(0, std::ios::end);
    std::streamsize sz = in.tellg();
    in.seekg(0, std::ios::beg);

    std::vector<uint8_t> out(static_cast<size_t>(sz));

    if (!in.read(reinterpret_cast<char*>(out.data()), sz)) {
        throw std::runtime_error("failed read");
    }

    return out;
}

template <class T>
static void read_one(std::ifstream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("read fail");
}

static uint64_t fnv1a64(const void* data, size_t len) {
    auto p = static_cast<const uint8_t*>(data);
    uint64_t h = 14695981039346656037ull;

    for (size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }

    return h;
}

static FMIndex load_fm(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("open fm fail");

    char magic[8];
    in.read(magic, 8);

    if (std::string(magic, magic + 8) != std::string("FMBINv2\0", 8)) {
        throw std::runtime_error("bad magic");
    }

    FMIndex fm;

    read_one(in, fm.n);
    read_one(in, fm.checkpoint_step);
    read_one(in, fm.num_blocks);

    if (fm.checkpoint_step == 0) throw std::runtime_error("bad checkpoint_step");
    if (fm.num_blocks == 0) throw std::runtime_error("bad num_blocks");

    for (size_t i = 0; i < 256; ++i) {
        read_one(in, fm.C[i]);
    }

    uint64_t payload = 0;
    uint64_t hash = 0;

    read_one(in, payload);
    read_one(in, hash);

    fm.checkpoints.resize(static_cast<size_t>(fm.num_blocks) * 256);

    in.read(
        reinterpret_cast<char*>(fm.checkpoints.data()),
        static_cast<std::streamsize>(fm.checkpoints.size() * sizeof(uint32_t))
    );

    if (!in) throw std::runtime_error("checkpoint read fail");

    uint64_t expected =
        static_cast<uint64_t>(fm.checkpoints.size() * sizeof(uint32_t));

    if (payload != expected) throw std::runtime_error("payload size mismatch");

    uint64_t got = fnv1a64(fm.checkpoints.data(), static_cast<size_t>(payload));

    if (got != hash) throw std::runtime_error("checkpoint checksum mismatch");

    return fm;
}

static std::vector<uint8_t> parse_hex_pattern(const std::string& hex) {
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

        if (ss.fail()) throw std::runtime_error("bad hex pattern");

        out.push_back(static_cast<uint8_t>(v));
    }

    return out;
}

static inline uint64_t occ_scan_scalar(
    const uint8_t* ptr,
    uint64_t len,
    uint8_t c
) {
    uint64_t total = 0;
    uint64_t i = 0;

    for (; i + 8 <= len; i += 8) {
        total += ptr[i + 0] == c;
        total += ptr[i + 1] == c;
        total += ptr[i + 2] == c;
        total += ptr[i + 3] == c;
        total += ptr[i + 4] == c;
        total += ptr[i + 5] == c;
        total += ptr[i + 6] == c;
        total += ptr[i + 7] == c;
    }

    for (; i < len; ++i) {
        total += ptr[i] == c;
    }

    return total;
}

static inline uint64_t occ_scan_avx2(
    const uint8_t* ptr,
    uint64_t len,
    uint8_t c
) {
    __m256i target = _mm256_set1_epi8((char)c);

    uint64_t total = 0;
    uint64_t i = 0;

    for (; i + 32 <= len; i += 32) {
        __m256i data =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + i));

        __m256i eq =
            _mm256_cmpeq_epi8(data, target);

        uint32_t mask =
            static_cast<uint32_t>(_mm256_movemask_epi8(eq));

        total += __builtin_popcount(mask);
    }

    for (; i < len; ++i) {
        total += ptr[i] == c;
    }

    return total;
}

static uint64_t occ(
    const FMIndex& fm,
    const std::vector<uint8_t>& bwt,
    uint8_t c,
    uint64_t pos
) {
    if (pos == 0) return 0;
    if (pos > fm.n) pos = fm.n;

    uint64_t block = pos / fm.checkpoint_step;
    uint64_t offset = pos % fm.checkpoint_step;

    if (block >= fm.num_blocks) {
        block = fm.num_blocks - 1;
        offset = pos - block * fm.checkpoint_step;
    }

    uint64_t base =
        fm.checkpoints[static_cast<size_t>(block) * 256 + c];

    uint64_t start =
        block * fm.checkpoint_step;

    const uint8_t* ptr =
        bwt.data() + start;

    uint64_t extra =
        (offset < OCC_ADAPTIVE_THRESHOLD)
            ? occ_scan_scalar(ptr, offset, c)
            : occ_scan_avx2(ptr, offset, c);

    return base + extra;
}

static std::pair<uint64_t, uint64_t> backward_search(
    const FMIndex& fm,
    const std::vector<uint8_t>& bwt,
    const std::vector<uint8_t>& pattern
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

static std::string byte_to_hex(uint8_t b) {
    const char* digits = "0123456789abcdef";

    std::string out;
    out.push_back(digits[(b >> 4) & 15]);
    out.push_back(digits[b & 15]);

    return out;
}

static std::vector<std::string> make_hex_patterns(
    const std::vector<uint8_t>& bwt,
    uint64_t len,
    uint64_t iterations
) {
    std::vector<std::string> out;
    out.reserve(static_cast<size_t>(iterations));

    uint64_t max_base = 1;

    if (bwt.size() > len + 1) {
        max_base = bwt.size() - len;
    }

    for (uint64_t i = 0; i < iterations; ++i) {
        uint64_t base =
            (i * 1315423911ull) % max_base;

        std::string hex;
        hex.reserve(static_cast<size_t>(len * 2));

        for (uint64_t j = 0; j < len; ++j) {
            hex += byte_to_hex(bwt[static_cast<size_t>(base + j)]);
        }

        out.push_back(hex);
    }

    return out;
}

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr
                << "Usage: bench_backward_search_v1 <fm.bin> <bwt.bin> <iterations>\n";
            return 1;
        }

        FMIndex fm = load_fm(argv[1]);
        auto bwt = read_file_bytes(argv[2]);

        if (bwt.size() != fm.n) {
            throw std::runtime_error("bwt size != fm.n");
        }

        uint64_t it = std::stoull(argv[3]);

        for (uint64_t len : {4, 8, 16, 32}) {
            auto patterns =
                make_hex_patterns(bwt, len, it);

            std::vector<uint64_t> samples;
            samples.reserve(static_cast<size_t>(it));

            volatile uint64_t sink = 0;

            auto bench0 =
                std::chrono::high_resolution_clock::now();

            for (uint64_t i = 0; i < it; ++i) {
                auto t0 =
                    std::chrono::high_resolution_clock::now();

                auto pattern =
                    parse_hex_pattern(patterns[static_cast<size_t>(i)]);

                auto interval =
                    backward_search(fm, bwt, pattern);

                sink +=
                    (interval.second - interval.first);

                auto t1 =
                    std::chrono::high_resolution_clock::now();

                samples.push_back(
                    uint64_t(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            t1 - t0
                        ).count()
                    )
                );
            }

            auto bench1 =
                std::chrono::high_resolution_clock::now();

            double sec =
                double(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        bench1 - bench0
                    ).count()
                ) / 1000000000.0;

            uint64_t qps =
                uint64_t(double(it) / sec);

            std::cout
                << "{"
                << "\"bench\":\"BACKWARD_SEARCH_BENCH_V1\","
                << "\"pipeline\":\"parse_hex+backward_search+adaptive_occ\","
                << "\"pattern_len\":" << len << ","
                << "\"iterations\":" << it << ","
                << "\"adaptive_threshold\":" << OCC_ADAPTIVE_THRESHOLD << ","
                << "\"p50_ns\":" << pct(samples, 50) << ","
                << "\"p95_ns\":" << pct(samples, 95) << ","
                << "\"p99_ns\":" << pct(samples, 99) << ","
                << "\"queries_per_sec\":" << qps << ","
                << "\"sink\":" << sink
                << "}\n";
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr
            << "ERROR: "
            << e.what()
            << "\n";

        return 2;
    }
}