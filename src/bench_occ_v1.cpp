#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace fs = std::filesystem;

static constexpr uint64_t ADAPTIVE_THRESHOLD = 32;

enum class OccVariant {
    Scalar,
    SimdAvx2,
    Adaptive
};

struct FMIndex {
    uint64_t n = 0;
    uint32_t checkpoint_step = 0;
    uint64_t num_blocks = 0;
    std::array<uint64_t,256> C{};
    std::vector<uint32_t> checkpoints;
};

static std::vector<uint8_t> read_file_bytes(const fs::path& path) {
    std::ifstream in(path,std::ios::binary);
    if(!in) throw std::runtime_error("failed to open file");

    in.seekg(0,std::ios::end);
    std::streamsize sz=in.tellg();
    in.seekg(0,std::ios::beg);

    std::vector<uint8_t> out(static_cast<size_t>(sz));

    if(!in.read(reinterpret_cast<char*>(out.data()), sz)) {
        throw std::runtime_error("failed read");
    }

    return out;
}

template<class T>
static void read_one(std::ifstream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if(!in) throw std::runtime_error("read fail");
}

static uint64_t fnv1a64(const void* data, size_t len) {
    auto p = static_cast<const uint8_t*>(data);
    uint64_t h = 14695981039346656037ull;

    for(size_t i=0;i<len;++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }

    return h;
}

static FMIndex load_fm(const fs::path& path) {
    std::ifstream in(path,std::ios::binary);
    if(!in) throw std::runtime_error("open fm fail");

    char magic[8];
    in.read(magic,8);

    if(std::string(magic, magic+8) != std::string("FMBINv2\0", 8)) {
        throw std::runtime_error("bad magic");
    }

    FMIndex fm;

    read_one(in,fm.n);
    read_one(in,fm.checkpoint_step);
    read_one(in,fm.num_blocks);

    for(size_t i=0;i<256;++i) {
        read_one(in,fm.C[i]);
    }

    uint64_t payload;
    uint64_t hash;

    read_one(in,payload);
    read_one(in,hash);

    fm.checkpoints.resize(fm.num_blocks * 256);

    in.read(
        reinterpret_cast<char*>(fm.checkpoints.data()),
        fm.checkpoints.size() * sizeof(uint32_t)
    );

    auto got = fnv1a64(fm.checkpoints.data(), payload);

    if(got != hash) {
        throw std::runtime_error("checksum fail");
    }

    return fm;
}

static inline uint64_t occ_scan_scalar(
    const uint8_t* ptr,
    uint64_t len,
    uint8_t c
) {
    uint64_t total = 0;
    uint64_t i = 0;

    for(; i + 8 <= len; i += 8) {
        total += ptr[i+0] == c;
        total += ptr[i+1] == c;
        total += ptr[i+2] == c;
        total += ptr[i+3] == c;
        total += ptr[i+4] == c;
        total += ptr[i+5] == c;
        total += ptr[i+6] == c;
        total += ptr[i+7] == c;
    }

    for(; i < len; ++i) {
        total += ptr[i] == c;
    }

    return total;
}

static inline uint64_t occ_scan_simd(
    const uint8_t* ptr,
    uint64_t len,
    uint8_t c
) {
    __m256i target = _mm256_set1_epi8((char)c);

    uint64_t total = 0;
    uint64_t i = 0;

    for(; i + 32 <= len; i += 32) {
        __m256i data = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(ptr+i)
        );

        __m256i eq = _mm256_cmpeq_epi8(data, target);

        uint32_t mask = static_cast<uint32_t>(
            _mm256_movemask_epi8(eq)
        );

        total += __builtin_popcount(mask);
    }

    for(; i < len; ++i) {
        total += ptr[i] == c;
    }

    return total;
}

static inline uint64_t compute_offset(
    const FMIndex& fm,
    uint64_t pos
) {
    if(pos > fm.n) {
        pos = fm.n;
    }

    uint64_t block = pos / fm.checkpoint_step;
    uint64_t offset = pos % fm.checkpoint_step;

    if(block >= fm.num_blocks) {
        block = fm.num_blocks - 1;
        offset = pos - block * fm.checkpoint_step;
    }

    return offset;
}

static inline uint64_t occ_core(
    const FMIndex& fm,
    const std::vector<uint8_t>& bwt,
    uint8_t c,
    uint64_t pos,
    OccVariant variant
) {
    if(pos > fm.n) {
        pos = fm.n;
    }

    uint64_t block = pos / fm.checkpoint_step;
    uint64_t offset = pos % fm.checkpoint_step;

    if(block >= fm.num_blocks) {
        block = fm.num_blocks - 1;
        offset = pos - block * fm.checkpoint_step;
    }

    uint64_t base = fm.checkpoints[block * 256 + c];

    auto ptr = bwt.data() + block * fm.checkpoint_step;

    uint64_t extra = 0;

    if(variant == OccVariant::Scalar) {
        extra = occ_scan_scalar(ptr, offset, c);
    } else if(variant == OccVariant::SimdAvx2) {
        extra = occ_scan_simd(ptr, offset, c);
    } else {
        extra = (offset < ADAPTIVE_THRESHOLD)
            ? occ_scan_scalar(ptr, offset, c)
            : occ_scan_simd(ptr, offset, c);
    }

    return base + extra;
}

static const char* variant_name(OccVariant v) {
    if(v == OccVariant::Scalar) return "scalar";
    if(v == OccVariant::SimdAvx2) return "simd_avx2";
    return "adaptive";
}

static uint64_t pct(std::vector<uint64_t> x, double p) {
    std::sort(x.begin(), x.end());

    return x[
        size_t((p / 100.0) * (x.size() - 1))
    ];
}

int main(int argc, char** argv) {
    if(argc != 4) {
        std::cerr << "Usage: bench_occ_v1 <fm.bin> <bwt.bin> <iterations>\n";
        return 1;
    }

    FMIndex fm = load_fm(argv[1]);
    auto bwt = read_file_bytes(argv[2]);
    uint64_t it = std::stoull(argv[3]);

    for(OccVariant variant : {
        OccVariant::Scalar,
        OccVariant::SimdAvx2,
        OccVariant::Adaptive
    }) {
        std::vector<uint64_t> s;
        s.reserve(static_cast<size_t>(it));

        volatile uint64_t sink = 0;
        uint64_t total_scan = 0;

        for(uint64_t i=0; i<it; ++i) {
            uint8_t c = i % 256;

            uint64_t pos =
                (i * 1315423911ull) % (fm.n + 1);

            total_scan += compute_offset(fm, pos);

            auto t0 = std::chrono::high_resolution_clock::now();

            sink += occ_core(
                fm,
                bwt,
                c,
                pos,
                variant
            );

            auto t1 = std::chrono::high_resolution_clock::now();

            s.push_back(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    t1 - t0
                ).count()
            );
        }

        std::cout
            << "{"
            << "\"bench\":\"OCC_BENCH_V1\","
            << "\"occ_variant\":\"" << variant_name(variant) << "\","
            << "\"adaptive_threshold\":" << ADAPTIVE_THRESHOLD << ","
            << "\"iterations\":" << it << ","
            << "\"avg_scan_bytes\":" << (total_scan / it) << ","
            << "\"p50_ns\":" << pct(s,50) << ","
            << "\"p95_ns\":" << pct(s,95) << ","
            << "\"p99_ns\":" << pct(s,99) << ","
            << "\"sink\":" << sink
            << "}\n";
    }

    return 0;
}
