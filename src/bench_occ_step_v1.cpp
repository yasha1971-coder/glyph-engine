#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

static inline uint32_t glyph_popcount32(uint32_t x) {
#if defined(_MSC_VER)
    return static_cast<uint32_t>(__popcnt(x));
#else
    return static_cast<uint32_t>(__builtin_popcount(x));
#endif
}


namespace fs = std::filesystem;

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

static inline uint64_t occ_scan_scalar(
    const uint8_t* ptr,
    uint64_t len,
    uint8_t c
) {
    uint64_t total = 0;
    uint64_t i = 0;

    for (; i + 8 <= len; i += 8) {
        total += ptr[i+0] == c;
        total += ptr[i+1] == c;
        total += ptr[i+2] == c;
        total += ptr[i+3] == c;
        total += ptr[i+4] == c;
        total += ptr[i+5] == c;
        total += ptr[i+6] == c;
        total += ptr[i+7] == c;
    }

    for (; i < len; ++i) {
        total += ptr[i] == c;
    }

    return total;
}

static inline uint64_t occ_scan_simd_avx2(
    const uint8_t* ptr,
    uint64_t len,
    uint8_t c
) {
    __m256i target = _mm256_set1_epi8((char)c);

    uint64_t total = 0;
    uint64_t i = 0;

    for (; i + 32 <= len; i += 32) {
        __m256i data = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(ptr + i)
        );

        __m256i eq = _mm256_cmpeq_epi8(data, target);

        uint32_t mask = static_cast<uint32_t>(
            _mm256_movemask_epi8(eq)
        );

        total += glyph_popcount32(mask);
    }

    for (; i < len; ++i) {
        total += ptr[i] == c;
    }

    return total;
}

static uint64_t pct(std::vector<uint64_t> x, double p) {
    std::sort(x.begin(), x.end());

    return x[
        size_t((p / 100.0) * (x.size() - 1))
    ];
}

static void bench_fixed_scan(
    const std::vector<uint8_t>& bwt,
    uint64_t scan_len,
    uint64_t iterations,
    bool simd
) {
    std::vector<uint64_t> samples;
    samples.reserve(static_cast<size_t>(iterations));

    volatile uint64_t sink = 0;

    uint64_t max_base = 1;
    if (bwt.size() > scan_len + 1) {
        max_base = bwt.size() - scan_len;
    }

    for (uint64_t i = 0; i < iterations; ++i) {
        uint8_t c = static_cast<uint8_t>(i % 256);

        uint64_t base =
            (i * 1315423911ull) % max_base;

        const uint8_t* ptr = bwt.data() + base;

        auto t0 = std::chrono::high_resolution_clock::now();

        sink += simd
            ? occ_scan_simd_avx2(ptr, scan_len, c)
            : occ_scan_scalar(ptr, scan_len, c);

        auto t1 = std::chrono::high_resolution_clock::now();

        samples.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                t1 - t0
            ).count()
        );
    }

    std::cout
        << "{"
        << "\"bench\":\"OCC_STEP_BENCH_V1\","
        << "\"scan_len\":" << scan_len << ","
        << "\"occ_variant\":\"" << (simd ? "simd_avx2" : "scalar") << "\","
        << "\"iterations\":" << iterations << ","
        << "\"p50_ns\":" << pct(samples, 50) << ","
        << "\"p95_ns\":" << pct(samples, 95) << ","
        << "\"p99_ns\":" << pct(samples, 99) << ","
        << "\"sink\":" << sink
        << "}\n";
}

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "Usage: bench_occ_step_v1 <bwt.bin> <iterations>\n";
            return 1;
        }

        auto bwt = read_file_bytes(argv[1]);
        uint64_t iterations = std::stoull(argv[2]);

        for (uint64_t scan_len : {16, 32, 64, 128, 256, 512, 1024}) {
            bench_fixed_scan(bwt, scan_len, iterations, false);
            bench_fixed_scan(bwt, scan_len, iterations, true);
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 2;
    }
}
