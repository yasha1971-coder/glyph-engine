#pragma once

#include <immintrin.h>
#include <cstdint>

static inline uint64_t occ_simd_avx2(
    const uint8_t* ptr,
    uint64_t len,
    uint8_t symbol
) {
    const __m256i target =
        _mm256_set1_epi8((char)symbol);

    uint64_t total = 0;
    uint64_t i = 0;

    for (; i + 32 <= len; i += 32) {

        __m256i data =
            _mm256_loadu_si256(
                (const __m256i*)(ptr + i)
            );

        __m256i eq =
            _mm256_cmpeq_epi8(
                data,
                target
            );

        uint32_t mask =
            (uint32_t)
            _mm256_movemask_epi8(eq);

        total +=
            (uint64_t)
            __builtin_popcount(mask);
    }

    for (; i < len; ++i) {
        total += (ptr[i] == symbol);
    }

    return total;
}
