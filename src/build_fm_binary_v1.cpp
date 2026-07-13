#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "binary_runtime_v1_format.hpp"

namespace fs = std::filesystem;
using namespace glyph::binary_v1;

struct BwtFile {
    uint64_t corpus_bytes = 0;
    uint64_t row_count = 0;
    std::vector<uint16_t> symbols;
};

static BwtFile load_bwt(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);

    if (!in) {
        throw std::runtime_error(
            "failed to open BWT input"
        );
    }

    require_magic(in, BWT_MAGIC);

    if (read_u32_le(in) != FORMAT_VERSION) {
        throw std::runtime_error(
            "unsupported BWT version"
        );
    }

    BwtFile result;
    result.corpus_bytes = read_u64_le(in);
    result.row_count = read_u64_le(in);

    const uint32_t alphabet_size =
        read_u32_le(in);
    const uint32_t logical_sentinel =
        read_u32_le(in);
    const uint32_t width_bits =
        read_u32_le(in);
    const uint64_t payload_bytes =
        read_u64_le(in);
    const uint64_t payload_checksum =
        read_u64_le(in);

    if (
        alphabet_size != ALPHABET_SIZE
        || logical_sentinel != LOGICAL_SENTINEL
        || width_bits != BWT_SYMBOL_WIDTH_BITS
    ) {
        throw std::runtime_error(
            "BWT format parameters mismatch"
        );
    }

    if (
        payload_bytes
        != result.row_count * sizeof(uint16_t)
    ) {
        throw std::runtime_error(
            "BWT payload size mismatch"
        );
    }

    result.symbols.resize(
        static_cast<size_t>(result.row_count)
    );

    for (uint16_t &symbol : result.symbols) {
        symbol = read_u16_le(in);

        if (symbol > LOGICAL_SENTINEL) {
            throw std::runtime_error(
                "BWT symbol outside alphabet"
            );
        }
    }

    if (
        checksum_u16(result.symbols)
        != payload_checksum
    ) {
        throw std::runtime_error(
            "BWT payload checksum mismatch"
        );
    }

    require_no_trailing_data(in);

    return result;
}

int main(int argc, char **argv) {
    try {
        if (argc != 4) {
            std::cerr
                << "Usage: build_fm_binary_v1 "
                << "<bwt.bin> <fm.bin> "
                << "<checkpoint_step>\n";
            return 1;
        }

        const fs::path bwt_path = argv[1];
        const fs::path output_path = argv[2];

        const uint64_t step_input =
            std::stoull(argv[3]);

        if (
            step_input == 0
            || step_input
                > std::numeric_limits<uint32_t>::max()
        ) {
            throw std::runtime_error(
                "invalid checkpoint_step"
            );
        }

        const uint32_t checkpoint_step =
            static_cast<uint32_t>(step_input);

        const BwtFile bwt = load_bwt(bwt_path);

        if (bwt.symbols.empty()) {
            throw std::runtime_error(
                "BWT cannot be empty"
            );
        }

        std::array<uint64_t, ALPHABET_SIZE> hist{};
        hist.fill(0);

        for (uint16_t symbol : bwt.symbols) {
            ++hist[symbol];
        }

        if (hist[LOGICAL_SENTINEL] != 1) {
            throw std::runtime_error(
                "FM requires exactly one sentinel"
            );
        }

        std::array<uint64_t, ALPHABET_SIZE> c_array{};
        c_array.fill(0);

        c_array[LOGICAL_SENTINEL] = 0;

        uint64_t running =
            hist[LOGICAL_SENTINEL];

        for (uint32_t byte = 0; byte < 256; ++byte) {
            c_array[byte] = running;
            running += hist[byte];
        }

        if (running != bwt.row_count) {
            throw std::runtime_error(
                "FM histogram total mismatch"
            );
        }

        const uint64_t num_checkpoints =
            bwt.row_count / checkpoint_step + 1;

        if (
            num_checkpoints
            > std::numeric_limits<size_t>::max()
                / ALPHABET_SIZE
        ) {
            throw std::runtime_error(
                "checkpoint array too large"
            );
        }

        std::vector<uint32_t> checkpoints(
            static_cast<size_t>(
                num_checkpoints * ALPHABET_SIZE
            ),
            0
        );

        std::array<uint64_t, ALPHABET_SIZE> counts{};
        counts.fill(0);

        for (
            uint64_t position = 0;
            position < bwt.row_count;
            ++position
        ) {
            ++counts[
                bwt.symbols[
                    static_cast<size_t>(position)
                ]
            ];

            if (
                (position + 1) % checkpoint_step
                == 0
            ) {
                const uint64_t checkpoint_index =
                    (position + 1)
                    / checkpoint_step;

                for (
                    uint32_t symbol = 0;
                    symbol < ALPHABET_SIZE;
                    ++symbol
                ) {
                    if (
                        counts[symbol]
                        > std::numeric_limits<uint32_t>::max()
                    ) {
                        throw std::runtime_error(
                            "checkpoint count overflow"
                        );
                    }

                    checkpoints[
                        static_cast<size_t>(
                            checkpoint_index
                            * ALPHABET_SIZE
                            + symbol
                        )
                    ] = static_cast<uint32_t>(
                        counts[symbol]
                    );
                }
            }
        }

        Fnv1a64 payload_hash;

        for (uint64_t value : c_array) {
            payload_hash.update_u64_le(value);
        }

        for (uint32_t value : checkpoints) {
            payload_hash.update_u32_le(value);
        }

        const uint64_t c_payload_bytes =
            ALPHABET_SIZE * sizeof(uint64_t);

        const uint64_t checkpoint_payload_bytes =
            static_cast<uint64_t>(
                checkpoints.size()
                * sizeof(uint32_t)
            );

        const uint64_t payload_bytes =
            c_payload_bytes
            + checkpoint_payload_bytes;

        ensure_parent_directory(output_path);

        std::ofstream out(
            output_path,
            std::ios::binary
        );

        if (!out) {
            throw std::runtime_error(
                "failed to open FM output"
            );
        }

        write_magic(out, FM_MAGIC);
        write_u32_le(out, FORMAT_VERSION);
        write_u64_le(out, bwt.corpus_bytes);
        write_u64_le(out, bwt.row_count);
        write_u32_le(out, checkpoint_step);
        write_u64_le(out, num_checkpoints);
        write_u32_le(out, ALPHABET_SIZE);
        write_u32_le(out, LOGICAL_SENTINEL);
        write_u64_le(out, payload_bytes);
        write_u64_le(out, payload_hash.value());

        for (uint64_t value : c_array) {
            write_u64_le(out, value);
        }

        for (uint32_t value : checkpoints) {
            write_u32_le(out, value);
        }

        std::cout
            << "{"
            << "\"ok\":true,"
            << "\"format\":\"GLYPH_FM_BINARY_V1\","
            << "\"bwt_rows\":"
            << bwt.row_count << ","
            << "\"alphabet_size\":"
            << ALPHABET_SIZE << ","
            << "\"checkpoint_step\":"
            << checkpoint_step << ","
            << "\"num_checkpoints\":"
            << num_checkpoints
            << "}\n";

        return 0;
    } catch (const std::exception &error) {
        std::cerr
            << "ERROR: "
            << error.what()
            << "\n";
        return 2;
    }
}
