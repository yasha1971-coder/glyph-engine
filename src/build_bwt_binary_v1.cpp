#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "binary_runtime_v1_format.hpp"

namespace fs = std::filesystem;
using namespace glyph::binary_v1;

struct SaFile {
    uint64_t corpus_bytes = 0;
    uint64_t row_count = 0;
    std::vector<uint32_t> rows;
};

static SaFile load_sa(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);

    if (!in) {
        throw std::runtime_error(
            "failed to open SA input"
        );
    }

    require_magic(in, SA_MAGIC);

    const uint32_t version = read_u32_le(in);

    if (version != FORMAT_VERSION) {
        throw std::runtime_error(
            "unsupported SA version"
        );
    }

    SaFile result;
    result.corpus_bytes = read_u64_le(in);
    result.row_count = read_u64_le(in);

    const uint32_t alphabet_size =
        read_u32_le(in);
    const uint32_t logical_sentinel =
        read_u32_le(in);
    const uint32_t internal_sentinel =
        read_u32_le(in);
    const uint32_t byte_shift =
        read_u32_le(in);
    const uint64_t payload_bytes =
        read_u64_le(in);
    const uint64_t payload_checksum =
        read_u64_le(in);

    if (
        alphabet_size != ALPHABET_SIZE
        || logical_sentinel != LOGICAL_SENTINEL
        || internal_sentinel != INTERNAL_SENTINEL
        || byte_shift != BYTE_SHIFT
    ) {
        throw std::runtime_error(
            "SA symbol mapping mismatch"
        );
    }

    if (
        result.row_count
        > static_cast<uint64_t>(
            std::numeric_limits<size_t>::max()
        )
    ) {
        throw std::runtime_error(
            "SA row count too large"
        );
    }

    const uint64_t expected_payload_bytes =
        result.row_count * sizeof(uint32_t);

    if (payload_bytes != expected_payload_bytes) {
        throw std::runtime_error(
            "SA payload size mismatch"
        );
    }

    result.rows.resize(
        static_cast<size_t>(result.row_count)
    );

    for (uint32_t &value : result.rows) {
        value = read_u32_le(in);
    }

    if (
        checksum_u32(result.rows)
        != payload_checksum
    ) {
        throw std::runtime_error(
            "SA payload checksum mismatch"
        );
    }

    require_no_trailing_data(in);

    return result;
}

int main(int argc, char **argv) {
    try {
        if (argc != 4) {
            std::cerr
                << "Usage: build_bwt_binary_v1 "
                << "<corpus.bin> <sa.bin> <bwt.bin>\n";
            return 1;
        }

        const fs::path corpus_path = argv[1];
        const fs::path sa_path = argv[2];
        const fs::path output_path = argv[3];

        const std::vector<uint8_t> corpus =
            read_file_bytes(corpus_path);
        const SaFile sa = load_sa(sa_path);

        const uint64_t expected_rows =
            static_cast<uint64_t>(
                corpus.size()
            ) + 1;

        if (
            sa.corpus_bytes != corpus.size()
            || sa.row_count != expected_rows
            || sa.rows.size() != expected_rows
        ) {
            throw std::runtime_error(
                "corpus and SA identity mismatch"
            );
        }

        std::vector<uint16_t> bwt;
        bwt.reserve(sa.rows.size());

        size_t sentinel_count = 0;

        for (
            size_t row = 0;
            row < sa.rows.size();
            ++row
        ) {
            const uint32_t suffix_offset =
                sa.rows[row];

            if (suffix_offset > corpus.size()) {
                throw std::runtime_error(
                    "SA offset outside corpus"
                );
            }

            if (suffix_offset == 0) {
                bwt.push_back(
                    static_cast<uint16_t>(
                        LOGICAL_SENTINEL
                    )
                );
                ++sentinel_count;
            } else {
                bwt.push_back(
                    static_cast<uint16_t>(
                        corpus[
                            static_cast<size_t>(
                                suffix_offset - 1
                            )
                        ]
                    )
                );
            }
        }

        if (sentinel_count != 1) {
            throw std::runtime_error(
                "BWT must contain exactly one sentinel"
            );
        }

        const uint64_t payload_bytes =
            static_cast<uint64_t>(
                bwt.size() * sizeof(uint16_t)
            );

        const uint64_t payload_checksum =
            checksum_u16(bwt);

        ensure_parent_directory(output_path);

        std::ofstream out(
            output_path,
            std::ios::binary
        );

        if (!out) {
            throw std::runtime_error(
                "failed to open BWT output"
            );
        }

        write_magic(out, BWT_MAGIC);
        write_u32_le(out, FORMAT_VERSION);
        write_u64_le(
            out,
            static_cast<uint64_t>(corpus.size())
        );
        write_u64_le(
            out,
            static_cast<uint64_t>(bwt.size())
        );
        write_u32_le(out, ALPHABET_SIZE);
        write_u32_le(out, LOGICAL_SENTINEL);
        write_u32_le(
            out,
            BWT_SYMBOL_WIDTH_BITS
        );
        write_u64_le(out, payload_bytes);
        write_u64_le(out, payload_checksum);

        for (uint16_t symbol : bwt) {
            write_u16_le(out, symbol);
        }

        std::cout
            << "{"
            << "\"ok\":true,"
            << "\"format\":\"GLYPH_BWT_BINARY_V1\","
            << "\"corpus_bytes\":"
            << corpus.size() << ","
            << "\"bwt_rows\":"
            << bwt.size() << ","
            << "\"logical_sentinel\":"
            << LOGICAL_SENTINEL << ","
            << "\"sentinel_count\":"
            << sentinel_count
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
