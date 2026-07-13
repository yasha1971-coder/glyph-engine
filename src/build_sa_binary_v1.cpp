#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "binary_runtime_v1_format.hpp"
#include "libsais.h"

namespace fs = std::filesystem;
using namespace glyph::binary_v1;

static void validate_sa(
    const std::vector<int32_t> &sa,
    size_t row_count,
    size_t terminal_offset
) {
    if (sa.size() != row_count) {
        throw std::runtime_error(
            "SA row count mismatch"
        );
    }

    std::vector<uint8_t> seen(row_count, 0);

    for (size_t row = 0; row < row_count; ++row) {
        const int32_t offset = sa[row];

        if (
            offset < 0
            || static_cast<size_t>(offset)
                >= row_count
        ) {
            throw std::runtime_error(
                "SA offset outside range at row "
                + std::to_string(row)
            );
        }

        const size_t index =
            static_cast<size_t>(offset);

        if (seen[index] != 0) {
            throw std::runtime_error(
                "duplicate SA offset at row "
                + std::to_string(row)
            );
        }

        seen[index] = 1;
    }

    if (
        sa.empty()
        || static_cast<size_t>(sa.front())
            != terminal_offset
    ) {
        throw std::runtime_error(
            "terminal suffix is not first SA row"
        );
    }
}

int main(int argc, char **argv) {
    try {
        if (argc != 3) {
            std::cerr
                << "Usage: build_sa_binary_v1 "
                << "<corpus.bin> <sa_binary_v1.bin>\n";
            return 1;
        }

        const fs::path corpus_path = argv[1];
        const fs::path output_path = argv[2];

        const std::vector<uint8_t> corpus =
            read_file_bytes(corpus_path);

        if (
            corpus.size()
            >= static_cast<size_t>(
                std::numeric_limits<int32_t>::max()
            )
        ) {
            throw std::runtime_error(
                "corpus too large for int32 SA"
            );
        }

        const size_t row_count = corpus.size() + 1;

        std::vector<int32_t> integer_text(
            row_count,
            0
        );

        for (size_t i = 0; i < corpus.size(); ++i) {
            integer_text[i] =
                static_cast<int32_t>(corpus[i])
                + static_cast<int32_t>(BYTE_SHIFT);
        }

        integer_text[corpus.size()] =
            static_cast<int32_t>(
                INTERNAL_SENTINEL
            );

        std::vector<int32_t> sa(row_count, 0);

        const int32_t rc = libsais_int(
            integer_text.data(),
            sa.data(),
            static_cast<int32_t>(row_count),
            static_cast<int32_t>(ALPHABET_SIZE),
            0
        );

        if (rc < 0) {
            throw std::runtime_error(
                "libsais_int failed with code "
                + std::to_string(rc)
            );
        }

        validate_sa(
            sa,
            row_count,
            corpus.size()
        );

        std::vector<uint32_t> serialized_sa;
        serialized_sa.reserve(row_count);

        for (int32_t value : sa) {
            serialized_sa.push_back(
                static_cast<uint32_t>(value)
            );
        }

        const uint64_t payload_bytes =
            static_cast<uint64_t>(
                serialized_sa.size()
                * sizeof(uint32_t)
            );

        const uint64_t payload_checksum =
            checksum_u32(serialized_sa);

        ensure_parent_directory(output_path);

        std::ofstream out(
            output_path,
            std::ios::binary
        );

        if (!out) {
            throw std::runtime_error(
                "failed to open SA output"
            );
        }

        write_magic(out, SA_MAGIC);
        write_u32_le(out, FORMAT_VERSION);
        write_u64_le(
            out,
            static_cast<uint64_t>(corpus.size())
        );
        write_u64_le(
            out,
            static_cast<uint64_t>(row_count)
        );
        write_u32_le(out, ALPHABET_SIZE);
        write_u32_le(out, LOGICAL_SENTINEL);
        write_u32_le(out, INTERNAL_SENTINEL);
        write_u32_le(out, BYTE_SHIFT);
        write_u64_le(out, payload_bytes);
        write_u64_le(out, payload_checksum);

        for (uint32_t value : serialized_sa) {
            write_u32_le(out, value);
        }

        std::cout
            << "{"
            << "\"ok\":true,"
            << "\"format\":\"GLYPH_SA_BINARY_V1\","
            << "\"corpus_bytes\":"
            << corpus.size() << ","
            << "\"sa_rows\":"
            << row_count << ","
            << "\"alphabet_size\":"
            << ALPHABET_SIZE << ","
            << "\"logical_sentinel\":"
            << LOGICAL_SENTINEL << ","
            << "\"terminal_suffix_row\":0"
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
