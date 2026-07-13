#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "binary_runtime_v1_format.hpp"

namespace fs = std::filesystem;
using namespace glyph::binary_v1;

struct BwtFile {
    uint64_t corpus_bytes = 0;
    uint64_t row_count = 0;
    std::vector<uint16_t> symbols;
};

struct FmFile {
    uint64_t corpus_bytes = 0;
    uint64_t row_count = 0;
    uint32_t checkpoint_step = 0;
    uint64_t num_checkpoints = 0;
    std::array<uint64_t, ALPHABET_SIZE> c_array{};
    std::vector<uint32_t> checkpoints;
};

static BwtFile load_bwt(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);

    if (!in) {
        throw std::runtime_error(
            "failed to open BWT"
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
    const uint32_t sentinel =
        read_u32_le(in);
    const uint32_t width_bits =
        read_u32_le(in);
    const uint64_t payload_bytes =
        read_u64_le(in);
    const uint64_t checksum =
        read_u64_le(in);

    if (
        alphabet_size != ALPHABET_SIZE
        || sentinel != LOGICAL_SENTINEL
        || width_bits != BWT_SYMBOL_WIDTH_BITS
    ) {
        throw std::runtime_error(
            "BWT format mismatch"
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

    if (checksum_u16(result.symbols) != checksum) {
        throw std::runtime_error(
            "BWT checksum mismatch"
        );
    }

    require_no_trailing_data(in);

    return result;
}

static FmFile load_fm(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);

    if (!in) {
        throw std::runtime_error(
            "failed to open FM"
        );
    }

    require_magic(in, FM_MAGIC);

    if (read_u32_le(in) != FORMAT_VERSION) {
        throw std::runtime_error(
            "unsupported FM version"
        );
    }

    FmFile result;
    result.corpus_bytes = read_u64_le(in);
    result.row_count = read_u64_le(in);
    result.checkpoint_step = read_u32_le(in);
    result.num_checkpoints = read_u64_le(in);

    const uint32_t alphabet_size =
        read_u32_le(in);
    const uint32_t sentinel =
        read_u32_le(in);
    const uint64_t payload_bytes =
        read_u64_le(in);
    const uint64_t expected_checksum =
        read_u64_le(in);

    if (
        alphabet_size != ALPHABET_SIZE
        || sentinel != LOGICAL_SENTINEL
        || result.checkpoint_step == 0
        || result.num_checkpoints == 0
    ) {
        throw std::runtime_error(
            "FM format parameters mismatch"
        );
    }

    for (uint64_t &value : result.c_array) {
        value = read_u64_le(in);
    }

    const uint64_t checkpoint_count =
        result.num_checkpoints
        * ALPHABET_SIZE;

    result.checkpoints.resize(
        static_cast<size_t>(checkpoint_count)
    );

    for (uint32_t &value : result.checkpoints) {
        value = read_u32_le(in);
    }

    const uint64_t expected_payload_bytes =
        ALPHABET_SIZE * sizeof(uint64_t)
        + checkpoint_count * sizeof(uint32_t);

    if (payload_bytes != expected_payload_bytes) {
        throw std::runtime_error(
            "FM payload size mismatch"
        );
    }

    Fnv1a64 actual_checksum;

    for (uint64_t value : result.c_array) {
        actual_checksum.update_u64_le(value);
    }

    for (uint32_t value : result.checkpoints) {
        actual_checksum.update_u32_le(value);
    }

    if (
        actual_checksum.value()
        != expected_checksum
    ) {
        throw std::runtime_error(
            "FM checksum mismatch"
        );
    }

    require_no_trailing_data(in);

    return result;
}

static std::vector<uint8_t> parse_query_hex(
    const std::string &hex
) {
    if (hex.empty()) {
        throw std::runtime_error("EMPTY_QUERY");
    }

    if (hex.size() % 2 != 0) {
        throw std::runtime_error(
            "query_hex must have even length"
        );
    }

    for (char character : hex) {
        const bool digit =
            character >= '0' && character <= '9';
        const bool lower_hex =
            character >= 'a' && character <= 'f';

        if (!digit && !lower_hex) {
            throw std::runtime_error(
                "query_hex must be canonical lowercase"
            );
        }
    }

    std::vector<uint8_t> result;
    result.reserve(hex.size() / 2);

    const auto nibble = [](char character) -> uint8_t {
        if (
            character >= '0'
            && character <= '9'
        ) {
            return static_cast<uint8_t>(
                character - '0'
            );
        }

        return static_cast<uint8_t>(
            character - 'a' + 10
        );
    };

    for (size_t i = 0; i < hex.size(); i += 2) {
        result.push_back(
            static_cast<uint8_t>(
                (nibble(hex[i]) << 4u)
                | nibble(hex[i + 1])
            )
        );
    }

    return result;
}

static uint64_t occ(
    const FmFile &fm,
    const BwtFile &bwt,
    uint16_t symbol,
    uint64_t position
) {
    if (symbol >= ALPHABET_SIZE) {
        throw std::runtime_error(
            "occ symbol outside alphabet"
        );
    }

    if (position > fm.row_count) {
        throw std::runtime_error(
            "occ position outside BWT"
        );
    }

    const uint64_t block =
        position / fm.checkpoint_step;

    if (block >= fm.num_checkpoints) {
        throw std::runtime_error(
            "occ checkpoint outside FM"
        );
    }

    const uint64_t start =
        block * fm.checkpoint_step;

    uint64_t count = fm.checkpoints[
        static_cast<size_t>(
            block * ALPHABET_SIZE + symbol
        )
    ];

    for (
        uint64_t index = start;
        index < position;
        ++index
    ) {
        count += (
            bwt.symbols[
                static_cast<size_t>(index)
            ] == symbol
        );
    }

    return count;
}

static std::pair<uint64_t, uint64_t>
backward_search(
    const FmFile &fm,
    const BwtFile &bwt,
    const std::vector<uint8_t> &query
) {
    uint64_t left = 0;
    uint64_t right = fm.row_count;

    for (
        auto iterator = query.rbegin();
        iterator != query.rend();
        ++iterator
    ) {
        const uint16_t symbol = *iterator;

        left =
            fm.c_array[symbol]
            + occ(fm, bwt, symbol, left);

        right =
            fm.c_array[symbol]
            + occ(fm, bwt, symbol, right);

        if (left >= right) {
            return {left, left};
        }
    }

    return {left, right};
}

int main(int argc, char **argv) {
    try {
        if (argc != 4) {
            std::cerr
                << "Usage: query_fm_binary_v1 "
                << "<fm.bin> <bwt.bin> <query_hex>\n";
            return 1;
        }

        const fs::path fm_path = argv[1];
        const fs::path bwt_path = argv[2];
        const std::string query_hex = argv[3];

        const FmFile fm = load_fm(fm_path);
        const BwtFile bwt = load_bwt(bwt_path);

        if (
            fm.corpus_bytes != bwt.corpus_bytes
            || fm.row_count != bwt.row_count
            || bwt.symbols.size() != fm.row_count
        ) {
            throw std::runtime_error(
                "FM and BWT identity mismatch"
            );
        }

        const std::vector<uint8_t> query =
            parse_query_hex(query_hex);

        const auto [left, right] =
            backward_search(fm, bwt, query);

        std::cout
            << "{"
            << "\"ok\":true,"
            << "\"format\":\"GLYPH_QUERY_BINARY_V1\","
            << "\"query_hex\":\""
            << query_hex << "\","
            << "\"query_length_bytes\":"
            << query.size() << ","
            << "\"interval\":["
            << left << "," << right << "],"
            << "\"count\":"
            << (right - left) << ","
            << "\"alphabet_size\":"
            << ALPHABET_SIZE << ","
            << "\"logical_sentinel\":"
            << LOGICAL_SENTINEL
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
