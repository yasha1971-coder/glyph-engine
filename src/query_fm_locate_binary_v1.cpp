#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "binary_runtime_v1_format.hpp"

namespace fs = std::filesystem;
using namespace glyph::binary_v1;

struct SaFile {
    uint64_t corpus_bytes = 0;
    uint64_t row_count = 0;
    std::vector<uint32_t> rows;
};

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

static SaFile load_sa(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);

    if (!in) {
        throw std::runtime_error("failed to open SA");
    }

    require_magic(in, SA_MAGIC);

    if (read_u32_le(in) != FORMAT_VERSION) {
        throw std::runtime_error("unsupported SA version");
    }

    SaFile result;
    result.corpus_bytes = read_u64_le(in);
    result.row_count = read_u64_le(in);

    const uint32_t alphabet_size = read_u32_le(in);
    const uint32_t logical_sentinel = read_u32_le(in);
    const uint32_t internal_sentinel = read_u32_le(in);
    const uint32_t byte_shift = read_u32_le(in);
    const uint64_t payload_bytes = read_u64_le(in);
    const uint64_t payload_checksum = read_u64_le(in);

    if (
        alphabet_size != ALPHABET_SIZE
        || logical_sentinel != LOGICAL_SENTINEL
        || internal_sentinel != INTERNAL_SENTINEL
        || byte_shift != BYTE_SHIFT
    ) {
        throw std::runtime_error("SA format parameters mismatch");
    }

    if (
        result.row_count
        > static_cast<uint64_t>(
            std::numeric_limits<size_t>::max()
        )
    ) {
        throw std::runtime_error("SA row count too large");
    }

    if (
        payload_bytes
        != result.row_count * sizeof(uint32_t)
    ) {
        throw std::runtime_error("SA payload size mismatch");
    }

    result.rows.resize(
        static_cast<size_t>(result.row_count)
    );

    for (uint32_t &value : result.rows) {
        value = read_u32_le(in);
    }

    if (checksum_u32(result.rows) != payload_checksum) {
        throw std::runtime_error("SA checksum mismatch");
    }

    require_no_trailing_data(in);

    if (result.row_count != result.corpus_bytes + 1) {
        throw std::runtime_error("SA cardinality mismatch");
    }

    std::vector<uint8_t> seen(
        static_cast<size_t>(result.row_count),
        0
    );

    for (uint32_t offset : result.rows) {
        if (offset >= result.row_count) {
            throw std::runtime_error("SA offset outside range");
        }

        if (seen[offset] != 0) {
            throw std::runtime_error("duplicate SA offset");
        }

        seen[offset] = 1;
    }

    if (
        result.rows.empty()
        || result.rows.front() != result.corpus_bytes
    ) {
        throw std::runtime_error(
            "terminal suffix is not first SA row"
        );
    }

    return result;
}

static BwtFile load_bwt(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);

    if (!in) {
        throw std::runtime_error("failed to open BWT");
    }

    require_magic(in, BWT_MAGIC);

    if (read_u32_le(in) != FORMAT_VERSION) {
        throw std::runtime_error("unsupported BWT version");
    }

    BwtFile result;
    result.corpus_bytes = read_u64_le(in);
    result.row_count = read_u64_le(in);

    const uint32_t alphabet_size = read_u32_le(in);
    const uint32_t sentinel = read_u32_le(in);
    const uint32_t width_bits = read_u32_le(in);
    const uint64_t payload_bytes = read_u64_le(in);
    const uint64_t checksum = read_u64_le(in);

    if (
        alphabet_size != ALPHABET_SIZE
        || sentinel != LOGICAL_SENTINEL
        || width_bits != BWT_SYMBOL_WIDTH_BITS
    ) {
        throw std::runtime_error("BWT format mismatch");
    }

    if (
        payload_bytes
        != result.row_count * sizeof(uint16_t)
    ) {
        throw std::runtime_error("BWT payload size mismatch");
    }

    result.symbols.resize(
        static_cast<size_t>(result.row_count)
    );

    size_t sentinel_count = 0;

    for (uint16_t &symbol : result.symbols) {
        symbol = read_u16_le(in);

        if (symbol > LOGICAL_SENTINEL) {
            throw std::runtime_error(
                "BWT symbol outside alphabet"
            );
        }

        if (symbol == LOGICAL_SENTINEL) {
            ++sentinel_count;
        }
    }

    if (sentinel_count != 1) {
        throw std::runtime_error(
            "BWT must contain exactly one sentinel"
        );
    }

    if (checksum_u16(result.symbols) != checksum) {
        throw std::runtime_error("BWT checksum mismatch");
    }

    require_no_trailing_data(in);

    return result;
}

static FmFile load_fm(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);

    if (!in) {
        throw std::runtime_error("failed to open FM");
    }

    require_magic(in, FM_MAGIC);

    if (read_u32_le(in) != FORMAT_VERSION) {
        throw std::runtime_error("unsupported FM version");
    }

    FmFile result;
    result.corpus_bytes = read_u64_le(in);
    result.row_count = read_u64_le(in);
    result.checkpoint_step = read_u32_le(in);
    result.num_checkpoints = read_u64_le(in);

    const uint32_t alphabet_size = read_u32_le(in);
    const uint32_t sentinel = read_u32_le(in);
    const uint64_t payload_bytes = read_u64_le(in);
    const uint64_t expected_checksum = read_u64_le(in);

    if (
        alphabet_size != ALPHABET_SIZE
        || sentinel != LOGICAL_SENTINEL
        || result.checkpoint_step == 0
        || result.num_checkpoints == 0
    ) {
        throw std::runtime_error("FM format mismatch");
    }

    for (uint64_t &value : result.c_array) {
        value = read_u64_le(in);
    }

    if (
        result.num_checkpoints
        > std::numeric_limits<size_t>::max()
            / ALPHABET_SIZE
    ) {
        throw std::runtime_error(
            "FM checkpoint table too large"
        );
    }

    const uint64_t checkpoint_count =
        result.num_checkpoints * ALPHABET_SIZE;

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
        throw std::runtime_error("FM payload size mismatch");
    }

    Fnv1a64 checksum;

    for (uint64_t value : result.c_array) {
        checksum.update_u64_le(value);
    }

    for (uint32_t value : result.checkpoints) {
        checksum.update_u32_le(value);
    }

    if (checksum.value() != expected_checksum) {
        throw std::runtime_error("FM checksum mismatch");
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

    const auto nibble = [](char character) -> uint8_t {
        if (
            character >= '0'
            && character <= '9'
        ) {
            return static_cast<uint8_t>(
                character - '0'
            );
        }

        if (
            character >= 'a'
            && character <= 'f'
        ) {
            return static_cast<uint8_t>(
                character - 'a' + 10
            );
        }

        throw std::runtime_error(
            "query_hex must be canonical lowercase"
        );
    };

    std::vector<uint8_t> result;
    result.reserve(hex.size() / 2);

    for (size_t index = 0; index < hex.size(); index += 2) {
        result.push_back(
            static_cast<uint8_t>(
                (nibble(hex[index]) << 4u)
                | nibble(hex[index + 1])
            )
        );
    }

    return result;
}

static uint64_t parse_max_offsets(
    const std::string &text
) {
    if (text.empty() || text.front() == '-') {
        throw std::runtime_error("invalid max_offsets");
    }

    size_t consumed = 0;
    const unsigned long long value =
        std::stoull(text, &consumed, 10);

    if (consumed != text.size()) {
        throw std::runtime_error("invalid max_offsets");
    }

    return static_cast<uint64_t>(value);
}

static uint64_t occ(
    const FmFile &fm,
    const BwtFile &bwt,
    uint16_t symbol,
    uint64_t position
) {
    if (
        symbol >= ALPHABET_SIZE
        || position > fm.row_count
    ) {
        throw std::runtime_error("invalid occ request");
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

static bool byte_check(
    const std::vector<uint8_t> &corpus,
    const std::vector<uint8_t> &query,
    uint64_t offset
) {
    if (offset > corpus.size()) {
        return false;
    }

    if (query.size() > corpus.size() - offset) {
        return false;
    }

    return std::equal(
        query.begin(),
        query.end(),
        corpus.begin() + static_cast<size_t>(offset)
    );
}

int main(int argc, char **argv) {
    try {
        if (argc != 6 && argc != 7) {
            std::cerr
                << "Usage: query_fm_locate_binary_v1 "
                << "<fm.bin> <bwt.bin> <sa.bin> "
                << "<corpus.bin> <query_hex> [max_offsets]\n";
            return 1;
        }

        const fs::path fm_path = argv[1];
        const fs::path bwt_path = argv[2];
        const fs::path sa_path = argv[3];
        const fs::path corpus_path = argv[4];
        const std::string query_hex = argv[5];

        const bool bounded_request = argc == 7;
        const uint64_t max_offsets =
            bounded_request
                ? parse_max_offsets(argv[6])
                : std::numeric_limits<uint64_t>::max();

        const std::vector<uint8_t> corpus =
            read_file_bytes(corpus_path);
        const SaFile sa = load_sa(sa_path);
        const BwtFile bwt = load_bwt(bwt_path);
        const FmFile fm = load_fm(fm_path);

        const uint64_t expected_rows =
            static_cast<uint64_t>(corpus.size()) + 1;

        if (
            sa.corpus_bytes != corpus.size()
            || bwt.corpus_bytes != corpus.size()
            || fm.corpus_bytes != corpus.size()
            || sa.row_count != expected_rows
            || bwt.row_count != expected_rows
            || fm.row_count != expected_rows
            || sa.rows.size() != expected_rows
            || bwt.symbols.size() != expected_rows
        ) {
            throw std::runtime_error(
                "runtime file identity mismatch"
            );
        }

        const std::vector<uint8_t> query =
            parse_query_hex(query_hex);

        const auto [left, right] =
            backward_search(fm, bwt, query);

        std::vector<uint64_t> all_offsets;
        all_offsets.reserve(
            static_cast<size_t>(right - left)
        );

        for (uint64_t row = left; row < right; ++row) {
            const uint64_t offset =
                sa.rows[static_cast<size_t>(row)];

            if (offset == corpus.size()) {
                throw std::runtime_error(
                    "terminal suffix returned for byte query"
                );
            }

            if (!byte_check(corpus, query, offset)) {
                throw std::runtime_error(
                    "located coordinate failed byte check"
                );
            }

            all_offsets.push_back(offset);
        }

        std::sort(all_offsets.begin(), all_offsets.end());

        if (
            std::adjacent_find(
                all_offsets.begin(),
                all_offsets.end()
            ) != all_offsets.end()
        ) {
            throw std::runtime_error(
                "duplicate located coordinate"
            );
        }

        const uint64_t match_count = right - left;

        if (all_offsets.size() != match_count) {
            throw std::runtime_error(
                "located count differs from FM interval"
            );
        }

        const uint64_t returned_count =
            std::min<uint64_t>(
                max_offsets,
                match_count
            );

        const bool bounded =
            returned_count < match_count;

        std::cout
            << "{"
            << "\"ok\":true,"
            << "\"format\":\"GLYPH_QUERY_LOCATE_BINARY_V1\","
            << "\"runtime_profile\":\"GLYPH_BINARY_RUNTIME_V1\","
            << "\"document_count\":1,"
            << "\"query_hex\":\"" << query_hex << "\","
            << "\"query_length_bytes\":" << query.size() << ","
            << "\"interval\":[" << left << "," << right << "],"
            << "\"match_count\":" << match_count << ","
            << "\"returned_count\":" << returned_count << ","
            << "\"bounded\":" << (bounded ? "true" : "false") << ","
            << "\"offsets_complete\":" << (!bounded ? "true" : "false") << ","
            << "\"byte_check\":true,";

        if (bounded_request) {
            std::cout
                << "\"max_offsets\":"
                << max_offsets << ",";
        }

        std::cout << "\"offsets\":[";

        for (uint64_t index = 0; index < returned_count; ++index) {
            if (index != 0) {
                std::cout << ",";
            }

            std::cout << all_offsets[
                static_cast<size_t>(index)
            ];
        }

        std::cout << "],\"coordinates\":[";

        for (uint64_t index = 0; index < returned_count; ++index) {
            if (index != 0) {
                std::cout << ",";
            }

            std::cout
                << "[0,"
                << all_offsets[
                    static_cast<size_t>(index)
                ]
                << "]";
        }

        std::cout
            << "],"
            << "\"alphabet_size\":" << ALPHABET_SIZE << ","
            << "\"logical_sentinel\":" << LOGICAL_SENTINEL
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
