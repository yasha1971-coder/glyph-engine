#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace glyph::binary_v1 {

namespace fs = std::filesystem;

inline constexpr uint32_t FORMAT_VERSION = 1;
inline constexpr uint32_t ALPHABET_SIZE = 257;
inline constexpr uint32_t LOGICAL_SENTINEL = 256;
inline constexpr uint32_t INTERNAL_SENTINEL = 0;
inline constexpr uint32_t BYTE_SHIFT = 1;
inline constexpr uint32_t BWT_SYMBOL_WIDTH_BITS = 16;

inline constexpr std::array<char, 8> SA_MAGIC = {
    'G', 'L', 'Y', 'S', 'A', 'B', '1', '\0'
};

inline constexpr std::array<char, 8> BWT_MAGIC = {
    'G', 'L', 'Y', 'B', 'W', 'T', '1', '\0'
};

inline constexpr std::array<char, 8> FM_MAGIC = {
    'G', 'L', 'Y', 'F', 'M', 'B', '1', '\0'
};

inline void ensure_parent_directory(const fs::path &path) {
    const fs::path parent = path.parent_path();

    if (!parent.empty()) {
        fs::create_directories(parent);
    }
}

inline std::vector<uint8_t> read_file_bytes(
    const fs::path &path
) {
    std::ifstream in(path, std::ios::binary);

    if (!in) {
        throw std::runtime_error(
            "failed to open input: " + path.string()
        );
    }

    in.seekg(0, std::ios::end);
    const std::streamsize size = in.tellg();

    if (size < 0) {
        throw std::runtime_error(
            "failed to get input size: " + path.string()
        );
    }

    in.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(
        static_cast<size_t>(size)
    );

    if (
        size > 0
        && !in.read(
            reinterpret_cast<char *>(data.data()),
            size
        )
    ) {
        throw std::runtime_error(
            "failed to read input: " + path.string()
        );
    }

    return data;
}

inline void write_exact(
    std::ofstream &out,
    const void *data,
    size_t size
) {
    if (size == 0) {
        return;
    }

    out.write(
        static_cast<const char *>(data),
        static_cast<std::streamsize>(size)
    );

    if (!out) {
        throw std::runtime_error(
            "failed while writing binary output"
        );
    }
}

inline void read_exact(
    std::ifstream &in,
    void *data,
    size_t size
) {
    if (size == 0) {
        return;
    }

    in.read(
        static_cast<char *>(data),
        static_cast<std::streamsize>(size)
    );

    if (!in) {
        throw std::runtime_error(
            "failed while reading binary input"
        );
    }
}

inline void write_u16_le(
    std::ofstream &out,
    uint16_t value
) {
    const uint8_t bytes[2] = {
        static_cast<uint8_t>(value & 0xffu),
        static_cast<uint8_t>((value >> 8u) & 0xffu),
    };

    write_exact(out, bytes, sizeof(bytes));
}

inline void write_u32_le(
    std::ofstream &out,
    uint32_t value
) {
    const uint8_t bytes[4] = {
        static_cast<uint8_t>(value & 0xffu),
        static_cast<uint8_t>((value >> 8u) & 0xffu),
        static_cast<uint8_t>((value >> 16u) & 0xffu),
        static_cast<uint8_t>((value >> 24u) & 0xffu),
    };

    write_exact(out, bytes, sizeof(bytes));
}

inline void write_u64_le(
    std::ofstream &out,
    uint64_t value
) {
    uint8_t bytes[8];

    for (unsigned int i = 0; i < 8; ++i) {
        bytes[i] = static_cast<uint8_t>(
            (value >> (8u * i)) & 0xffu
        );
    }

    write_exact(out, bytes, sizeof(bytes));
}

inline uint16_t read_u16_le(std::ifstream &in) {
    uint8_t bytes[2];
    read_exact(in, bytes, sizeof(bytes));

    return static_cast<uint16_t>(
        static_cast<uint16_t>(bytes[0])
        | (
            static_cast<uint16_t>(bytes[1])
            << 8u
        )
    );
}

inline uint32_t read_u32_le(std::ifstream &in) {
    uint8_t bytes[4];
    read_exact(in, bytes, sizeof(bytes));

    return
        static_cast<uint32_t>(bytes[0])
        | (
            static_cast<uint32_t>(bytes[1])
            << 8u
        )
        | (
            static_cast<uint32_t>(bytes[2])
            << 16u
        )
        | (
            static_cast<uint32_t>(bytes[3])
            << 24u
        );
}

inline uint64_t read_u64_le(std::ifstream &in) {
    uint8_t bytes[8];
    read_exact(in, bytes, sizeof(bytes));

    uint64_t value = 0;

    for (unsigned int i = 0; i < 8; ++i) {
        value |= (
            static_cast<uint64_t>(bytes[i])
            << (8u * i)
        );
    }

    return value;
}

inline void write_magic(
    std::ofstream &out,
    const std::array<char, 8> &magic
) {
    write_exact(out, magic.data(), magic.size());
}

inline void require_magic(
    std::ifstream &in,
    const std::array<char, 8> &expected
) {
    std::array<char, 8> actual{};
    read_exact(in, actual.data(), actual.size());

    if (actual != expected) {
        throw std::runtime_error(
            "binary format magic mismatch"
        );
    }
}

class Fnv1a64 {
public:
    void update_byte(uint8_t value) {
        value_ ^= static_cast<uint64_t>(value);
        value_ *= 1099511628211ull;
    }

    void update_u16_le(uint16_t value) {
        update_byte(
            static_cast<uint8_t>(value & 0xffu)
        );
        update_byte(
            static_cast<uint8_t>(
                (value >> 8u) & 0xffu
            )
        );
    }

    void update_u32_le(uint32_t value) {
        for (unsigned int i = 0; i < 4; ++i) {
            update_byte(
                static_cast<uint8_t>(
                    (value >> (8u * i)) & 0xffu
                )
            );
        }
    }

    void update_u64_le(uint64_t value) {
        for (unsigned int i = 0; i < 8; ++i) {
            update_byte(
                static_cast<uint8_t>(
                    (value >> (8u * i)) & 0xffu
                )
            );
        }
    }

    [[nodiscard]] uint64_t value() const {
        return value_;
    }

private:
    uint64_t value_ = 14695981039346656037ull;
};

inline uint64_t checksum_u16(
    const std::vector<uint16_t> &values
) {
    Fnv1a64 hash;

    for (uint16_t value : values) {
        hash.update_u16_le(value);
    }

    return hash.value();
}

inline uint64_t checksum_u32(
    const std::vector<uint32_t> &values
) {
    Fnv1a64 hash;

    for (uint32_t value : values) {
        hash.update_u32_le(value);
    }

    return hash.value();
}

inline void require_no_trailing_data(
    std::ifstream &in
) {
    char extra = 0;

    if (in.read(&extra, 1)) {
        throw std::runtime_error(
            "unexpected trailing binary data"
        );
    }

    if (!in.eof()) {
        throw std::runtime_error(
            "failed while checking binary EOF"
        );
    }
}

} // namespace glyph::binary_v1
