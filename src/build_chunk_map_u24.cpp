#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::vector<uint32_t> read_u32_binary(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open input: " + path.string());
    }

    in.seekg(0, std::ios::end);
    std::streamsize sz = in.tellg();
    if (sz < 0 || (sz % 4) != 0) {
        throw std::runtime_error("bad u32 file size: " + path.string());
    }
    in.seekg(0, std::ios::beg);

    size_t n = static_cast<size_t>(sz / 4);
    std::vector<uint32_t> data(n);
    if (!in.read(reinterpret_cast<char *>(data.data()), sz)) {
        throw std::runtime_error("failed to read file: " + path.string());
    }
    return data;
}

static std::vector<uint64_t> parse_chunk_starts_csv(const std::string &csv_path) {
    std::ifstream in(csv_path);
    if (!in) {
        throw std::runtime_error("failed to open chunk starts csv: " + csv_path);
    }

    std::vector<uint64_t> starts;
    uint64_t x;
    while (in >> x) {
        starts.push_back(x);
        if (in.peek() == ',' || in.peek() == '\n' || in.peek() == ' ') {
            in.ignore();
        }
    }

    if (starts.empty()) {
        throw std::runtime_error("empty chunk starts");
    }
    return starts;
}

static void write_u24_binary(const fs::path &path, const std::vector<uint32_t> &data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output: " + path.string());
    }

    for (size_t i = 0; i < data.size(); ++i) {
        uint32_t v = data[i];
        if (v > 0xFFFFFFu) {
            throw std::runtime_error("chunk id too large for u24 at i=" + std::to_string(i));
        }
        uint8_t b[3];
        b[0] = static_cast<uint8_t>(v & 0xFFu);
        b[1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
        b[2] = static_cast<uint8_t>((v >> 16) & 0xFFu);
        out.write(reinterpret_cast<const char *>(b), 3);
    }

    if (!out) {
        throw std::runtime_error("failed to write output: " + path.string());
    }
}

int main(int argc, char **argv) {
    try {
        if (argc != 5) {
            std::cerr << "Usage: build_chunk_map <sa.bin> <chunk_starts.csv> <corpus_bytes> <output_chunk_map.bin>\n";
            return 1;
        }

        fs::path sa_path = argv[1];
        std::string chunk_starts_csv = argv[2];
        uint64_t corpus_bytes = std::stoull(argv[3]);
        fs::path out_path = argv[4];

        std::cout << "============================================\n";
        std::cout << " BUILD CHUNK MAP\n";
        std::cout << "============================================\n";
        std::cout << "sa:           " << sa_path << "\n";
        std::cout << "chunk_starts: " << chunk_starts_csv << "\n";
        std::cout << "corpus_bytes: " << corpus_bytes << "\n";
        std::cout << "output:       " << out_path << "\n";

        std::vector<uint32_t> sa = read_u32_binary(sa_path);
        std::vector<uint64_t> chunk_starts = parse_chunk_starts_csv(chunk_starts_csv);

        if (sa.size() != corpus_bytes) {
            throw std::runtime_error("sa size != corpus_bytes");
        }

        std::vector<uint32_t> chunk_map(sa.size(), 0);

        for (size_t i = 0; i < sa.size(); ++i) {
            uint64_t pos = static_cast<uint64_t>(sa[i]);

            auto it = std::upper_bound(chunk_starts.begin(), chunk_starts.end(), pos);
            if (it == chunk_starts.begin()) {
                throw std::runtime_error("failed to map SA position to chunk");
            }
            size_t chunk_id = static_cast<size_t>((it - chunk_starts.begin()) - 1);
            chunk_map[i] = static_cast<uint32_t>(chunk_id);
        }

        fs::create_directories(out_path.parent_path());
        write_u24_binary(out_path, chunk_map);

        std::cout << "chunk_map_ok\n";
        std::cout << "entries: " << chunk_map.size() << "\n";
        std::cout << "saved: " << out_path << "\n";
        std::cout << "done\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 2;
    }
}