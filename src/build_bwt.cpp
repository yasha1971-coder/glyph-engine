#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::vector<uint8_t> read_file_bytes(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open input: " + path.string());
    }

    in.seekg(0, std::ios::end);
    std::streamsize sz = in.tellg();
    if (sz < 0) {
        throw std::runtime_error("failed to get size: " + path.string());
    }
    in.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(static_cast<size_t>(sz));
    if (!in.read(reinterpret_cast<char *>(data.data()), sz)) {
        throw std::runtime_error("failed to read input: " + path.string());
    }
    return data;
}

static std::vector<uint32_t> read_u32_binary(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open SA input: " + path.string());
    }

    in.seekg(0, std::ios::end);
    std::streamsize sz = in.tellg();
    if (sz < 0 || (sz % 4) != 0) {
        throw std::runtime_error("bad SA file size: " + path.string());
    }
    in.seekg(0, std::ios::beg);

    size_t n = static_cast<size_t>(sz / 4);
    std::vector<uint32_t> data(n);
    if (!in.read(reinterpret_cast<char *>(data.data()), sz)) {
        throw std::runtime_error("failed to read SA file: " + path.string());
    }
    return data;
}

static void write_bytes(const fs::path &path, const std::vector<uint8_t> &data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output: " + path.string());
    }

    out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!out) {
        throw std::runtime_error("failed to write output: " + path.string());
    }
}

static void validate_sizes(size_t corpus_n, size_t sa_n) {
    if (corpus_n != sa_n) {
        throw std::runtime_error(
            "size mismatch: corpus bytes = " + std::to_string(corpus_n) +
            ", sa entries = " + std::to_string(sa_n)
        );
    }
}

int main(int argc, char **argv) {
    try {
        if (argc != 5) {
            std::cerr << "Usage: build_bwt <input_corpus.bin> <input_sa.bin> <output_bwt.bin> <sentinel>\n";
            return 1;
        }

        fs::path corpus_path = argv[1];
        fs::path sa_path = argv[2];
        fs::path bwt_path = argv[3];
        int sentinel_int = std::stoi(argv[4]);

        if (sentinel_int < 0 || sentinel_int > 255) {
            throw std::runtime_error("sentinel must be in [0,255]");
        }
        uint8_t sentinel = static_cast<uint8_t>(sentinel_int);

        std::cout << "============================================\n";
        std::cout << " BUILD TRUE BWT\n";
        std::cout << "============================================\n";
        std::cout << "corpus:   " << corpus_path << "\n";
        std::cout << "sa:       " << sa_path << "\n";
        std::cout << "output:   " << bwt_path << "\n";
        std::cout << "sentinel: " << static_cast<int>(sentinel) << "\n";

        std::vector<uint8_t> text = read_file_bytes(corpus_path);
        std::vector<uint32_t> sa = read_u32_binary(sa_path);

        validate_sizes(text.size(), sa.size());

        size_t n = text.size();
        std::vector<uint8_t> bwt(n, 0);

        for (size_t i = 0; i < n; ++i) {
            uint32_t s = sa[i];
            if (s == 0) {
                bwt[i] = sentinel;
            } else {
                bwt[i] = text[static_cast<size_t>(s - 1)];
            }
        }

        fs::create_directories(bwt_path.parent_path());
        write_bytes(bwt_path, bwt);

        std::cout << "bwt_ok\n";
        std::cout << "saved: " << bwt_path << "\n";
        std::cout << "done\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 2;
    }
}