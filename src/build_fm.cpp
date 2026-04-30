#include <array>
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

template <typename T>
static void write_one(std::ofstream &out, const T &value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(T));
    if (!out) {
        throw std::runtime_error("failed while writing output");
    }
}

int main(int argc, char **argv) {
    try {
        if (argc != 4) {
            std::cerr << "Usage: build_fm <input_bwt.bin> <output_fm.bin> <checkpoint_step>\n";
            return 1;
        }

        fs::path bwt_path = argv[1];
        fs::path fm_path = argv[2];
        uint32_t checkpoint_step = static_cast<uint32_t>(std::stoul(argv[3]));
        if (checkpoint_step == 0) {
            throw std::runtime_error("checkpoint_step must be > 0");
        }

        std::cout << "============================================\n";
        std::cout << " BUILD FM V1\n";
        std::cout << "============================================\n";
        std::cout << "bwt:             " << bwt_path << "\n";
        std::cout << "output:          " << fm_path << "\n";
        std::cout << "checkpoint_step: " << checkpoint_step << "\n";

        std::vector<uint8_t> bwt = read_file_bytes(bwt_path);
        uint64_t n = static_cast<uint64_t>(bwt.size());
        if (n == 0) {
            throw std::runtime_error("empty BWT input");
        }

        std::cout << "bwt_bytes: " << n << "\n";

        // Histogram
        std::array<uint64_t, 256> hist{};
        hist.fill(0);
        for (uint8_t c : bwt) {
            hist[c]++;
        }

        // C array: number of chars strictly smaller than c
        std::array<uint64_t, 256> C{};
        uint64_t running = 0;
        for (size_t c = 0; c < 256; ++c) {
            C[c] = running;
            running += hist[c];
        }

        // Build checkpoints
        uint64_t num_blocks = (n + checkpoint_step - 1) / checkpoint_step + 1;
        std::vector<uint32_t> checkpoints;
        checkpoints.resize(static_cast<size_t>(num_blocks) * 256);

        std::array<uint64_t, 256> counts{};
        counts.fill(0);

        uint64_t block_idx = 0;
        for (size_t c = 0; c < 256; ++c) {
            checkpoints[block_idx * 256 + c] = 0;
        }
        block_idx++;

        for (uint64_t i = 0; i < n; ++i) {
            counts[bwt[static_cast<size_t>(i)]]++;

            if ((i + 1) % checkpoint_step == 0) {
                for (size_t c = 0; c < 256; ++c) {
                    uint64_t v = counts[c];
                    if (v > UINT32_MAX) {
                        throw std::runtime_error("checkpoint count overflow uint32");
                    }
                    checkpoints[block_idx * 256 + c] = static_cast<uint32_t>(v);
                }
                block_idx++;
            }
        }

        // final partial block if needed
        if (n % checkpoint_step != 0) {
            for (size_t c = 0; c < 256; ++c) {
                uint64_t v = counts[c];
                if (v > UINT32_MAX) {
                    throw std::runtime_error("checkpoint count overflow uint32");
                }
                checkpoints[(num_blocks - 1) * 256 + c] = static_cast<uint32_t>(v);
            }
        }

        fs::create_directories(fm_path.parent_path());
        std::ofstream out(fm_path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("failed to open output: " + fm_path.string());
        }

        // Header
        const char magic[8] = {'F','M','B','I','N','v','1','\0'};
        out.write(magic, 8);
        if (!out) {
            throw std::runtime_error("failed to write magic");
        }

        write_one(out, n);
        write_one(out, checkpoint_step);
        write_one(out, num_blocks);

        // C[256]
        for (size_t c = 0; c < 256; ++c) {
            write_one(out, C[c]);
        }

        // checkpoints
        out.write(reinterpret_cast<const char *>(checkpoints.data()),
                  static_cast<std::streamsize>(checkpoints.size() * sizeof(uint32_t)));
        if (!out) {
            throw std::runtime_error("failed to write checkpoints");
        }

        std::cout << "hist_ok\n";
        std::cout << "C_ok\n";
        std::cout << "num_blocks: " << num_blocks << "\n";
        std::cout << "saved: " << fm_path << "\n";
        std::cout << "done\n";

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 2;
    }
}