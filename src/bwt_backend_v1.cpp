#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static std::vector<uint8_t> read_bytes(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open: " + path);
    in.seekg(0, std::ios::end);
    std::streamsize sz = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> data((size_t)sz);
    in.read(reinterpret_cast<char*>(data.data()), sz);
    return data;
}

static std::vector<int32_t> read_i32(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open: " + path);
    in.seekg(0, std::ios::end);
    std::streamsize sz = in.tellg();
    in.seekg(0, std::ios::beg);
    if (sz % 4 != 0) throw std::runtime_error("file size not divisible by 4: " + path);
    std::vector<int32_t> data((size_t)(sz / 4));
    in.read(reinterpret_cast<char*>(data.data()), sz);
    return data;
}

static void write_bytes(const std::string &path, const std::vector<uint8_t> &data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open for write: " + path);
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size()));
    if (!out) throw std::runtime_error("failed writing: " + path);
}

int main(int argc, char **argv) {
    try {
        if (argc != 4) {
            std::cerr << "usage: bwt_backend_v1 <corpus.bin> <sa.bin> <out_bwt.bin>\n";
            return 2;
        }

        const std::string corpus_path = argv[1];
        const std::string sa_path = argv[2];
        const std::string out_path = argv[3];

        std::cout << "=== bwt_backend_v1 ===\n";
        std::cout << "corpus: " << corpus_path << "\n";
        std::cout << "sa:     " << sa_path << "\n";
        std::cout << "out:    " << out_path << "\n";

        auto corpus = read_bytes(corpus_path);
        auto sa = read_i32(sa_path);

        if (corpus.size() != sa.size()) {
            std::cerr << "FAIL: corpus size != sa size\n";
            return 3;
        }

        const size_t n = corpus.size();
        std::vector<uint8_t> bwt(n);

        for (size_t i = 0; i < n; ++i) {
            int32_t pos = sa[i];
            if (pos < 0 || (size_t)pos >= n) {
                std::cerr << "FAIL: invalid sa entry at i=" << i << " val=" << pos << "\n";
                return 4;
            }
            bwt[i] = (pos == 0) ? corpus[n - 1] : corpus[(size_t)pos - 1];
        }

        write_bytes(out_path, bwt);

        std::cout << "bytes: " << n << "\n";
        std::cout << "bwt[0..9]: ";
        for (size_t i = 0; i < 10 && i < bwt.size(); ++i) {
            std::cout << (int)bwt[i] << (i + 1 < 10 ? " " : "\n");
        }
        std::cout << "OK\n";
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "fatal: " << e.what() << "\n";
        return 1;
    }
}