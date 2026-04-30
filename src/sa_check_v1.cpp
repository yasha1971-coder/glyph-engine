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
    if (sz % 4 != 0) throw std::runtime_error("sa file size not divisible by 4");
    std::vector<int32_t> data((size_t)(sz / 4));
    in.read(reinterpret_cast<char*>(data.data()), sz);
    return data;
}

static int cmp_suffix(const std::vector<uint8_t> &corpus, int32_t a, int32_t b, int max_check = 256) {
    const int n = (int)corpus.size();
    int steps = 0;
    while (a < n && b < n && steps < max_check) {
        if (corpus[(size_t)a] < corpus[(size_t)b]) return -1;
        if (corpus[(size_t)a] > corpus[(size_t)b]) return 1;
        ++a; ++b; ++steps;
    }
    if (steps >= max_check) return 0;
    if (a == n && b == n) return 0;
    if (a == n) return -1;
    if (b == n) return 1;
    return 0;
}

int main(int argc, char **argv) {
    try {
        if (argc != 3) {
            std::cerr << "usage: sa_check_v1 <corpus.bin> <sa.bin>\n";
            return 2;
        }

        auto corpus = read_bytes(argv[1]);
        auto sa = read_i32(argv[2]);

        std::cout << "=== sa_check_v1 ===\n";
        std::cout << "corpus_bytes: " << corpus.size() << "\n";
        std::cout << "sa_entries:   " << sa.size() << "\n";

        if (sa.size() != corpus.size()) {
            std::cerr << "FAIL: sa size != corpus size\n";
            return 3;
        }

        bool range_ok = true;
        for (size_t i = 0; i < sa.size(); ++i) {
            if (sa[i] < 0 || (uint64_t)sa[i] >= corpus.size()) {
                std::cerr << "FAIL: out-of-range SA at i=" << i << " val=" << sa[i] << "\n";
                range_ok = false;
                break;
            }
        }
        if (!range_ok) return 4;

        std::cout << "range_check: OK\n";

        bool local_sorted_ok = true;
        size_t bad_i = 0;
        for (size_t i = 1; i < sa.size(); ++i) {
            int c = cmp_suffix(corpus, sa[i - 1], sa[i]);
            if (c > 0) {
                local_sorted_ok = false;
                bad_i = i;
                break;
            }
        }

        if (!local_sorted_ok) {
            std::cerr << "FAIL: suffix order violation at i=" << bad_i
                      << " prev=" << sa[bad_i - 1]
                      << " curr=" << sa[bad_i] << "\n";
            return 5;
        }

        std::cout << "order_check: OK\n";
        std::cout << "sa[0..9]: ";
        for (int i = 0; i < 10 && i < (int)sa.size(); ++i) {
            std::cout << sa[(size_t)i] << (i + 1 < 10 ? " " : "\n");
        }

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "fatal: " << e.what() << "\n";
        return 1;
    }
}