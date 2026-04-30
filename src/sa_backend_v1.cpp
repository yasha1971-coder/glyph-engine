#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if __has_include(<libsais.h>)
  #include <libsais.h>
  #define HAVE_LIBSAIS 1
#else
  #define HAVE_LIBSAIS 0
#endif

#if __has_include(<divsufsort.h>)
  #include <divsufsort.h>
  #define HAVE_DIVSUFSORT 1
#else
  #define HAVE_DIVSUFSORT 0
#endif

namespace fs = std::filesystem;

static std::vector<uint8_t> read_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open input file: " + path);
    }
    in.seekg(0, std::ios::end);
    std::streamsize sz = in.tellg();
    in.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(static_cast<size_t>(sz));
    if (!in.read(reinterpret_cast<char *>(data.data()), sz)) {
        throw std::runtime_error("failed to read input file: " + path);
    }
    return data;
}

static void write_file_i32(const std::string &path, const std::vector<int32_t> &data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot open output file: " + path);
    }
    out.write(reinterpret_cast<const char *>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(int32_t)));
    if (!out) {
        throw std::runtime_error("failed to write output file: " + path);
    }
}

int main(int argc, char **argv) {
    try {
        if (argc != 3) {
            std::cerr << "usage: sa_backend_v1 <corpus.bin> <out_sa.bin>\n";
            return 2;
        }

        const std::string corpus_path = argv[1];
        const std::string out_path = argv[2];

        std::cout << "=== sa_backend_v1 ===\n";
        std::cout << "input:  " << corpus_path << "\n";
        std::cout << "output: " << out_path << "\n";

        auto t0 = std::chrono::steady_clock::now();
        std::vector<uint8_t> corpus = read_file(corpus_path);
        auto t1 = std::chrono::steady_clock::now();

        if (corpus.empty()) {
            throw std::runtime_error("empty corpus");
        }
        if (corpus.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("corpus too large for int32 SA backend");
        }

        const int32_t n = static_cast<int32_t>(corpus.size());
        std::vector<int32_t> sa(static_cast<size_t>(n), 0);

        std::cout << "bytes: " << n << "\n";
        std::cout << "load_sec: "
                  << std::chrono::duration<double>(t1 - t0).count() << "\n";

#if HAVE_LIBSAIS
        std::cout << "backend: libsais\n";
        auto t2 = std::chrono::steady_clock::now();
        int rc = libsais(corpus.data(), sa.data(), n, 0, nullptr);
        auto t3 = std::chrono::steady_clock::now();
        if (rc < 0) {
            std::cerr << "libsais failed, rc=" << rc << "\n";
            return 3;
        }
        std::cout << "sa_build_sec: "
                  << std::chrono::duration<double>(t3 - t2).count() << "\n";
#elif HAVE_DIVSUFSORT
        std::cout << "backend: divsufsort\n";
        auto t2 = std::chrono::steady_clock::now();
        saint_t rc = divsufsort(
            reinterpret_cast<const sauchar_t *>(corpus.data()),
            reinterpret_cast<saidx_t *>(sa.data()),
            static_cast<saidx_t>(n)
        );
        auto t3 = std::chrono::steady_clock::now();
        if (rc != 0) {
            std::cerr << "divsufsort failed, rc=" << rc << "\n";
            return 4;
        }
        std::cout << "sa_build_sec: "
                  << std::chrono::duration<double>(t3 - t2).count() << "\n";
#else
        std::cerr << "No SA backend headers found.\n";
        std::cerr << "Need one of: libsais.h or divsufsort.h\n";
        return 5;
#endif

        auto t4 = std::chrono::steady_clock::now();
        write_file_i32(out_path, sa);
        auto t5 = std::chrono::steady_clock::now();

        std::cout << "save_sec: "
                  << std::chrono::duration<double>(t5 - t4).count() << "\n";
        std::cout << "total_sec: "
                  << std::chrono::duration<double>(t5 - t0).count() << "\n";

        std::cout << "sa[0..4]: ";
        for (int i = 0; i < 5 && i < n; ++i) {
            std::cout << sa[static_cast<size_t>(i)] << (i + 1 < 5 ? " " : "\n");
        }

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "fatal: " << e.what() << "\n";
        return 1;
    }
}
