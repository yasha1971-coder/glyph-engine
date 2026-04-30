#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static uint32_t read_u32(std::istream &in) {
    uint32_t x;
    in.read(reinterpret_cast<char*>(&x), sizeof(x));
    if (!in) throw std::runtime_error("failed to read u32");
    return x;
}

static uint64_t read_u64(std::istream &in) {
    uint64_t x;
    in.read(reinterpret_cast<char*>(&x), sizeof(x));
    if (!in) throw std::runtime_error("failed to read u64");
    return x;
}

static void write_u32(std::ostream &out, uint32_t x) {
    out.write(reinterpret_cast<const char*>(&x), sizeof(x));
    if (!out) throw std::runtime_error("failed to write u32");
}

static void write_u64(std::ostream &out, uint64_t x) {
    out.write(reinterpret_cast<const char*>(&x), sizeof(x));
    if (!out) throw std::runtime_error("failed to write u64");
}

struct FMCore {
    uint64_t bwt_bytes = 0;
    uint32_t checkpoint_step = 0;
    uint64_t num_checkpoints = 0;
    std::vector<uint64_t> C;
    std::vector<uint64_t> freq;
    std::vector<uint64_t> rank_checkpoints;
    std::vector<uint8_t> bwt;
};

struct LocateCore {
    uint64_t sa_size = 0;
    uint32_t sample_step = 0;
    uint64_t sampled_count = 0;
    std::unordered_map<uint64_t, uint64_t> sampled_sa;
};

struct Range {
    uint64_t l;
    uint64_t r;
};

static FMCore load_fm_core(const std::string &fm_core_path, const std::string &bwt_path) {
    FMCore fm;

    {
        std::ifstream in(fm_core_path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open fm_core: " + fm_core_path);

        char magic[4];
        in.read(magic, 4);
        if (!in || std::string(magic, 4) != "FMV1") {
            throw std::runtime_error("bad FMV1 magic");
        }

        fm.bwt_bytes = read_u64(in);
        fm.checkpoint_step = read_u32(in);
        fm.num_checkpoints = read_u64(in);

        fm.C.resize(256);
        fm.freq.resize(256);
        for (int i = 0; i < 256; ++i) fm.C[i] = read_u64(in);
        for (int i = 0; i < 256; ++i) fm.freq[i] = read_u64(in);

        fm.rank_checkpoints.resize(static_cast<size_t>(fm.num_checkpoints) * 256ULL);
        for (uint64_t i = 0; i < fm.num_checkpoints * 256ULL; ++i) {
            fm.rank_checkpoints[static_cast<size_t>(i)] = read_u64(in);
        }
    }

    {
        std::ifstream in(bwt_path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open bwt: " + bwt_path);
        in.seekg(0, std::ios::end);
        auto sz = static_cast<uint64_t>(in.tellg());
        in.seekg(0, std::ios::beg);

        if (sz != fm.bwt_bytes) throw std::runtime_error("bwt size mismatch vs fm_core");

        fm.bwt.resize(static_cast<size_t>(sz));
        in.read(reinterpret_cast<char*>(fm.bwt.data()), static_cast<std::streamsize>(sz));
        if (!in) throw std::runtime_error("failed reading bwt");
    }

    return fm;
}

static LocateCore load_locate_core(const std::string &locate_core_path) {
    LocateCore loc;

    std::ifstream in(locate_core_path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open locate_core: " + locate_core_path);

    char magic[4];
    in.read(magic, 4);
    if (!in || std::string(magic, 4) != "LOC1") {
        throw std::runtime_error("bad LOC1 magic");
    }

    loc.sa_size = read_u64(in);
    loc.sample_step = read_u32(in);
    loc.sampled_count = read_u64(in);

    loc.sampled_sa.reserve(static_cast<size_t>(loc.sampled_count * 1.3));
    for (uint64_t i = 0; i < loc.sampled_count; ++i) {
        uint64_t idx = read_u64(in);
        uint64_t pos = read_u64(in);
        loc.sampled_sa[idx] = pos;
    }

    return loc;
}

static uint64_t rank_symbol(const FMCore &fm, uint8_t c, uint64_t pos) {
    if (pos == 0) return 0;

    uint64_t block = pos / fm.checkpoint_step;
    uint64_t offset = pos % fm.checkpoint_step;
    uint64_t base = fm.rank_checkpoints[static_cast<size_t>(block * 256ULL + c)];
    uint64_t start = block * fm.checkpoint_step;

    uint64_t cnt = 0;
    for (uint64_t i = 0; i < offset; ++i) {
        if (fm.bwt[static_cast<size_t>(start + i)] == c) ++cnt;
    }
    return base + cnt;
}

static uint64_t lf(const FMCore &fm, uint64_t i) {
    uint8_t c = fm.bwt[static_cast<size_t>(i)];
    return fm.C[c] + rank_symbol(fm, c, i);
}

static std::pair<uint64_t, uint64_t> locate_one(const FMCore &fm, const LocateCore &loc, uint64_t i) {
    uint64_t steps = 0;
    uint64_t cur = i;

    while (true) {
        auto it = loc.sampled_sa.find(cur);
        if (it != loc.sampled_sa.end()) {
            uint64_t pos = (it->second + steps) % fm.bwt_bytes;
            return {pos, steps};
        }
        cur = lf(fm, cur);
        ++steps;
    }
}

static bool read_request_from_stdin(std::vector<Range> &ranges) {
    char magic[4];
    std::cin.read(magic, 4);
    if (!std::cin) return false; // normal EOF
    if (std::string(magic, 4) != "REQ1") throw std::runtime_error("bad stdin request magic");

    uint32_t num_ranges = read_u32(std::cin);
    ranges.clear();
    ranges.reserve(num_ranges);

    for (uint32_t i = 0; i < num_ranges; ++i) {
        Range rg;
        rg.l = read_u64(std::cin);
        rg.r = read_u64(std::cin);
        ranges.push_back(rg);
    }
    return true;
}

static void write_response_to_stdout(
    const std::vector<std::vector<uint64_t>> &all_positions,
    const std::vector<uint64_t> &total_steps,
    const std::vector<uint64_t> &max_steps
) {
    std::cout.write("RES1", 4);
    write_u32(std::cout, static_cast<uint32_t>(all_positions.size()));

    for (size_t i = 0; i < all_positions.size(); ++i) {
        write_u64(std::cout, static_cast<uint64_t>(all_positions[i].size()));
        write_u64(std::cout, total_steps[i]);
        write_u64(std::cout, max_steps[i]);
        for (uint64_t p : all_positions[i]) {
            write_u64(std::cout, p);
        }
    }
    std::cout.flush();
}

int main(int argc, char **argv) {
    try {
        if (argc != 4) {
            std::cerr << "usage: locate_backend_v2 <fm_core.bin> <locate_core.bin> <bwt.bin>\n";
            return 2;
        }

        const std::string fm_core_path = argv[1];
        const std::string locate_core_path = argv[2];
        const std::string bwt_path = argv[3];

        std::cerr << "locate_backend_v2 started\n";
        std::cerr << "loading fm_core...\n";
        FMCore fm = load_fm_core(fm_core_path, bwt_path);

        std::cerr << "loading locate_core...\n";
        LocateCore loc = load_locate_core(locate_core_path);

        std::cerr << "ready\n";

        std::vector<Range> ranges;

        while (read_request_from_stdin(ranges)) {
            std::vector<std::vector<uint64_t>> all_positions;
            std::vector<uint64_t> total_steps;
            std::vector<uint64_t> max_steps;

            all_positions.reserve(ranges.size());
            total_steps.reserve(ranges.size());
            max_steps.reserve(ranges.size());

            for (const auto &rg : ranges) {
                if (rg.r < rg.l || rg.r > fm.bwt_bytes) {
                    throw std::runtime_error("bad request range");
                }

                std::vector<uint64_t> pos;
                pos.reserve(static_cast<size_t>(rg.r - rg.l));

                uint64_t steps_sum = 0;
                uint64_t steps_max = 0;

                for (uint64_t i = rg.l; i < rg.r; ++i) {
                    auto [p, s] = locate_one(fm, loc, i);
                    pos.push_back(p);
                    steps_sum += s;
                    if (s > steps_max) steps_max = s;
                }

                all_positions.push_back(std::move(pos));
                total_steps.push_back(steps_sum);
                max_steps.push_back(steps_max);
            }

            write_response_to_stdout(all_positions, total_steps, max_steps);
        }

        std::cerr << "exit\n";
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "fatal: " << e.what() << "\n";
        return 1;
    }
}