#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static uint32_t read_u32(std::ifstream &in) {
    uint32_t x;
    in.read(reinterpret_cast<char*>(&x), sizeof(x));
    if (!in) throw std::runtime_error("failed to read u32");
    return x;
}

static uint64_t read_u64(std::ifstream &in) {
    uint64_t x;
    in.read(reinterpret_cast<char*>(&x), sizeof(x));
    if (!in) throw std::runtime_error("failed to read u64");
    return x;
}

static void write_u32(std::ofstream &out, uint32_t x) {
    out.write(reinterpret_cast<const char*>(&x), sizeof(x));
    if (!out) throw std::runtime_error("failed to write u32");
}

static void write_u64(std::ofstream &out, uint64_t x) {
    out.write(reinterpret_cast<const char*>(&x), sizeof(x));
    if (!out) throw std::runtime_error("failed to write u64");
}

struct FMCore {
    uint64_t bwt_bytes = 0;
    uint32_t checkpoint_step = 0;
    uint64_t num_checkpoints = 0;
    std::vector<uint64_t> C;                // 256
    std::vector<uint64_t> freq;             // 256
    std::vector<uint64_t> rank_checkpoints; // num_checkpoints * 256
    std::vector<uint8_t> bwt;               // loaded separately
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
        if (sz != fm.bwt_bytes) {
            throw std::runtime_error("bwt size mismatch vs fm_core");
        }
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

static std::vector<Range> load_request(const std::string &request_path) {
    std::ifstream in(request_path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open request: " + request_path);

    char magic[4];
    in.read(magic, 4);
    if (!in || std::string(magic, 4) != "REQ1") {
        throw std::runtime_error("bad REQ1 magic");
    }

    uint32_t num_ranges = read_u32(in);
    std::vector<Range> ranges;
    ranges.reserve(num_ranges);

    for (uint32_t i = 0; i < num_ranges; ++i) {
        Range rg;
        rg.l = read_u64(in);
        rg.r = read_u64(in);
        ranges.push_back(rg);
    }

    return ranges;
}

static uint64_t rank_symbol(const FMCore &fm, uint8_t c, uint64_t pos) {
    if (pos == 0) return 0;

    uint64_t block = pos / fm.checkpoint_step;
    uint64_t offset = pos % fm.checkpoint_step;

    uint64_t base = fm.rank_checkpoints[static_cast<size_t>(block * 256ULL + c)];
    uint64_t start = block * fm.checkpoint_step;
    uint64_t end = start + offset;

    uint64_t cnt = 0;
    for (uint64_t i = start; i < end; ++i) {
        if (fm.bwt[static_cast<size_t>(i)] == c) ++cnt;
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

static void write_response(
    const std::string &response_path,
    const std::vector<std::vector<uint64_t>> &all_positions,
    const std::vector<uint64_t> &total_steps,
    const std::vector<uint64_t> &max_steps
) {
    std::ofstream out(response_path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open response: " + response_path);

    out.write("RES1", 4);
    write_u32(out, static_cast<uint32_t>(all_positions.size()));

    for (size_t i = 0; i < all_positions.size(); ++i) {
        write_u64(out, static_cast<uint64_t>(all_positions[i].size()));
        write_u64(out, total_steps[i]);
        write_u64(out, max_steps[i]);
        for (uint64_t p : all_positions[i]) {
            write_u64(out, p);
        }
    }
}

int main(int argc, char **argv) {
    try {
        if (argc != 5) {
            std::cerr << "usage: locate_backend_v1 <fm_core.bin> <locate_core.bin> <request.bin> <response.bin>\n";
            return 2;
        }

        const std::string fm_core_path = argv[1];
        const std::string locate_core_path = argv[2];
        const std::string request_path = argv[3];
        const std::string response_path = argv[4];

        auto derive_bwt_path = [](const std::string &fm_core) {
            auto pos = fm_core.rfind('/');
            std::string dir = (pos == std::string::npos) ? "." : fm_core.substr(0, pos);
            return dir + "/../py_true/corpus2_true_backend.bwt.bin";
        };

        // practical fixed path resolution for current project layout
        std::string bwt_path = derive_bwt_path(fm_core_path);

        std::cout << "=== locate_backend_v1 ===\n";
        std::cout << "fm_core:     " << fm_core_path << "\n";
        std::cout << "locate_core: " << locate_core_path << "\n";
        std::cout << "request:     " << request_path << "\n";
        std::cout << "response:    " << response_path << "\n";
        std::cout << "bwt:         " << bwt_path << "\n";

        FMCore fm = load_fm_core(fm_core_path, bwt_path);
        LocateCore loc = load_locate_core(locate_core_path);
        std::vector<Range> ranges = load_request(request_path);

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

        write_response(response_path, all_positions, total_steps, max_steps);

        std::cout << "ranges: " << ranges.size() << "\n";
        for (size_t i = 0; i < ranges.size(); ++i) {
            std::cout << "  range[" << i << "] count=" << all_positions[i].size()
                      << " total_steps=" << total_steps[i]
                      << " max_steps=" << max_steps[i] << "\n";
        }
        std::cout << "OK\n";
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "fatal: " << e.what() << "\n";
        return 1;
    }
}