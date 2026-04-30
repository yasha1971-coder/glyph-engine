#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

struct Candidate {
    uint32_t cid;
    uint32_t score;
};

static uint32_t read_u32(std::istream& in) {
    uint32_t x = 0;
    in.read(reinterpret_cast<char*>(&x), 4);
    return x;
}

static uint64_t read_u64(std::istream& in) {
    uint64_t x = 0;
    in.read(reinterpret_cast<char*>(&x), 8);
    return x;
}

static void write_u32(std::ostream& out, uint32_t x) {
    out.write(reinterpret_cast<const char*>(&x), 4);
}

struct ChunkMap {
    uint64_t sa_len = 0;
    uint32_t chunk_size = 0;
    uint32_t num_chunks = 0;
    std::vector<uint32_t> map;   // map[sa_rank] = chunk_id
};

static ChunkMap load_chunk_map(const std::string& path) {
    ChunkMap cm;
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open chunk_map: " + path);

    char magic[8];
    in.read(magic, 8);
    if (std::string(magic, 8) != std::string("CHMAPV1\0", 8)) {
        throw std::runtime_error("bad chunk_map magic");
    }

    cm.sa_len = read_u64(in);
    cm.chunk_size = read_u32(in);
    cm.num_chunks = read_u32(in);

    cm.map.resize(cm.sa_len);
    in.read(reinterpret_cast<char*>(cm.map.data()), cm.sa_len * sizeof(uint32_t));
    if (!in) throw std::runtime_error("failed while reading chunk_map payload");

    return cm;
}

struct RangeReq {
    uint32_t l;
    uint32_t r;
    uint32_t weight;
};

static bool read_request(std::istream& in, std::vector<RangeReq>& ranges, uint32_t& topk) {
    char magic[8];
    in.read(magic, 8);
    if (in.gcount() == 0) return false;
    if (in.gcount() != 8) throw std::runtime_error("truncated request header");
    if (std::string(magic, 8) != std::string("FMHREQ1\0", 8)) {
        throw std::runtime_error("bad request magic");
    }

    uint32_t nranges = read_u32(in);
    topk = read_u32(in);

    ranges.clear();
    ranges.reserve(nranges);

    for (uint32_t i = 0; i < nranges; ++i) {
        RangeReq rr;
        rr.l = read_u32(in);
        rr.r = read_u32(in);
        rr.weight = read_u32(in);
        ranges.push_back(rr);
    }
    return true;
}

static void write_response(std::ostream& out, const std::vector<Candidate>& ranked) {
    out.write("FMHRES1\0", 8);
    write_u32(out, static_cast<uint32_t>(ranked.size()));
    for (const auto& c : ranked) {
        write_u32(out, c.cid);
        write_u32(out, c.score);
    }
    out.flush();
}

static std::vector<Candidate> run_histogram(
    const ChunkMap& cm,
    const std::vector<RangeReq>& ranges,
    uint32_t topk
) {
    std::vector<uint32_t> hist(cm.num_chunks, 0);

    for (const auto& rr : ranges) {
        uint64_t l = rr.l;
        uint64_t r = rr.r;
        uint32_t w = rr.weight;

        if (l >= cm.sa_len) continue;
        if (r > cm.sa_len) r = cm.sa_len;
        if (l >= r) continue;

        for (uint64_t i = l; i < r; ++i) {
             uint32_t cid = cm.map[i];
        if (cid != 0xFFFFFFFFu && cid < cm.num_chunks) {
            hist[cid] += w;
        }
        }
    }

    std::vector<Candidate> ranked;
    ranked.reserve(cm.num_chunks);

    for (uint32_t cid = 0; cid < cm.num_chunks; ++cid) {
        if (hist[cid] > 0) {
            ranked.push_back({cid, hist[cid]});
        }
    }

    std::sort(ranked.begin(), ranked.end(), [](const Candidate& a, const Candidate& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.cid < b.cid;
    });

    if (ranked.size() > topk) ranked.resize(topk);
    return ranked;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: fm_chunk_hist_backend_v1 <chunk_map.bin>\n";
        return 1;
    }

    std::cerr << "fm_chunk_hist_backend_v1 started\n";
    std::cerr << "loading chunk_map...\n";
    ChunkMap cm = load_chunk_map(argv[1]);
    std::cerr << "ready\n";

    std::vector<RangeReq> ranges;
    uint32_t topk = 0;

    while (true) {
        bool ok = read_request(std::cin, ranges, topk);
        if (!ok) break;

        auto ranked = run_histogram(cm, ranges, topk);
        write_response(std::cout, ranked);
    }

    std::cerr << "exit\n";
    return 0;
}