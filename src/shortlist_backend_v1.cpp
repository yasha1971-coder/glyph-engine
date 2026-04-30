#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

struct DensePosting {
    uint32_t cid;
    uint32_t cnt;
};

struct RareAnchor {
    uint64_t sig;
    uint32_t cnt;
    uint32_t df;
};

struct ChunkState {
    uint32_t dense_present = 0;
    uint32_t dense_total = 0;
    uint32_t rare_present = 0;
    double rare_score = 0.0;
};

static uint32_t read_u32(std::istream& in) {
    uint32_t x = 0;
    in.read(reinterpret_cast<char*>(&x), 4);
    return x;
}

static uint64_t read_u64_8bytes(std::istream& in) {
    uint64_t x = 0;
    in.read(reinterpret_cast<char*>(&x), 8);
    return x;
}

static uint64_t sig8_to_u64(const uint8_t* p) {
    uint64_t x = 0;
    for (int i = 0; i < 8; ++i) {
        x |= (uint64_t)p[i] << (8 * i);
    }
    return x;
}

struct DenseIndex {
    std::unordered_map<uint64_t, std::vector<DensePosting>> inv;
};

struct RareIndex {
    std::unordered_map<uint32_t, std::unordered_map<uint64_t, RareAnchor>> by_chunk;
};

static DenseIndex load_dense(const std::string& path) {
    DenseIndex idx;
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open dense file");

    char magic[8];
    in.read(magic, 8);
    uint32_t nkeys = read_u32(in);

    for (uint32_t i = 0; i < nkeys; ++i) {
        uint64_t sig = read_u64_8bytes(in);
        uint32_t n = read_u32(in);
        auto& vec = idx.inv[sig];
        vec.reserve(n);
        for (uint32_t j = 0; j < n; ++j) {
            DensePosting p;
            p.cid = read_u32(in);
            p.cnt = read_u32(in);
            vec.push_back(p);
        }
    }
    return idx;
}

static RareIndex load_rare(const std::string& path) {
    RareIndex idx;
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open rare file");

    char magic[8];
    in.read(magic, 8);
    uint32_t nchunks = read_u32(in);

    for (uint32_t i = 0; i < nchunks; ++i) {
        uint32_t cid = read_u32(in);
        uint32_t n = read_u32(in);
        auto& cmap = idx.by_chunk[cid];
        for (uint32_t j = 0; j < n; ++j) {
            uint64_t sig = read_u64_8bytes(in);
            RareAnchor a;
            a.sig = sig;
            a.cnt = read_u32(in);
            a.df = read_u32(in);
            cmap[sig] = a;
        }
    }
    return idx;
}

struct Candidate {
    uint32_t cid;
    double score;
};

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: shortlist_backend_v1 <dense.bin> <rare.bin> <request.bin> <response.bin>\n";
        return 1;
    }

    const std::string dense_path = argv[1];
    const std::string rare_path = argv[2];
    const std::string req_path = argv[3];
    const std::string resp_path = argv[4];

    auto dense = load_dense(dense_path);
    auto rare = load_rare(rare_path);

    std::ifstream req(req_path, std::ios::binary);
    if (!req) throw std::runtime_error("failed to open request");

    char magic[8] = {0};
    req.read(magic, 8); // "SLREQV1\0"
    uint32_t nfrag = read_u32(req);
    uint32_t topk  = read_u32(req);

    std::vector<std::string> frags;
    frags.reserve(nfrag);
    for (uint32_t i = 0; i < nfrag; ++i) {
        uint32_t len = read_u32(req);
        std::string s(len, '\0');
        req.read(s.data(), len);
        frags.push_back(std::move(s));
    }

    std::unordered_map<uint32_t, ChunkState> scores;

    for (const auto& frag : frags) {
        std::unordered_map<uint32_t, bool> seen_dense;
        std::unordered_map<uint32_t, bool> seen_rare;

        for (size_t i = 0; i + 8 <= frag.size(); ++i) {
            uint64_t sig = sig8_to_u64(reinterpret_cast<const uint8_t*>(frag.data() + i));

            auto dit = dense.inv.find(sig);
            if (dit != dense.inv.end()) {
                for (const auto& p : dit->second) {
                    scores[p.cid].dense_total += p.cnt;
                    seen_dense[p.cid] = true;
                }
            }

            for (auto& [cid, cmap] : rare.by_chunk) {
                auto rit = cmap.find(sig);
                if (rit != cmap.end()) {
                    const auto& a = rit->second;
                    scores[cid].rare_score += double(a.cnt) / double(a.df + 1.0);
                    seen_rare[cid] = true;
                }
            }
        }

        for (auto& [cid, _] : seen_dense) scores[cid].dense_present += 1;
        for (auto& [cid, _] : seen_rare) scores[cid].rare_present += 1;
    }

    std::vector<Candidate> ranked;
    ranked.reserve(scores.size());
    for (const auto& [cid, s] : scores) {
        double score =
            double(s.rare_present) * 5.0 +
            double(s.dense_present) * 2.0 +
            s.rare_score;
        ranked.push_back({cid, score});
    }

    std::sort(ranked.begin(), ranked.end(), [](const Candidate& a, const Candidate& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.cid < b.cid;
    });

    if (ranked.size() > topk) ranked.resize(topk);

    std::ofstream out(resp_path, std::ios::binary);
    if (!out) throw std::runtime_error("failed to open response");

    out.write("SLRESP1\0", 8);
    uint32_t n = static_cast<uint32_t>(ranked.size());
    out.write(reinterpret_cast<const char*>(&n), 4);
    for (const auto& c : ranked) {
        out.write(reinterpret_cast<const char*>(&c.cid), 4);
        out.write(reinterpret_cast<const char*>(&c.score), 8);
    }

    return 0;
}