#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

struct DenseIndex {
    std::unordered_map<uint64_t, std::vector<DensePosting>> inv;
};

struct RareIndex {
    std::unordered_map<uint32_t, std::unordered_map<uint64_t, RareAnchor>> by_chunk;
};

struct Candidate {
    uint32_t cid;
    double score;
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

static void write_f64(std::ostream& out, double x) {
    out.write(reinterpret_cast<const char*>(&x), 8);
}

static uint64_t sig8_to_u64(const uint8_t* p) {
    uint64_t x = 0;
    for (int i = 0; i < 8; ++i) {
        x |= (uint64_t)p[i] << (8 * i);
    }
    return x;
}

static DenseIndex load_dense(const std::string& path) {
    DenseIndex idx;
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open dense file: " + path);

    char magic[8];
    in.read(magic, 8);
    uint32_t nkeys = read_u32(in);

    idx.inv.reserve(nkeys);

    for (uint32_t i = 0; i < nkeys; ++i) {
        uint64_t sig = read_u64(in);
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
    if (!in) throw std::runtime_error("failed to open rare file: " + path);

    char magic[8];
    in.read(magic, 8);
    uint32_t nchunks = read_u32(in);

    idx.by_chunk.reserve(nchunks);

    for (uint32_t i = 0; i < nchunks; ++i) {
        uint32_t cid = read_u32(in);
        uint32_t n = read_u32(in);
        auto& cmap = idx.by_chunk[cid];
        cmap.reserve(n);

        for (uint32_t j = 0; j < n; ++j) {
            uint64_t sig = read_u64(in);
            RareAnchor a;
            a.sig = sig;
            a.cnt = read_u32(in);
            a.df = read_u32(in);
            cmap[sig] = a;
        }
    }

    return idx;
}

static bool read_exact(std::istream& in, char* buf, std::size_t n) {
    in.read(buf, n);
    return static_cast<std::size_t>(in.gcount()) == n;
}

static bool read_request(std::istream& in, std::vector<std::string>& frags, uint32_t& topk) {
    char magic[8];
    if (!read_exact(in, magic, 8)) return false;

    if (std::string(magic, 8) != std::string("SLREQV1\0", 8)) {
        throw std::runtime_error("bad request magic");
    }

    uint32_t nfrag = read_u32(in);
    topk = read_u32(in);

    frags.clear();
    frags.reserve(nfrag);

    for (uint32_t i = 0; i < nfrag; ++i) {
        uint32_t len = read_u32(in);
        std::string s(len, '\0');
        if (!read_exact(in, s.data(), len)) {
            throw std::runtime_error("truncated request fragment");
        }
        frags.push_back(std::move(s));
    }

    return true;
}

static void write_response(std::ostream& out, const std::vector<Candidate>& ranked) {
    out.write("SLRESP1\0", 8);
    uint32_t n = static_cast<uint32_t>(ranked.size());
    write_u32(out, n);
    for (const auto& c : ranked) {
        write_u32(out, c.cid);
        write_f64(out, c.score);
    }
    out.flush();
}

static std::vector<Candidate> run_shortlist(
    const DenseIndex& dense,
    const RareIndex& rare,
    const std::vector<std::string>& frags,
    uint32_t topk
) {
    std::unordered_map<uint32_t, ChunkState> scores;

    for (const auto& frag : frags) {
        std::unordered_set<uint32_t> seen_dense;
        std::unordered_set<uint32_t> seen_rare;

        if (frag.size() < 8) continue;

        for (size_t i = 0; i + 8 <= frag.size(); ++i) {
            uint64_t sig = sig8_to_u64(reinterpret_cast<const uint8_t*>(frag.data() + i));

            auto dit = dense.inv.find(sig);
            if (dit != dense.inv.end()) {
                for (const auto& p : dit->second) {
                    scores[p.cid].dense_total += p.cnt;
                    seen_dense.insert(p.cid);
                }
            }

            for (const auto& kv : rare.by_chunk) {
                uint32_t cid = kv.first;
                const auto& cmap = kv.second;
                auto rit = cmap.find(sig);
                if (rit != cmap.end()) {
                    const auto& a = rit->second;
                    scores[cid].rare_score += double(a.cnt) / double(a.df + 1.0);
                    seen_rare.insert(cid);
                }
            }
        }

        for (uint32_t cid : seen_dense) scores[cid].dense_present += 1;
        for (uint32_t cid : seen_rare) scores[cid].rare_present += 1;
    }

    std::vector<Candidate> ranked;
    ranked.reserve(scores.size());

    for (const auto& kv : scores) {
        uint32_t cid = kv.first;
        const auto& s = kv.second;
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
    return ranked;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: shortlist_backend_v2 <dense.bin> <rare.bin>\n";
        return 1;
    }

    const std::string dense_path = argv[1];
    const std::string rare_path = argv[2];

    std::cerr << "shortlist_backend_v2 started\n";
    std::cerr << "loading dense...\n";
    DenseIndex dense = load_dense(dense_path);
    std::cerr << "loading rare...\n";
    RareIndex rare = load_rare(rare_path);
    std::cerr << "ready\n";

    std::vector<std::string> frags;
    uint32_t topk = 0;

    while (true) {
        bool ok = read_request(std::cin, frags, topk);
        if (!ok) break;

        auto ranked = run_shortlist(dense, rare, frags, topk);
        write_response(std::cout, ranked);
    }

    std::cerr << "exit\n";
    return 0;
}