#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint64_t RLB_HEADER_BYTES = 4 + 4 + 8 + 8 + 32;
static constexpr uint64_t RLR_HEADER_BYTES = 4 + 4 + 8 + 4 + 8 + 32;
static constexpr uint64_t RANK_RECORD_BYTES = 8 + 8 + 8 + 256ULL * 8ULL;

struct RankRecord {
    uint64_t cp_raw_pos = 0;
    uint64_t stream_offset = 0;
    uint64_t run_offset = 0;
    std::array<uint64_t, 256> counts{};
};

struct RLBWTCore {
    uint64_t raw_len = 0;
    uint64_t run_count = 0;
    uint32_t rank_step = 0;
    uint64_t record_count = 0;
    std::vector<uint8_t> rlb_data;
    std::vector<RankRecord> records;
    std::vector<uint64_t> raw_positions;
    std::array<uint64_t, 256> freq{};
    std::array<uint64_t, 256> C{};
};

static uint32_t read_u32_at(const std::vector<uint8_t>& d, size_t off) {
    if (off + 4 > d.size()) throw std::runtime_error("read_u32_at out of range");
    return (uint32_t)d[off]
        | ((uint32_t)d[off + 1] << 8)
        | ((uint32_t)d[off + 2] << 16)
        | ((uint32_t)d[off + 3] << 24);
}

static uint64_t read_u64_at(const std::vector<uint8_t>& d, size_t off) {
    if (off + 8 > d.size()) throw std::runtime_error("read_u64_at out of range");
    uint64_t x = 0;
    for (int i = 0; i < 8; ++i) x |= ((uint64_t)d[off + i]) << (8 * i);
    return x;
}

static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open: " + path);
    in.seekg(0, std::ios::end);
    auto sz = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(sz);
    if (sz) in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(sz));
    if (!in && sz) throw std::runtime_error("failed reading: " + path);
    return data;
}

static uint64_t read_varint_at(const std::vector<uint8_t>& d, size_t& off) {
    uint64_t x = 0;
    uint32_t shift = 0;

    while (true) {
        if (off >= d.size()) throw std::runtime_error("varint eof");
        uint8_t b = d[off++];

        x |= (uint64_t)(b & 0x7FULL) << shift;

        if ((b & 0x80U) == 0) break;

        shift += 7;
        if (shift >= 64) throw std::runtime_error("varint too large");
    }

    return x;
}

static std::vector<uint8_t> parse_hex(const std::string& hex) {
    std::string s;
    s.reserve(hex.size());

    for (char ch : hex) {
        if (!std::isspace(static_cast<unsigned char>(ch))) s.push_back(ch);
    }

    if (s.size() % 2 != 0) throw std::runtime_error("hex length must be even");

    auto val = [](char c) -> uint8_t {
        if ('0' <= c && c <= '9') return (uint8_t)(c - '0');
        if ('a' <= c && c <= 'f') return (uint8_t)(10 + c - 'a');
        if ('A' <= c && c <= 'F') return (uint8_t)(10 + c - 'A');
        throw std::runtime_error("bad hex character");
    };

    std::vector<uint8_t> out;
    out.reserve(s.size() / 2);

    for (size_t i = 0; i < s.size(); i += 2) {
        out.push_back((uint8_t)((val(s[i]) << 4) | val(s[i + 1])));
    }

    return out;
}

static void load_rlbwt(RLBWTCore& core, const std::string& path) {
    core.rlb_data = read_file(path);

    if (core.rlb_data.size() < RLB_HEADER_BYTES) throw std::runtime_error("RLB1 too small");
    if (std::string(reinterpret_cast<char*>(core.rlb_data.data()), 4) != "RLB1") {
        throw std::runtime_error("bad RLB1 magic");
    }

    uint32_t version = read_u32_at(core.rlb_data, 4);
    if (version != 1) throw std::runtime_error("unsupported RLB1 version");

    core.raw_len = read_u64_at(core.rlb_data, 8);
    core.run_count = read_u64_at(core.rlb_data, 16);

    core.freq.fill(0);

    size_t off = RLB_HEADER_BYTES;
    for (uint64_t i = 0; i < core.run_count; ++i) {
        if (off >= core.rlb_data.size()) throw std::runtime_error("RLB1 run symbol eof");
        uint8_t sym = core.rlb_data[off++];
        uint64_t len = read_varint_at(core.rlb_data, off);
        core.freq[sym] += len;
    }

    uint64_t sum = 0;
    for (uint64_t x : core.freq) sum += x;
    if (sum != core.raw_len) throw std::runtime_error("RLB1 freq sum != raw_len");

    uint64_t running = 0;
    for (int c = 0; c < 256; ++c) {
        core.C[c] = running;
        running += core.freq[c];
    }
}

static void load_rank(RLBWTCore& core, const std::string& path) {
    std::vector<uint8_t> d = read_file(path);

    if (d.size() < RLR_HEADER_BYTES) throw std::runtime_error("RLR1 too small");
    if (std::string(reinterpret_cast<char*>(d.data()), 4) != "RLR1") {
        throw std::runtime_error("bad RLR1 magic");
    }

    uint32_t version = read_u32_at(d, 4);
    if (version != 1) throw std::runtime_error("unsupported RLR1 version");

    uint64_t raw_len = read_u64_at(d, 8);
    core.rank_step = read_u32_at(d, 16);
    core.record_count = read_u64_at(d, 20);

    if (raw_len != core.raw_len) throw std::runtime_error("RLR1 raw_len mismatch");

    uint64_t expected = RLR_HEADER_BYTES + core.record_count * RANK_RECORD_BYTES;
    if (expected != d.size()) throw std::runtime_error("RLR1 size mismatch");

    core.records.clear();
    core.records.resize(static_cast<size_t>(core.record_count));
    core.raw_positions.clear();
    core.raw_positions.reserve(static_cast<size_t>(core.record_count));

    size_t off = RLR_HEADER_BYTES;
    for (uint64_t i = 0; i < core.record_count; ++i) {
        RankRecord rec;
        rec.cp_raw_pos = read_u64_at(d, off); off += 8;
        rec.stream_offset = read_u64_at(d, off); off += 8;
        rec.run_offset = read_u64_at(d, off); off += 8;

        for (int c = 0; c < 256; ++c) {
            rec.counts[c] = read_u64_at(d, off);
            off += 8;
        }

        core.raw_positions.push_back(rec.cp_raw_pos);
        core.records[static_cast<size_t>(i)] = rec;
    }
}

static uint64_t rank_symbol(const RLBWTCore& core, uint8_t c, uint64_t pos) {
    if (pos == 0) return 0;
    if (pos > core.raw_len) throw std::runtime_error("rank pos > raw_len");

    auto it = std::upper_bound(core.raw_positions.begin(), core.raw_positions.end(), pos);
    if (it == core.raw_positions.begin()) throw std::runtime_error("no rank record before pos");

    size_t rec_idx = static_cast<size_t>((it - core.raw_positions.begin()) - 1);
    const RankRecord& rec = core.records[rec_idx];

    uint64_t out = rec.counts[c];
    uint64_t cur = rec.cp_raw_pos;
    size_t off = static_cast<size_t>(rec.stream_offset);
    bool first = true;

    while (cur < pos) {
        if (off >= core.rlb_data.size()) throw std::runtime_error("rank scan eof");

        uint8_t sym = core.rlb_data[off++];
        uint64_t run_len = read_varint_at(core.rlb_data, off);

        uint64_t skip = first ? rec.run_offset : 0;
        first = false;

        if (skip > run_len) throw std::runtime_error("rank run_offset > run_len");

        uint64_t available = run_len - skip;
        if (available == 0) continue;

        uint64_t need = pos - cur;
        uint64_t take = std::min(available, need);

        if (sym == c) out += take;

        cur += take;

        if (take < available) break;
    }

    return out;
}

static std::pair<uint64_t, uint64_t> backward_search(const RLBWTCore& core, const std::vector<uint8_t>& pat) {
    uint64_t l = 0;
    uint64_t r = core.raw_len;

    for (auto it = pat.rbegin(); it != pat.rend(); ++it) {
        uint8_t c = *it;
        l = core.C[c] + rank_symbol(core, c, l);
        r = core.C[c] + rank_symbol(core, c, r);

        if (l >= r) break;
    }

    return {l, r};
}

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr << "usage: rlbwt_query_core_v1 <bwt.rlbwt> <bwt.rlbwt.rank> <query_hex>\n";
            return 2;
        }

        RLBWTCore core;
        load_rlbwt(core, argv[1]);
        load_rank(core, argv[2]);

        std::vector<uint8_t> query = parse_hex(argv[3]);
        auto [l, r] = backward_search(core, query);

        std::cout << "raw_bwt_bytes: " << core.raw_len << "\n";
        std::cout << "run_count: " << core.run_count << "\n";
        std::cout << "rank_step: " << core.rank_step << "\n";
        std::cout << "rank_records: " << core.record_count << "\n";
        std::cout << "fm_interval: [" << l << ", " << r << "]\n";
        std::cout << "match_count: " << (r - l) << "\n";
        std::cout << "count:    " << (r - l) << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "rlbwt_query_core_v1 error: " << e.what() << "\n";
        return 1;
    }
}
