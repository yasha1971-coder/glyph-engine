// External benchmark adapter for pinned SDSL-lite, NOT a GLYPH runtime component.
// Pinned upstream: simongog/sdsl-lite c32874cb2d8524119f25f3b501526fe692df29f4
// Acknowledges continuous nonzero-byte text only; no multi-document/ACL proof.
#include <sdsl/suffix_arrays.hpp>
#include <sdsl/io.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

std::vector<uint8_t> parse_hex(const std::string& str) {
    if (str.empty() || (str.size() & 1)) throw std::runtime_error("bad hex length");
    std::vector<uint8_t> out;
    auto value = [](char c) -> int {
        if ('0' <= c && c <= '9') return c-'0';
        if ('a' <= c && c <= 'f') return c-'a'+10;
        throw std::runtime_error("noncanonical pattern hex");
    };
    for (size_t i=0; i<str.size(); i+=2)
        out.push_back(static_cast<uint8_t>(16*value(str[i]) + value(str[i+1])));
    return out;
}

void build(const std::string& corpus, const std::string& filename) {
    sdsl::csa_wt<> idx;
    const auto t0 = Clock::now();
    sdsl::construct(idx, corpus, 1);
    const double seconds = std::chrono::duration<double>(Clock::now()-t0).count();
    if (!sdsl::store_to_file(idx, filename))
        throw std::runtime_error("SDSL serialization failed");
    std::ifstream file(filename, std::ios::binary|std::ios::ate);
    if (!file) throw std::runtime_error("index missing");
    auto bytes = file.tellg();
    std::cout << "{\"format\":\"SDSL_CSA_WT_COMPARABLE_V0\","
              << "\"source_bytes\":" << (idx.size()-1)
              << ",\"index_bytes\":" << bytes
              << ",\"build_seconds\":" << std::setprecision(9) << seconds
              << "}" << std::endl;
}

void query(const std::string& filename, const std::string& patterns) {
    sdsl::csa_wt<> idx;
    const auto t0 = Clock::now();
    if (!sdsl::load_from_file(idx, filename))
        throw std::runtime_error("index load failed");
    const double seconds = std::chrono::duration<double>(Clock::now()-t0).count();
    std::cerr << "SDSL_CSA_LOAD_SECONDS=" << std::setprecision(10) << seconds << "\n";
    std::ifstream in(patterns);
    if (!in) throw std::runtime_error("pattern list missing");
    for (std::string h; std::getline(in,h); ) {
        auto pattern = parse_hex(h);
        const auto started = Clock::now();
        const auto c = sdsl::count(idx, pattern.begin(), pattern.end());
        auto offsets = sdsl::locate(idx, pattern.begin(), pattern.end());
        const double elapsed = std::chrono::duration<double>(Clock::now()-started).count();
        if (c != offsets.size()) throw std::runtime_error("count/locate disagree");
        std::sort(offsets.begin(), offsets.end());
        for (auto offset: offsets)
            if (offset >= idx.size()-1) throw std::runtime_error("terminal suffix matched");
        std::cout << "{\"pattern_hex\":\"" << h
                  << "\",\"count\":" << c << ",\"offsets\":[";
        for (size_t i=0;i<offsets.size();++i) {
            if (i) std::cout << ',';
            std::cout << offsets[i];
        }
        std::cout << "],\"query_seconds\":" << std::setprecision(12)
                  << elapsed << "}\n";
    }
}

void restore(const std::string& filename, const std::string& output) {
    sdsl::csa_wt<> idx;
    if (!sdsl::load_from_file(idx,filename))
        throw std::runtime_error("index load failed");
    std::ofstream out(output, std::ios::binary);
    if (!out) throw std::runtime_error("output not writable");
    const auto n = idx.size()-1;
    for (uint64_t offset=0;offset<n;offset+=65536) {
        uint64_t last = std::min<uint64_t>(n-1,offset+65535);
        std::string chunk = sdsl::extract(idx,offset,last);
        if (chunk.size() != last-offset+1)
            throw std::runtime_error("short extract");
        out.write(chunk.data(),chunk.size());
        if (!out) throw std::runtime_error("failed write");
    }
    std::cerr << "SDSL_CSA_BYTE_RESTORE_OK=" << n << "\n";
}

int main(int argc, char** argv) {
    try {
        if (argc==4 && std::string(argv[1])=="build") {
            build(argv[2],argv[3]); return 0;
        }
        if (argc==4 && std::string(argv[1])=="query") {
            query(argv[2],argv[3]); return 0;
        }
        if (argc==4 && std::string(argv[1])=="restore") {
            restore(argv[2],argv[3]); return 0;
        }
        std::cerr << "Usage: sdsl_baseline {build <raw> <index>|query <index> <patterns>|restore <index> <raw>}\n";
        return 2;
    } catch(const std::exception& e) {
        std::cerr << "SDSL_COMPARATOR_REJECTED: " << e.what() << "\n";
        return 3;
    }
}
