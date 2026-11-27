// Minimal benchmark driver for Move-r f48c9e1726a30861afab9b681b89616b2e45eda3.
#include <move_r/move_r.hpp>
#include <charconv>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "build") {
        std::string input;
        move_r_params params{.input_file=argv[2], .mode=_suffix_array, .num_threads=1};
        move_r<_locate_move> index(input, params);
        std::ofstream out(argv[3], std::ios::binary);
        index.serialize(out);
        return out ? 0 : 1;
    }
    if(argc==4 && std::string(argv[1])=="extract") {
        std::ifstream in(argv[2],std::ios::binary);move_r<_locate_move> index;index.load(in);
        auto data=index.revert<std::string>(1);
        std::ofstream out(argv[3],std::ios::binary);out.write(data.data(),data.size());return out?0:1;
    }
    if (argc != 5 || std::string(argv[1]) != "query") return 2;
    std::ifstream in(argv[2], std::ios::binary);
    if (!in) return 1;
    move_r<_locate_move> index;
    index.load(in);
    if (!in) return 1;
    std::ifstream queries(argv[3]);
    if (!queries) return 1;
    const size_t limit = std::stoul(argv[4]);
    std::string hex;
    while (std::getline(queries, hex)) {
        if (hex.empty() || hex.size() % 2) return 2;
        auto query = index.query();
        bool found = true;
        for (size_t end = hex.size(); end > 0; end -= 2) {
            unsigned value{};
            auto [p, ec] = std::from_chars(hex.data()+end-2, hex.data()+end, value, 16);
            if (ec != std::errc{} || p != hex.data()+end) return 2;
            if (!query.prepend(value)) { found = false; break; }
        }
        const auto count = found ? query.num_occ() : 0;
        std::cout << "{\"query_hex\":\"" << hex << "\",\"count\":" << count << ",\"offsets\":[";
        for (size_t i = 0; i < std::min<size_t>(count, limit); ++i) {
            if (i) std::cout << ',';
            std::cout << query.next_occ();
        }
        std::cout << "]}" << std::endl;
    }
    return 0; // preserve main implicit-success semantics when included under another name
}
