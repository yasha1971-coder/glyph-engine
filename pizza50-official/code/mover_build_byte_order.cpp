// Local test driver for Move-r f48c9e1726a30861afab9b681b89616b2e45eda3.
// Reuse the library's existing option to preserve unsigned input-byte order.
// Index format, search reader and library sources are unchanged.
#include <move_r/move_r.hpp>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    try {
        std::string input;
        move_r_params params{.input_file=argv[1], .mode=_suffix_array, .num_threads=1};
        params.reorder_alphabet = false;
        move_r<_locate_move> index(input, params);
        std::ofstream out(argv[2], std::ios::binary);
        index.serialize(out);
        return out ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
