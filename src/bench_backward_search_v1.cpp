// NEXT:
// replace synthetic loop
// -> parse_hex_pattern()
// -> backward_search()
// -> adaptive Occ()
// -> interval production

#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdexcept>

static uint64_t pct(
    std::vector<uint64_t> x,
    double p
){
    std::sort(
        x.begin(),
        x.end()
    );

    return x[
        size_t(
            (p/100.0)
            *
            (x.size()-1)
        )
    ];
}

// REAL PIPELINE NEXT:
//
// parse_hex_pattern()
// backward_search()
// adaptive Occ()
// interval [l,r)
// count
//
// remove fake_backward_search

static void fake_backward_search(
    uint64_t len,
    uint64_t iter,
    volatile uint64_t& sink
){

    uint64_t l=0;
    uint64_t r=1000000;

    for(
        uint64_t j=0;
        j<len;
        ++j
    ){

        uint8_t c=
            uint8_t(
                (j*17+iter)
                &255
            );

        l=
            (
                l
                ^
                (
                    uint64_t(c)
                    *
                    1315423911ull
                )
            )
            %
            (
                r+1
            );

        if(
            l>r
        ){

            std::swap(
                l,
                r
            );

        }

    }

    sink+=
    (
        r-l
    );

}

int main(
    int argc,
    char** argv
){

    try{

        if(
            argc!=4
        ){

            std::cerr
            <<
            "Usage: bench_backward_search_v1 <fm.bin> <bwt.bin> <iterations>\n";

            return 1;

        }

        uint64_t it=
            std::stoull(
                argv[3]
            );

        for(
            uint64_t len:
            {4,8,16,32}
        ){

            std::vector<uint64_t> s;

            s.reserve(
                size_t(it)
            );

            volatile
            uint64_t sink=0;

            auto bench0=
                std::chrono::
                high_resolution_clock
                ::now();

            for(
                uint64_t i=0;
                i<it;
                ++i
            ){

                auto t0=
                    std::chrono::
                    high_resolution_clock
                    ::now();

                fake_backward_search(
                    len,
                    i,
                    sink
                );

                auto t1=
                    std::chrono::
                    high_resolution_clock
                    ::now();

                s.push_back(

                    uint64_t(

                    std::chrono::
                    duration_cast<
                        std::chrono::
                        nanoseconds
                    >(
                        t1-t0
                    ).count()

                    )

                );

            }

            auto bench1=
                std::chrono::
                high_resolution_clock
                ::now();

            double sec=

            double(

            std::chrono::
            duration_cast<
                std::chrono::
                nanoseconds
            >(
                bench1-bench0
            ).count()

            )

            /

            1000000000.0;

            uint64_t qps=

            uint64_t(

                double(it)
                /
                sec

            );

            std::cout

            <<"{"

            <<"\"bench\":\"BACKWARD_SEARCH_BENCH_V1\","

            <<"\"pattern_len\":"
            <<len
            <<","

            <<"\"iterations\":"
            <<it
            <<","

            <<"\"p50_ns\":"
            <<pct(
                s,
                50
            )
            <<","

            <<"\"p95_ns\":"
            <<pct(
                s,
                95
            )
            <<","

            <<"\"p99_ns\":"
            <<pct(
                s,
                99
            )
            <<","

            <<"\"queries_per_sec\":"
            <<qps
            <<","

            <<"\"sink\":"
            <<sink

            <<"}\n";

        }

        return 0;

    }
    catch(
        const std::exception& e
    ){

        std::cerr
        <<
        "ERROR: "
        <<
        e.what()
        <<
        "\n";

        return 2;

    }

}
