// GLYPH research: native inverse BWT from frozen, binary-safe RLB2 ONLY.
// No raw corpus, suffix array, BWT temp, LOC1/LOC2/EF, Python or RLR2
// is required for this decoder's actual byte recovery.
// NOT an accepted production parser / signed proof / constant-memory decoder.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;
static constexpr uint64_t FNV_INIT = 14695981039346656037ULL;
static constexpr uint64_t FNV_PRIME = 1099511628211ULL;
static constexpr size_t HEADER = 160;

static void demand(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
}
static uint32_t u32(const std::vector<uint8_t>& b, size_t i) {
    demand(i <= b.size() && b.size() - i >= 4, "truncated u32");
    return uint32_t(b[i]) | (uint32_t(b[i+1]) << 8)
         | (uint32_t(b[i+2]) << 16) | (uint32_t(b[i+3]) << 24);
}
static uint64_t u64(const std::vector<uint8_t>& b, size_t i) {
    demand(i <= b.size() && b.size() - i >= 8, "truncated u64");
    return uint64_t(u32(b,i)) | (uint64_t(u32(b,i+4)) << 32);
}
static uint64_t fnv_byte(uint64_t h, uint8_t b) {
    return (h ^ b) * FNV_PRIME;
}
struct HeaderV2 {
    uint32_t escape = 0;
    uint64_t corpus = 0, rows = 0, runs = 0, payload = 0;
    uint64_t bwt_fnv = 0, payload_fnv = 0;
};
static HeaderV2 read_header(const std::vector<uint8_t>& b) {
    demand(b.size() >= HEADER, "RLB2 header truncated");
    const std::array<uint8_t,8> magic={'G','L','Y','R','L','B','2',0};
    demand(std::equal(magic.begin(),magic.end(),b.begin()),"RLB2 wrong magic");
    demand(u32(b,8)==2 && u32(b,12)==HEADER,"RLB2 wrong version/header bytes");
    HeaderV2 h;
    h.corpus=u64(b,16);
    h.rows=u64(b,24);
    h.runs=u64(b,32);
    h.escape=u32(b,48);
    h.payload=u64(b,64);
    h.bwt_fnv=u64(b,80);
    h.payload_fnv=u64(b,88);
    demand(h.corpus + 1 == h.rows && h.rows>0, "RLB2 row geometry");
    demand(h.rows <= std::numeric_limits<uint32_t>::max(),
           "research LF u32 index limit: requires SA64 successor for larger corpus");
    demand(h.runs>0 && h.runs<=h.rows,"RLB2 run-count geometry");
    demand(u32(b,40)==257 && u32(b,44)==256 && h.escape<=255 &&
           u32(b,52)==1 && u32(b,56)==1 && u32(b,60)==0,
           "RLB2 wrong alphabet, sentinel, escape or canonical codecs");
    demand(u64(b,72)==h.rows*2,"RLB2 decoded BWT symbol geometry");
    demand(h.payload == uint64_t(b.size())-HEADER,"RLB2 payload file geometry");
    return h;
}
template<class Callback>
static void each_run(const std::vector<uint8_t>& b,
                     const HeaderV2& h, Callback callback) {
    size_t i=HEADER;
    uint64_t pos=0;
    int prior=-1;
    for (uint64_t r=0;r<h.runs;++r) {
        demand(i<b.size(),"truncated RLB2 run head");
        int sym=b[i++];
        if (sym==int(h.escape)) {
            demand(i<b.size(),"truncated RLB2 escape");
            uint8_t tag=b[i++];
            demand(tag==0 || tag==1,"invalid RLB2 escaped symbol");
            sym=(tag==1) ? 256 : int(h.escape);
        }
        demand(sym>=0 && sym<=256 && sym!=prior,"noncanonical repeated/invalid RLB2 run");
        uint64_t length=0;
        unsigned consumed=0;
        for (;consumed<10;++consumed) {
            demand(i<b.size(),"truncated RLB2 ULEB128");
            const uint8_t byte=b[i++];
            const uint8_t part=byte & 0x7f;
            demand(consumed<9 || part<=1,"ULEB128 overflow");
            length |= uint64_t(part) << (7*consumed);
            if (!(byte & 0x80)) {++consumed; break;}
        }
        demand(consumed>0 && consumed<=10 && length>0,"invalid RLB2 length");
        uint64_t normalized=length;
        unsigned required=1;
        while (normalized >= 128) {++required;normalized>>=7;}
        demand(consumed==required,"noncanonical RLB2 ULEB128");
        demand(length<=h.rows-pos,"RLB2 run extends beyond BWT");
        demand(sym!=256 || length==1,"logical sentinel repeated");
        callback(uint32_t(sym),pos,length);
        pos+=length;
        prior=sym;
    }
    demand(i==b.size() && pos==h.rows,"run count/payload/decoded rows mismatch");
}
static std::vector<uint8_t> load_bytes(const fs::path& p) {
    demand(fs::is_regular_file(p) && !fs::is_symlink(p),"RLB2 must be regular file");
    auto bytes=fs::file_size(p);
    demand(bytes>=HEADER && bytes <= (1ULL<<33),"research memory admission cap");
    std::ifstream in(p,std::ios::binary);
    demand(bool(in),"cannot open RLB2");
    std::vector<uint8_t> b(size_t(bytes),0);
    in.read(reinterpret_cast<char*>(b.data()),std::streamsize(b.size()));
    demand(bool(in),"truncated RLB2 read");
    return b;
}
static uint8_t byte_symbol_for_lf(uint32_t lf,
                                  const std::array<uint32_t,256>& starts) {
    demand(lf>0,"premature LF sentinel");
    auto it=std::upper_bound(starts.begin(),starts.end(),lf);
    demand(it!=starts.begin(),"LF outside byte alphabet");
    const size_t symbol=size_t(it-starts.begin())-1;
    demand(symbol<256,"invalid byte lookup");
    return uint8_t(symbol);
}

int main(int argc, char** argv) {
    try {
        demand(argc==3,"usage: native_inverse_rlb2 <frozen.rlb2> <new-output.raw>");
        const fs::path source=argv[1], target=argv[2];
        demand(!fs::exists(target) && !fs::is_symlink(target),
               "refuse preexisting output");
        const auto t0=clk::now();
        const std::vector<uint8_t> encoded=load_bytes(source);
        const auto h=read_header(encoded);
        uint64_t physical_fnv=FNV_INIT;
        for(size_t i=HEADER;i<encoded.size();++i)
            physical_fnv=fnv_byte(physical_fnv,encoded[i]);
        demand(physical_fnv==h.payload_fnv,"compressed payload FNV mismatch");
        std::array<uint64_t,257> frequencies{};
        uint64_t decoded_fnv=FNV_INIT;
        each_run(encoded,h,[&](uint32_t sym,uint64_t,uint64_t len){
            frequencies[sym]+=len;
            const uint8_t low=uint8_t(sym), high=uint8_t(sym>>8);
            for(uint64_t k=0;k<len;++k) {
                decoded_fnv=fnv_byte(decoded_fnv,low);
                decoded_fnv=fnv_byte(decoded_fnv,high);
            }
        });
        demand(decoded_fnv==h.bwt_fnv,"canonical decoded BWT FNV mismatch");
        demand(frequencies[256]==1,"logical sentinel cardinality not one");
        std::array<uint32_t,257> C{};
        C[256]=0;
        uint64_t running=1;
        for(unsigned c=0;c<256;++c) {
            demand(running<=std::numeric_limits<uint32_t>::max(),"rank C overflow");
            C[c]=uint32_t(running);
            running+=frequencies[c];
        }
        demand(running==h.rows,"BWT byte symbol histogram invalid");
        std::vector<uint32_t> lf(size_t(h.rows),0);
        std::array<uint64_t,257> observed{};
        each_run(encoded,h,[&](uint32_t sym,uint64_t row,uint64_t len) {
            uint64_t start=uint64_t(C[sym])+observed[sym];
            demand(start+len<=h.rows,"LF transition outside rows");
            for(uint64_t i=0;i<len;++i)
                lf[size_t(row+i)]=uint32_t(start+i);
            observed[sym]+=len;
        });
        for(unsigned c=0;c<257;++c)
            demand(observed[c]==frequencies[c],"LF counts disagree with BWT");
        const auto t1=clk::now();
        // C[byte] is byte-specific cumulative frequency. Symbol of BWT[row]
        // is determined by the destination LF[row], no separate BWT array.
        std::array<uint32_t,256> byte_starts{};
        for(size_t c=0;c<256;++c) byte_starts[c]=C[c];
        std::vector<uint8_t> raw(size_t(h.corpus),0);
        std::vector<uint8_t> visited(size_t((h.rows+7)/8),0);
        uint32_t row=0;
        for(uint64_t i=h.corpus;i>0;) {
            --i;
            demand(row<h.rows,"LF row out of range");
            auto& marker=visited[size_t(row>>3)];
            const uint8_t mask=uint8_t(1u<<(row&7));
            demand(!(marker&mask),"LF noncanonical multiple cycles");
            marker|=mask;
            const uint32_t next=lf[row];
            demand(next<h.rows,"LF successor outside range");
            raw[size_t(i)]=byte_symbol_for_lf(next,byte_starts);
            row=next;
        }
        demand(row<h.rows && lf[row]==0,"LF chain failed to close on sentinel");
        auto& marker=visited[size_t(row>>3)];
        const uint8_t mask=uint8_t(1u<<(row&7));
        demand(!(marker&mask),"LF sentinel already visited");
        marker|=mask;
        const auto t2=clk::now();
        std::ofstream out(target,std::ios::binary|std::ios::trunc);
        demand(bool(out),"cannot create recovered source file");
        if(!raw.empty())
            out.write(reinterpret_cast<const char*>(raw.data()),std::streamsize(raw.size()));
        out.flush();
        demand(bool(out),"failed write output");
        out.close();
        const auto t3=clk::now();
        auto seconds=[](auto a,auto b) {
            return std::chrono::duration<double>(b-a).count();
        };
        std::cout<<"{\"status\":\"NATIVE_RLB2_SELF_INDEX_RESTORE_RESEARCH_SUCCESS\","
                 <<"\"rows\":"<<h.rows<<",\"source_bytes\":"<<h.corpus
                 <<",\"rlb2_bytes\":"<<encoded.size()
                 <<",\"lf_u32_work_bytes\":"<<lf.size()*4
                 <<",\"visited_work_bytes\":"<<visited.size()
                 <<",\"source_work_bytes\":"<<raw.size()
                 <<",\"encoded_work_bytes\":"<<encoded.size()
                 <<",\"build_lf_seconds\":"<<std::fixed<<std::setprecision(6)
                 <<seconds(t0,t1)
                 <<",\"inverse_seconds\":"<<seconds(t1,t2)
                 <<",\"write_seconds\":"<<seconds(t2,t3)
                 <<",\"total_seconds\":"<<seconds(t0,t3)
                 <<",\"no_source_or_sa_input\":true,\"byte_hash_verified_by_driver\":false}\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"NATIVE_RLB2_INVERSE_REJECTED: "<<e.what()<<"\n";
        return 3;
    }
}
