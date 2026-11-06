// CORE-L5 RESEARCH ONLY: parallel inverse-BWT using *existing* source-position EF samples.
// Build with original native_inverse_rlb2.cpp in this same research directory.
// Retained inputs: frozen RLB2 and already-counted LOC-TEXT-EF; no raw source,
// SA, original BWT, RLR2, new restart file or Python runtime.
// Output is independently hash-checked by the benchmark driver.
// Known limitation: this builder retains an O(n) LF table in temporary RAM.
#define main glyph_original_rlb2_inverse_main
#include "native_inverse_rlb2.cpp"
#undef main
#include <atomic>
#include <mutex>
#include <thread>

static constexpr size_t EF_HEADER_BYTES=80;
static constexpr uint32_t EF_MISSING=0xffffffffU;

static uint64_t checked_div_ceil(uint64_t a,uint64_t b) {
    demand(b>0,"zero divisor");
    return a/b+(a%b!=0);
}
static uint64_t read_bits(const std::vector<uint8_t>& b,
                          uint64_t base,uint64_t byte_count,
                          uint64_t bit,unsigned width) {
    if(!width) return 0;
    demand(width<=32 && bit<=byte_count*8 && width<=byte_count*8-bit,
           "truncated sample bits");
    const uint64_t off=base+(bit>>3);
    unsigned shift=unsigned(bit&7);
    const unsigned bytes=(shift+width+7)/8;
    demand(off<=b.size() && bytes<=b.size()-off,
           "truncated packed data");
    uint64_t v=0;
    for(unsigned i=0;i<bytes;++i) v |= uint64_t(b[size_t(off+i)])<<(8*i);
    return (v>>shift)&((uint64_t(1)<<width)-1);
}
static std::vector<uint8_t> load_ef(const fs::path& p) {
    demand(fs::is_regular_file(p) && !fs::is_symlink(p),"EF input must be regular");
    const auto size=fs::file_size(p);
    demand(size>=EF_HEADER_BYTES && size<=(1ULL<<30),"EF input size admission");
    std::ifstream file(p,std::ios::binary);
    demand(bool(file),"cannot open EF");
    std::vector<uint8_t> content(size_t(size),0);
    file.read(reinterpret_cast<char*>(content.data()),std::streamsize(content.size()));
    demand(bool(file),"EF truncated during read");
    return content;
}
struct EFMap {
    uint64_t n=0, step=0, m=0;
    std::vector<uint32_t> q_to_row;
};
static EFMap parse_ef_map(const std::vector<uint8_t>& b,uint64_t expected_n) {
    const std::array<uint8_t,8> magic={'G','L','3','E','F','0','0','1'};
    demand(b.size()>=EF_HEADER_BYTES && std::equal(magic.begin(),magic.end(),b.begin()),
           "EF magic/size mismatch");
    const uint64_t n=u64(b,8), step=u64(b,16), m=u64(b,24), hb=u64(b,32);
    const unsigned low=b[40], width=b[41];
    const unsigned stride=unsigned(b[42])|(unsigned(b[43])<<8);
    const uint64_t lbs=u64(b,44), hbs=u64(b,52), qbs=u64(b,60), cbs=u64(b,68);
    demand(u32(b,76)==0 && stride==256 && n==expected_n &&
           n>0 && n<EF_MISSING && step>0 && m==(n-1)/step+1,
           "EF noncanonical geometry");
    const uint64_t floor_n_over_m=n/m;
    unsigned expected_low=0;
    while(expected_low<32 && (uint64_t(1)<<(expected_low+1))<=floor_n_over_m)
        ++expected_low;
    unsigned expected_width=0;
    while((uint64_t(1)<<expected_width)<m) ++expected_width;
    demand(low==expected_low && width==expected_width && low<=31 && width<=31,
           "EF wrong bitpacking widths");
    demand(hb==((n-1)>>low)+m+1 && lbs==checked_div_ceil(m*low,8) &&
           hbs==checked_div_ceil(hb,8) && qbs==checked_div_ceil(m*width,8) &&
           cbs==4*checked_div_ceil(m,256),"EF noncanonical section sizes");
    demand(lbs<=b.size() && hbs<=b.size()-lbs &&
           qbs<=b.size()-lbs-hbs && cbs<=b.size()-lbs-hbs-qbs &&
           EF_HEADER_BYTES+lbs+hbs+qbs+cbs==b.size(),"EF byte geometry mismatch");
    const uint64_t low_base=EF_HEADER_BYTES;
    const uint64_t high_base=low_base+lbs;
    const uint64_t quotient_base=high_base+hbs;
    const uint64_t cp_base=quotient_base+qbs;
    EFMap result;
    result.n=n;result.step=step;result.m=m;
    result.q_to_row.assign(size_t(m),EF_MISSING);
    uint64_t encountered=0, last_row=0;
    bool have_last=false;
    for(uint64_t high_pos=0;high_pos<hb;++high_pos) {
        if(!(b[size_t(high_base+(high_pos>>3))] & (uint8_t(1)<<(high_pos&7))))
            continue;
        demand(encountered<m && high_pos>=encountered,"EF excess set bit");
        const uint64_t lo=read_bits(b,low_base,lbs,encountered*low,low);
        const uint64_t row=((high_pos-encountered)<<low)|lo;
        const uint64_t quotient=read_bits(b,quotient_base,qbs,encountered*width,width);
        demand(row<n && (!have_last || row>last_row),"EF rows not strictly increasing");
        demand(quotient<m && quotient*step<n &&
               result.q_to_row[size_t(quotient)]==EF_MISSING,
               "EF quotient duplicate/out-of-range");
        if(encountered%256==0) {
            demand(high_pos<EF_MISSING &&
                   u32(b,size_t(cp_base+4*(encountered/256)))==high_pos,
                   "EF select checkpoint mismatch");
        }
        result.q_to_row[size_t(quotient)]=uint32_t(row);
        last_row=row;have_last=true; ++encountered;
    }
    demand(encountered==m,"EF missing samples");
    demand(std::find(result.q_to_row.begin(),result.q_to_row.end(),
                     EF_MISSING)==result.q_to_row.end(),
           "EF text sample coverage incomplete");
    if((n-1)%step==0)
        demand(result.q_to_row[size_t((n-1)/step)]==0,
               "terminal suffix is not row zero");
    return result;
}
static void store_exclusive(const fs::path& p,const std::vector<uint8_t>& data) {
    demand(!fs::exists(p) && !fs::is_symlink(p),"refuse existing output");
    std::ofstream file(p,std::ios::binary|std::ios::trunc);
    demand(bool(file),"cannot create output");
    if(!data.empty()) file.write(reinterpret_cast<const char*>(data.data()),std::streamsize(data.size()));
    file.flush();
    demand(bool(file),"output write failure");
}
#ifndef GLYPH_PARALLEL_LIBRARY_ONLY
int main(int argc,char** argv) {
    try {
        demand(argc==6,
          "usage: native_parallel_inverse_rlb2_ef <rlb2> <ef> <new.raw> <threads> <symbol_mode:rank|direct>");
        const fs::path source=argv[1], efpath=argv[2], target=argv[3];
        demand(!fs::exists(target) && !fs::is_symlink(target),"output already exists");
        const int threads=std::stoi(argv[4]);
        demand(threads>=1 && threads<=16,"research supports 1 to 16 threads");
        const std::string mode=argv[5];
        demand(mode=="rank" || mode=="direct","unsupported symbol mode");
        const bool direct=(mode=="direct");
        const auto t0=clk::now();
        std::vector<uint8_t> packed=load_bytes(source);
        const HeaderV2 h=read_header(packed);
        uint64_t physical_fnv=FNV_INIT;
        for(size_t i=HEADER;i<packed.size();++i)
            physical_fnv=fnv_byte(physical_fnv,packed[i]);
        demand(physical_fnv==h.payload_fnv,"RLB2 payload FNV mismatch");
        std::array<uint64_t,257> freq{};
        uint64_t bwt_fnv=FNV_INIT;
        each_run(packed,h,[&](uint32_t sym,uint64_t,uint64_t length) {
            freq[sym]+=length;
            for(uint64_t i=0;i<length;++i) {
                bwt_fnv=fnv_byte(bwt_fnv,uint8_t(sym));
                bwt_fnv=fnv_byte(bwt_fnv,uint8_t(sym>>8));
            }
        });
        demand(freq[256]==1 && bwt_fnv==h.bwt_fnv,
               "RLB2 BWT checksum/sentinel cardinality invalid");
        std::array<uint32_t,257> C{};
        uint64_t running=1;
        for(unsigned sym=0;sym<256;++sym) {
            demand(running<=UINT32_MAX,"LF rank geometry out of bounds");
            C[sym]=uint32_t(running);
            running+=freq[sym];
        }
        demand(running==h.rows,"wrong RLB2 histogram");
        std::vector<uint32_t> lf(size_t(h.rows),0);
        std::vector<uint8_t> symbol;
        if(direct) symbol.assign(size_t(h.rows),0);
        std::array<uint64_t,257> seen{};
        uint32_t sentinel_row=0;
        each_run(packed,h,[&](uint32_t s,uint64_t at,uint64_t len) {
            const uint64_t base=uint64_t(C[s])+seen[s];
            demand(base+len<=h.rows,"invalid LF destination");
            if(s==256) sentinel_row=uint32_t(at);
            for(uint64_t i=0;i<len;++i) {
                lf[size_t(at+i)]=uint32_t(base+i);
                if(direct) symbol[size_t(at+i)]=uint8_t(s);
            }
            seen[s]+=len;
        });
        for(unsigned s=0;s<257;++s) demand(seen[s]==freq[s],"LF histogram mismatch");
        const auto t1=clk::now();
        std::vector<uint8_t> ef_bytes=load_ef(efpath);
        EFMap samples=parse_ef_map(ef_bytes,h.rows);
        const auto t2=clk::now();
        const uint64_t n=h.corpus, step=samples.step;
        const uint64_t windows=checked_div_ceil(n,step);
        std::vector<uint8_t> raw(size_t(n),0);
        std::array<uint32_t,256> byte_starts{};
        for(size_t i=0;i<256;++i) byte_starts[i]=C[i];
        std::atomic<bool> failure(false);
        std::string failure_reason;
        std::mutex error_lock;
        auto worker=[&](int tid) {
            try {
                const uint64_t from=windows*uint64_t(tid)/uint64_t(threads);
                const uint64_t to=windows*uint64_t(tid+1)/uint64_t(threads);
                for(uint64_t k=from;k<to && !failure.load();++k) {
                    const uint64_t a=k*step, b=std::min<uint64_t>(n,(k+1)*step);
                    uint32_t row=(b==n)?0:samples.q_to_row[size_t(b/step)];
                    for(uint64_t pos=b;pos>a;) {
                        --pos;
                        demand(row<h.rows,"invalid LF transition origin");
                        demand(row!=sentinel_row,"premature logical terminal symbol");
                        const uint32_t next=lf[size_t(row)];
                        demand(next<h.rows && next!=0,
                               "premature LF cycle closure in window");
                        raw[size_t(pos)]=direct ? symbol[size_t(row)] :
                            byte_symbol_for_lf(next,byte_starts);
                        row=next;
                    }
                    demand(row==samples.q_to_row[size_t(a/step)],
                           "block LF endpoint does not reach adjacent sample");
                }
            } catch(const std::exception& e) {
                failure.store(true);
                std::lock_guard<std::mutex> lock(error_lock);
                if(failure_reason.empty()) failure_reason=e.what();
            }
        };
        const auto t3=clk::now();
        std::vector<std::thread> pool;
        for(int i=0;i<threads;++i) pool.emplace_back(worker,i);
        for(auto& t:pool) t.join();
        demand(!failure.load(),"thread/window verification failed: "+failure_reason);
        const auto t4=clk::now();
        // File is emitted only after every window's LF end boundary checked.
        store_exclusive(target,raw);
        const auto t5=clk::now();
        auto secs=[](auto a,auto b) {
            return std::chrono::duration<double>(b-a).count();
        };
        std::cout<<"{\"status\":\"PARALLEL_RLB2_EF_SOURCE_INVERSE_RESEARCH_SUCCESS\","
                 <<"\"source_bytes\":"<<n
                 <<",\"threads\":"<<threads
                 <<",\"step\":"<<step
                 <<",\"windows\":"<<windows
                 <<",\"symbol_mode\":\""<<mode<<"\""
                 <<",\"rlb2_bytes\":"<<packed.size()
                 <<",\"ef_bytes\":"<<ef_bytes.size()
                 <<",\"lf_work_bytes\":"<<lf.size()*4ULL
                 <<",\"ephemeral_sample_map_bytes\":"<<samples.q_to_row.size()*4ULL
                 <<",\"ephemeral_source_bytes\":"<<raw.size()
                 <<",\"direct_symbol_cache_bytes\":"<<symbol.size()
                 <<",\"bwt_scan_and_lf_seconds\":"<<std::fixed<<std::setprecision(6)<<secs(t0,t1)
                 <<",\"ef_parse_seconds\":"<<secs(t1,t2)
                 <<",\"thread_setup_seconds\":"<<secs(t2,t3)
                 <<",\"parallel_decode_seconds\":"<<secs(t3,t4)
                 <<",\"write_seconds\":"<<secs(t4,t5)
                 <<",\"total_seconds\":"<<secs(t0,t5)
                 <<",\"sha256_checked_by_external_driver\":false"
                 <<",\"uses_original_source_or_build_sa\":false}\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"PARALLEL_RLB2_EF_REJECTED: "<<e.what()<<"\n";
        return 3;
    }
}

#endif  // GLYPH_PARALLEL_LIBRARY_ONLY
