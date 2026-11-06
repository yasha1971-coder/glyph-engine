// CORE-L6 research: O(r)-additional-memory run-interval LF predecessor experiment.
// This is NOT Orbit, Move-r, balanced move, or an O(1) LF implementation.
// It independently reconstructs source from exactly the same RLB2 + existing
// EF text-position SA samples as CORE-L5. No additional persistent index.
// Run starts / targets / optional byte symbols / coarse hints are TRANSIENT.
// The predecessor implementation costs O(log runs-per-bucket) per LF.
// Frozen RLB2 and previous dense LF reference remain untouched.
#define GLYPH_PARALLEL_LIBRARY_ONLY
#include "native_parallel_inverse_rlb2_ef.cpp"
#undef GLYPH_PARALLEL_LIBRARY_ONLY

class RunIntervalLF {
public:
    std::vector<uint32_t> starts,targets,bucket;
    std::vector<uint8_t> symbols;
    uint32_t bucket_width=1, max_bucket_candidates=0;
    uint32_t sentinel_row=0;
    uint64_t n=0,r=0;

    RunIntervalLF(const std::vector<uint8_t>& data,const HeaderV2& h,
                  const std::array<uint32_t,257>& C,bool direct)
      : n(h.rows),r(h.runs) {
        demand(h.rows<EF_MISSING && h.runs<=h.rows,"u32 LF geometry");
        starts.reserve(size_t(r));
        targets.reserve(size_t(r));
        if(direct) symbols.reserve(size_t(r));
        std::array<uint64_t,257> seen{};
        uint64_t cursor=0;
        uint32_t previous_sym=257;
        each_run(data,h,[&](uint32_t sym,uint64_t at,uint64_t len) {
            demand(at==cursor && len>0 && sym!=previous_sym,
                   "inconsistent run stream");
            const uint64_t target=uint64_t(C[sym])+seen[sym];
            demand(at<EF_MISSING && target<=n-len && target<EF_MISSING,
                   "run interval maps beyond LF domain");
            starts.push_back(uint32_t(at));
            targets.push_back(uint32_t(target));
            if(direct) symbols.push_back(uint8_t(sym));
            if(sym==256) sentinel_row=uint32_t(at);
            seen[sym]+=len;
            cursor+=len;
            previous_sym=sym;
        });
        demand(cursor==n && starts.size()==r && targets.size()==r,
               "LF run construction wrong cardinality");
        demand(starts.front()==0,"first run must start at zero");
        // Bounded hint allocation: no more than r entries, even for highly
        // repetitive corpora with n/r extremely large.
        const uint64_t desired=std::min<uint64_t>(r,(n+511)/512);
        demand(desired>0,"zero run hints");
        const uint64_t width=(n+desired-1)/desired;
        demand(width>0 && width<=UINT32_MAX,"hint bucket width overflow");
        bucket_width=uint32_t(width);
        const uint64_t count=(n+width-1)/width;
        demand(count<=r,"coarse hints exceed number of BWT runs");
        bucket.reserve(size_t(count));
        size_t j=0;
        for(uint64_t b=0;b<count;++b) {
            const uint64_t row=b*width;
            while(j+1<starts.size() && starts[j+1]<=row) ++j;
            bucket.push_back(uint32_t(j));
        }
        for(size_t b=0;b<bucket.size();++b) {
            const size_t lo=bucket[b];
            const size_t hi=b+1<bucket.size()?size_t(bucket[b+1])+1:starts.size();
            demand(lo<hi && hi<=starts.size(),"hint predecessor bounds");
            max_bucket_candidates=std::max(max_bucket_candidates,uint32_t(hi-lo));
        }
    }

    inline uint32_t predecessor(uint32_t row) const {
        demand(row<n,"LF predecessor row outside BWT");
        const auto index=size_t(uint64_t(row)/bucket_width);
        demand(index<bucket.size(),"LF bucket out of range");
        const uint32_t begin=bucket[index];
        const uint32_t end=index+1<bucket.size()?
            bucket[index+1]+1:uint32_t(starts.size());
        demand(begin<end && end<=starts.size(),"invalid LF candidate slice");
        // The interval containing row is the last starting row <= row.
        auto found=std::upper_bound(starts.begin()+begin,
                                    starts.begin()+end,row);
        demand(found!=starts.begin()+begin,"invalid LF predecessor");
        return uint32_t((found-starts.begin())-1);
    }
    inline uint32_t next(uint32_t row,uint32_t run) const {
        const uint64_t dest=uint64_t(targets[run])+row-starts[run];
        demand(dest<n,"LF interval transition out of range");
        return uint32_t(dest);
    }
    uint64_t measured_transient_bytes() const {
        return uint64_t(starts.capacity())*sizeof(uint32_t)+
               uint64_t(targets.capacity())*sizeof(uint32_t)+
               uint64_t(bucket.capacity())*sizeof(uint32_t)+
               uint64_t(symbols.capacity())*sizeof(uint8_t);
    }
};

#ifndef GLYPH_RUN_INTERVAL_LIBRARY_ONLY
int main(int argc,char** argv) {
    try {
        demand(argc==6,
          "usage: native_run_interval_inverse_rlb2_ef <rlb2> <ef> <new.raw> <1..16> <rank|direct>");
        const fs::path source=argv[1],efpath=argv[2],target=argv[3];
        demand(!fs::exists(target) && !fs::is_symlink(target),
               "target already exists");
        const int threads=std::stoi(argv[4]);
        demand(threads>=1 && threads<=16,"invalid thread count");
        const std::string mode=argv[5];
        demand(mode=="rank" || mode=="direct","invalid symbol mode");
        const bool direct=mode=="direct";
        const auto t0=clk::now();
        auto encoded=load_bytes(source);
        const HeaderV2 h=read_header(encoded);
        uint64_t payload_fnv=FNV_INIT;
        for(size_t i=HEADER;i<encoded.size();++i)
            payload_fnv=fnv_byte(payload_fnv,encoded[i]);
        demand(payload_fnv==h.payload_fnv,"RLB2 payload FNV mismatch");
        std::array<uint64_t,257> freq{};
        uint64_t decoded_fnv=FNV_INIT;
        each_run(encoded,h,[&](uint32_t sym,uint64_t,uint64_t len) {
            freq[sym]+=len;
            for(uint64_t k=0;k<len;++k){
                decoded_fnv=fnv_byte(decoded_fnv,uint8_t(sym));
                decoded_fnv=fnv_byte(decoded_fnv,uint8_t(sym>>8));
            }
        });
        demand(freq[256]==1 && decoded_fnv==h.bwt_fnv,
               "RLB2 decoded FNV and logical sentinel cardinality");
        std::array<uint32_t,257> C{};
        uint64_t running=1;
        for(size_t sym=0;sym<256;++sym) {
            demand(running<=UINT32_MAX,"rank geometry overflow");
            C[sym]=uint32_t(running);
            running+=freq[sym];
        }
        demand(running==h.rows,"BWT frequency sum mismatch");
        RunIntervalLF lf(encoded,h,C,direct);
        const auto t1=clk::now();
        auto ef_bytes=load_ef(efpath);
        auto samples=parse_ef_map(ef_bytes,h.rows);
        const auto t2=clk::now();
        const uint64_t n=h.corpus,step=samples.step;
        const uint64_t windows=checked_div_ceil(n,step);
        std::vector<uint8_t> raw(size_t(n),0);
        std::array<uint32_t,256> byte_starts{};
        for(size_t sym=0;sym<256;++sym)byte_starts[sym]=C[sym];
        std::atomic<bool> fail(false);
        std::mutex error_mutex;
        std::string error_reason;
        auto worker=[&](int tid) {
            try {
                const uint64_t from=windows*uint64_t(tid)/uint64_t(threads);
                const uint64_t to=windows*uint64_t(tid+1)/uint64_t(threads);
                for(uint64_t k=from;k<to && !fail.load();++k) {
                    const uint64_t a=k*step,b=std::min(n,(k+1)*step);
                    uint32_t row=(b==n)?0:samples.q_to_row[size_t(b/step)];
                    for(uint64_t pos=b;pos>a;) {
                        --pos;
                        demand(row<h.rows && row!=lf.sentinel_row,
                               "premature sentinel/row outside BWT");
                        const uint32_t run=lf.predecessor(row);
                        const uint32_t next=lf.next(row,run);
                        demand(next!=0,"unexpected terminal LF transition");
                        raw[size_t(pos)]=direct ? lf.symbols[run] :
                            byte_symbol_for_lf(next,byte_starts);
                        row=next;
                    }
                    demand(row==samples.q_to_row[size_t(a/step)],
                           "window LF did not reach adjacent text sample");
                }
            } catch(const std::exception& e) {
                fail.store(true);
                std::lock_guard<std::mutex> lock(error_mutex);
                if(error_reason.empty())error_reason=e.what();
            }
        };
        const auto t3=clk::now();
        std::vector<std::thread> pool;
        for(int thread=0;thread<threads;++thread)pool.emplace_back(worker,thread);
        for(auto& thread:pool)thread.join();
        demand(!fail.load(),"failed run-based LF window: "+error_reason);
        const auto t4=clk::now();
        store_exclusive(target,raw);
        const auto t5=clk::now();
        auto seconds=[](auto a,auto b){
            return std::chrono::duration<double>(b-a).count();
        };
        std::cout<<"{\"status\":\"RUN_INTERVAL_RLB2_EF_EXACT_INVERSE_RESEARCH_SUCCESS\""
                 <<",\"source_bytes\":"<<n
                 <<",\"rows\":"<<h.rows
                 <<",\"runs\":"<<h.runs
                 <<",\"threads\":"<<threads
                 <<",\"step\":"<<step
                 <<",\"windows\":"<<windows
                 <<",\"symbol_mode\":\""<<mode<<"\""
                 <<",\"rlb2_bytes\":"<<encoded.size()
                 <<",\"ef_bytes\":"<<ef_bytes.size()
                 <<",\"coarse_hints\":"<<lf.bucket.size()
                 <<",\"coarse_hint_width\":"<<lf.bucket_width
                 <<",\"maximum_bucket_run_candidates\":"<<lf.max_bucket_candidates
                 <<",\"transient_run_index_bytes\":"<<lf.measured_transient_bytes()
                 <<",\"dense_lf_baseline_bytes\":"<<h.rows*4ULL
                 <<",\"ephemeral_sample_map_bytes\":"<<samples.q_to_row.size()*4ULL
                 <<",\"ephemeral_source_bytes\":"<<raw.size()
                 <<",\"compressed_run_index_build_seconds\":"<<std::fixed<<std::setprecision(6)
                 <<seconds(t0,t1)
                 <<",\"ef_parse_seconds\":"<<seconds(t1,t2)
                 <<",\"decode_seconds\":"<<seconds(t3,t4)
                 <<",\"write_seconds\":"<<seconds(t4,t5)
                 <<",\"total_seconds\":"<<seconds(t0,t5)
                 <<",\"sha256_checked_by_external_driver\":false"
                 <<",\"is_orbit_or_move_r_implementation\":false"
                 <<",\"uses_original_source_or_build_sa\":false}\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"RUN_INTERVAL_LF_REJECTED: "<<e.what()<<"\n";
        return 3;
    }
}

#endif  // GLYPH_RUN_INTERVAL_LIBRARY_ONLY
