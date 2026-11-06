// GLYPH CORE-L7: bounded-buffer source materialization from RLB2 + EF.
// NO in-memory O(n) raw output allocation. One transient per-thread buffer,
// 1MiB max. Independently verified by external SHA-256 gate, not signed proof.
// NOT production: no atomic cryptographic trust pin and no 64-bit LF support.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define GLYPH_RUN_INTERVAL_LIBRARY_ONLY
#include "native_run_interval_inverse_rlb2_ef.cpp"
#undef GLYPH_RUN_INTERVAL_LIBRARY_ONLY
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

static constexpr uint64_t CHUNK_GOAL=1024ULL*1024ULL;

struct CandidateOutput {
    int fd=-1;
    std::string temp_path;
    explicit CandidateOutput(const fs::path& target,uint64_t size) {
        demand(!fs::exists(target) && !fs::is_symlink(target),
               "destination path already exists");
        std::string candidate=target.string()+".glyph-incomplete.XXXXXX";
        std::vector<char> chars(candidate.begin(),candidate.end());
        chars.push_back(0);
        fd=::mkstemp(chars.data());
        demand(fd>=0,std::string("mkstemp failed: ")+std::strerror(errno));
        temp_path=chars.data();
        if(::ftruncate(fd,off_t(size))!=0)
            throw std::runtime_error(std::string("ftruncate failed: ")+std::strerror(errno));
    }
    CandidateOutput(const CandidateOutput&)=delete;
    CandidateOutput& operator=(const CandidateOutput&)=delete;
    ~CandidateOutput() {
        if(fd>=0) ::close(fd);
        if(!temp_path.empty()) ::unlink(temp_path.c_str());
    }
    void write_chunk(uint64_t pos,const uint8_t* bytes,size_t count) {
        size_t written=0;
        while(written<count) {
            const ssize_t n=::pwrite(fd,bytes+written,count-written,off_t(pos+written));
            if(n<0 && errno==EINTR) continue;
            demand(n>0,std::string("pwrite failed: ")+std::strerror(errno));
            written+=size_t(n);
        }
    }
    void publish_candidate(const fs::path& target) {
        demand(::fdatasync(fd)==0,"output data sync failure");
        demand(::close(fd)==0,"output close failure");
        fd=-1;
        // Hard-link publication refuses to overwrite an existing target.
        // It is a candidate file; only the external independent SHA check
        // may call its contents byte-verified. Does not claim a trust root.
        demand(::link(temp_path.c_str(),target.c_str())==0,
               std::string("exclusive candidate publication failed: ")+std::strerror(errno));
        demand(::unlink(temp_path.c_str())==0,"temporary name cleanup failed");
        temp_path.clear();
    }
};

int main(int argc,char** argv) {
    try {
        demand(argc==6,
            "usage: native_run_stream_inverse <RLB2> <EF> <new-file> <1..16> <rank|direct>");
        const fs::path rlb2path=argv[1],efpath=argv[2],target=argv[3];
        const int threads=std::stoi(argv[4]);
        demand(threads>=1 && threads<=16,"invalid thread count");
        const std::string mode=argv[5];
        demand(mode=="rank" || mode=="direct","unsupported symbol mode");
        const bool direct=mode=="direct";
        demand(!fs::exists(target) && !fs::is_symlink(target),
               "target pre-exists");
        const auto t0=clk::now();
        auto encoded=load_bytes(rlb2path);
        const HeaderV2 h=read_header(encoded);
        uint64_t fnv=FNV_INIT;
        for(size_t i=HEADER;i<encoded.size();++i)
            fnv=fnv_byte(fnv,encoded[i]);
        demand(fnv==h.payload_fnv,"compressed payload FNV mismatch");
        std::array<uint64_t,257> freq{};
        uint64_t decoded_fnv=FNV_INIT;
        each_run(encoded,h,[&](uint32_t sym,uint64_t,uint64_t len){
            freq[sym]+=len;
            for(uint64_t k=0;k<len;++k) {
                decoded_fnv=fnv_byte(decoded_fnv,uint8_t(sym));
                decoded_fnv=fnv_byte(decoded_fnv,uint8_t(sym>>8));
            }
        });
        demand(freq[256]==1 && decoded_fnv==h.bwt_fnv,
               "BWT decoded checksum or unique sentinel invalid");
        std::array<uint32_t,257> C{};
        uint64_t total=1;
        for(unsigned s=0;s<256;++s) {
            demand(total<=UINT32_MAX,"LF cumulative frequency overflow");
            C[s]=uint32_t(total);
            total+=freq[s];
        }
        demand(total==h.rows,"BWT frequency geometry");
        RunIntervalLF lf(encoded,h,C,direct);
        auto ef_bytes=load_ef(efpath);
        auto samples=parse_ef_map(ef_bytes,h.rows);
        demand(samples.n==h.rows && samples.step>0,"EF geometry");
        const auto t2=clk::now();
        const uint64_t n=h.corpus,step=samples.step;
        const uint64_t windows=checked_div_ceil(n,step);
        const uint64_t windows_per_chunk=std::max<uint64_t>(1,CHUNK_GOAL/step);
        const uint64_t max_chunk=std::min<uint64_t>(
            n,windows_per_chunk*step);
        demand(max_chunk<=uint64_t(1)<<32,"excessive streaming buffer");
        std::array<uint32_t,256> Cbytes{};
        for(size_t c=0;c<256;++c) Cbytes[c]=C[c];
        CandidateOutput file(target,n);
        const auto t3=clk::now();
        std::atomic<bool> failed(false);
        std::mutex reason_mutex;
        std::string reason;
        auto worker=[&](int tid) {
            try {
                const uint64_t first=windows*uint64_t(tid)/uint64_t(threads);
                const uint64_t limit=windows*uint64_t(tid+1)/uint64_t(threads);
                std::vector<uint8_t> buf;
                buf.reserve(size_t(max_chunk));
                for(uint64_t k=first;k<limit && !failed.load();) {
                    const uint64_t next_k=std::min<uint64_t>(limit,k+windows_per_chunk);
                    const uint64_t chunk_a=k*step;
                    const uint64_t chunk_b=std::min<uint64_t>(n,next_k*step);
                    demand(chunk_b>=chunk_a && chunk_b-chunk_a<=max_chunk,
                           "invalid materialization chunk geometry");
                    buf.assign(size_t(chunk_b-chunk_a),0);
                    for(uint64_t w=k;w<next_k && !failed.load();++w) {
                        const uint64_t a=w*step;
                        const uint64_t b=std::min<uint64_t>(n,(w+1)*step);
                        uint32_t row=(b==n)?0:samples.q_to_row[size_t(b/step)];
                        for(uint64_t pos=b;pos>a;) {
                            --pos;
                            demand(row<h.rows && row!=lf.sentinel_row,
                                   "premature LF sentinel or invalid row");
                            const uint32_t run=lf.predecessor(row);
                            const uint32_t next=lf.next(row,run);
                            demand(next!=0,"premature LF terminal transition");
                            buf[size_t(pos-chunk_a)]=direct ? lf.symbols[run] :
                                byte_symbol_for_lf(next,Cbytes);
                            row=next;
                        }
                        demand(row==samples.q_to_row[size_t(a/step)],
                               "LF window failed to reach adjacent sample");
                    }
                    demand(!failed.load(),"worker aborted before chunk publication");
                    file.write_chunk(chunk_a,buf.data(),buf.size());
                    k=next_k;
                }
            } catch(const std::exception& e) {
                failed.store(true);
                std::lock_guard<std::mutex> lock(reason_mutex);
                if(reason.empty()) reason=e.what();
            }
        };
        std::vector<std::thread> pool;
        for(int t=0;t<threads;++t)pool.emplace_back(worker,t);
        for(auto& t:pool)t.join();
        demand(!failed.load(),"streaming decode failure: "+reason);
        const auto t4=clk::now();
        file.publish_candidate(target);
        const auto t5=clk::now();
        auto sec=[](auto a,auto b){
            return std::chrono::duration<double>(b-a).count();
        };
        std::cout<<"{\"status\":\"CORE_L7_RUN_INTERVAL_STREAM_CANDIDATE_WRITTEN\""
                 <<",\"source_bytes\":"<<n
                 <<",\"rows\":"<<h.rows
                 <<",\"runs\":"<<h.runs
                 <<",\"step\":"<<step
                 <<",\"windows\":"<<windows
                 <<",\"threads\":"<<threads
                 <<",\"symbol_mode\":\""<<mode<<"\""
                 <<",\"lf_run_index_bytes\":"<<lf.measured_transient_bytes()
                 <<",\"physical_rlb2_bytes\":"<<encoded.size()
                 <<",\"physical_ef_bytes\":"<<ef_bytes.size()
                 <<",\"ef_sample_map_bytes\":"<<samples.q_to_row.size()*4ULL
                 <<",\"resident_raw_output_bytes\":0"
                 <<",\"max_one_worker_buffer_bytes\":"<<max_chunk
                 <<",\"all_worker_buffers_upper_bound_bytes\":"<<max_chunk*uint64_t(threads)
                 <<",\"preprocessing_seconds\":"<<std::fixed<<std::setprecision(6)<<sec(t0,t2)
                 <<",\"decode_and_pwrite_seconds\":"<<sec(t3,t4)
                 <<",\"publish_seconds\":"<<sec(t4,t5)
                 <<",\"total_seconds\":"<<sec(t0,t5)
                 <<",\"sha256_verified_by_external_driver\":false"
                 <<",\"no_second_retained_source_archive\":true"
                 <<",\"no_64_bit_lf_yet\":true"
                 <<",\"candidate_not_trusted_until_external_hash_check\":true}\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"CORE_L7_STREAM_REJECTED: "<<e.what()<<"\n";
        return 3;
    }
}
