// Experimental standalone format. NOT the GLYPH embedded ABI or index format.
#include <openssl/evp.h>
#include <zlib.h>
#include <array>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <memory>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

using Byte = unsigned char;
constexpr uint32_t BLOCK = 1U << 20;
constexpr std::array<Byte,8> MAGIC{'G','N','L','A','B',0,1,0};
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
struct FD {
    int n;
    explicit FD(int value): n(value) { require(n>=0,"open failed"); }
    ~FD(){ close(n); }
    FD(const FD&)=delete;
    FD& operator=(const FD&)=delete;
};
void write_all(int fd,const Byte* p,size_t n){
    while(n){ssize_t k=write(fd,p,n);if(k<0&&errno==EINTR)continue;
        require(k>0,"write failed");p+=k;n-=size_t(k);}
}
void read_all(int fd,Byte* p,size_t n){
    while(n){ssize_t k=read(fd,p,n);if(k<0&&errno==EINTR)continue;
        require(k>0,"truncated or unreadable input");p+=k;n-=size_t(k);}
}
size_t read_block(int fd,Byte* p,size_t n){
    size_t got=0;while(got<n){ssize_t k=read(fd,p+got,n-got);
        if(k<0&&errno==EINTR)continue;
        require(k>=0,"read failed");if(!k)break;got+=size_t(k);}return got;
}
void put32(Byte* p,uint32_t x){for(unsigned i=0;i<4;++i)p[i]=Byte(x>>(8*i));}
uint32_t get32(const Byte* p){uint32_t x=0;for(unsigned i=0;i<4;++i)x|=uint32_t(p[i])<<(8*i);return x;}
void put64(Byte* p,uint64_t x){for(unsigned i=0;i<8;++i)p[i]=Byte(x>>(8*i));}
uint64_t get64(const Byte* p){uint64_t x=0;for(unsigned i=0;i<8;++i)x|=uint64_t(p[i])<<(8*i);return x;}
struct Hash {
    std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> ctx{EVP_MD_CTX_new(),EVP_MD_CTX_free};
    Hash(){require(bool(ctx)&&EVP_DigestInit_ex(ctx.get(),EVP_sha256(),nullptr)==1,"hash init");}
    void add(const Byte*p,size_t n){require(EVP_DigestUpdate(ctx.get(),p,n)==1,"hash update");}
    std::array<Byte,32> finish(){std::array<Byte,32> v{};unsigned n=0;
        require(EVP_DigestFinal_ex(ctx.get(),v.data(),&n)==1&&n==32,"hash finish");return v;}
};
auto digest(const Byte*p,size_t n){Hash h;h.add(p,n);return h.finish();}
struct Input {
    FD fd;
    struct stat initial{};
    explicit Input(const char* path):fd(open(path,O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK)){
        require(fstat(fd.n,&initial)==0&&S_ISREG(initial.st_mode),"input must be regular file");
    }
    void unchanged(){struct stat now{};require(fstat(fd.n,&now)==0,"stat input");
        require(now.st_size==initial.st_size && now.st_mtim.tv_sec==initial.st_mtim.tv_sec &&
        now.st_mtim.tv_nsec==initial.st_mtim.tv_nsec && now.st_ctim.tv_sec==initial.st_ctim.tv_sec &&
        now.st_ctim.tv_nsec==initial.st_ctim.tv_nsec,"input changed during operation");}
};
// Private temp on destination filesystem; hard-link publication refuses overwrite.
// Directory must be trusted, local, and support hard links + fsync.
struct Output {
    std::string target,temp,parent;
    std::unique_ptr<FD> fd;
    explicit Output(const char* path):target(path){
        auto slash=target.find_last_of('/');parent=slash==std::string::npos?".":target.substr(0,slash);
        if(parent.empty())parent="/";
        struct stat st{};require(lstat(target.c_str(),&st)!=0&&errno==ENOENT,"output already exists or inaccessible");
        temp=parent+"/.glyph-native-XXXXXX";std::vector<char> t(temp.begin(),temp.end());t.push_back(0);
        int n=mkstemp(t.data());require(n>=0,"create temporary output");temp=t.data();fd=std::make_unique<FD>(n);
    }
    ~Output(){if(!temp.empty())unlink(temp.c_str());}
    void publish(){
        require(fsync(fd->n)==0,"fsync output failed");
        require(link(temp.c_str(),target.c_str())==0,"publish refused/failed");
        require(unlink(temp.c_str())==0,"remove temp link failed");temp.clear();
        FD dir(open(parent.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC));
        require(fsync(dir.n)==0,"directory fsync failed; output may already exist");
    }
};
void pack(const char* src,const char* dst){
    Input in(src);Output out(dst);write_all(out.fd->n,MAGIC.data(),MAGIC.size());
    std::vector<Byte> raw(BLOCK),encoded(compressBound(BLOCK));Hash whole;uint64_t total=0;
    while(size_t n=read_block(in.fd.n,raw.data(),raw.size())){
        whole.add(raw.data(),n);total+=n;uLongf z=encoded.size();
        require(compress2(encoded.data(),&z,raw.data(),n,1)==Z_OK,"compress failed");
        bool compressed=z<n;auto hash=digest(raw.data(),n);std::array<Byte,44> hdr{};
        put32(hdr.data(),uint32_t(n));put32(hdr.data()+4,uint32_t(compressed?z:n));
        put32(hdr.data()+8,compressed?1:0);std::copy(hash.begin(),hash.end(),hdr.begin()+12);
        write_all(out.fd->n,hdr.data(),hdr.size());
        write_all(out.fd->n,compressed?encoded.data():raw.data(),compressed?z:n);
    }
    std::array<Byte,44> end{};put64(end.data()+4,total);auto h=whole.finish();
    std::copy(h.begin(),h.end(),end.begin()+12);write_all(out.fd->n,end.data(),end.size());
    in.unchanged();require(total==uint64_t(in.initial.st_size),"source length changed");out.publish();
    std::cout<<"OK packed bytes="<<total<<"\n";
}
struct Matcher {
    std::vector<Byte> pattern;std::vector<size_t> prefix;size_t state=0;uint64_t pos=0,count=0;
    explicit Matcher(const std::string& hex){
        require(!hex.empty()&&hex.size()%2==0&&hex.size()<=8192,"hex query length must be 1..4096 bytes");
        auto digit=[](char c)->int{if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;
            if(c>='A'&&c<='F')return c-'A'+10;
            throw std::runtime_error("invalid hex query");};
        for(size_t i=0;i<hex.size();i+=2)pattern.push_back(Byte(16*digit(hex[i])+digit(hex[i+1])));
        prefix.resize(pattern.size());for(size_t i=1,k=0;i<pattern.size();++i){
            while(k&&pattern[i]!=pattern[k])k=prefix[k-1];
            if(pattern[i]==pattern[k])++k;
            prefix[i]=k;}
    }
    void feed(const Byte*p,size_t n){for(size_t i=0;i<n;++i){
        while(state&&p[i]!=pattern[state])state=prefix[state-1];
        if(p[i]==pattern[state])++state;
        if(state==pattern.size()){++count;state=prefix[state-1];}++pos;}}
};
void consume(const char* src,const char* dst,Matcher* match){
    Input in(src);std::unique_ptr<Output> out;if(dst)out=std::make_unique<Output>(dst);
    std::array<Byte,8> magic{};read_all(in.fd.n,magic.data(),magic.size());require(magic==MAGIC,"bad format/version");
    std::vector<Byte> raw(BLOCK),encoded(BLOCK);Hash whole;uint64_t total=0;
    for(;;){std::array<Byte,44> hdr{};read_all(in.fd.n,hdr.data(),hdr.size());uint32_t n=get32(hdr.data());
        if(!n){require(get64(hdr.data()+4)==total,"total length mismatch");auto h=whole.finish();
            require(std::equal(h.begin(),h.end(),hdr.begin()+12),"whole SHA256 mismatch");
            Byte extra;require(read_block(in.fd.n,&extra,1)==0,"trailing data");break;}
        uint32_t z=get32(hdr.data()+4),codec=get32(hdr.data()+8);
        require(n<=BLOCK&&z>0&&z<=BLOCK&&codec<=1,"invalid block bounds or codec");
        read_all(in.fd.n,encoded.data(),z);
        if(codec){uLongf len=n;uLong used=z;
            require(uncompress2(raw.data(),&len,encoded.data(),&used)==Z_OK&&len==n&&used==z,"invalid compressed block");}
        else{require(z==n,"raw length mismatch");std::copy_n(encoded.data(),n,raw.data());}
        auto h=digest(raw.data(),n);require(std::equal(h.begin(),h.end(),hdr.begin()+12),"block SHA256 mismatch");
        require(total<=UINT64_MAX-n,"length overflow");total+=n;whole.add(raw.data(),n);
        if(out)write_all(out->fd->n,raw.data(),n);
        if(match)match->feed(raw.data(),n);
    }
    in.unchanged();if(out)out->publish();
    if(match)std::cout<<"OK count="<<match->count<<" bytes="<<total<<"\n";
    else std::cout<<"OK verified bytes="<<total<<"\n";
}
int main(int argc,char**argv){try{
    require(argc>=3,"usage: glyph-native pack INPUT OUTPUT | restore ARCHIVE OUTPUT | verify ARCHIVE | count ARCHIVE HEX");
    std::string cmd=argv[1];
    if(cmd=="pack"&&argc==4)pack(argv[2],argv[3]);
    else if(cmd=="restore"&&argc==4)consume(argv[2],argv[3],nullptr);
    else if(cmd=="verify"&&argc==3)consume(argv[2],nullptr,nullptr);
    else if(cmd=="count"&&argc==4){Matcher m(argv[3]);consume(argv[2],nullptr,&m);}
    else throw std::runtime_error("invalid command/arguments");
    return 0;
}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}}
