// P03: bounded exact binary Aho-Corasick source scan; research only.
// Nothing is output before full EOF, SHA-256 and stable source fstat.
// Not an RLB2/FM proof, owner signature, or freshness witness.
#include <openssl/evp.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <array>
#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
namespace {
constexpr uint64_t MAX_SOURCE=100000000;
constexpr size_t MAX_BYTES=4096,MAX_Q=16,MAX_HITS=256;
constexpr uint32_t NIL=UINT32_MAX;
void gate(bool condition,const char* why){if(!condition)throw std::runtime_error(why);}
uint64_t number(const std::string&s){gate(!s.empty(),"empty number");uint64_t n=0;for(char c:s){gate(c>='0'&&c<='9',"bad number");uint64_t d=c-'0';gate(n<=(UINT64_MAX-d)/10,"number overflow");n=n*10+d;}return n;}
int hexval(char c){if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;return -1;}
std::vector<uint8_t> unhex(const std::string&s){gate(!s.empty()&&!(s.size()%2),"bad hex size");std::vector<uint8_t>b;for(size_t i=0;i<s.size();i+=2){int a=hexval(s[i]),c=hexval(s[i+1]);gate(a>=0&&c>=0,"bad hex");b.push_back(static_cast<uint8_t>(16*a+c));}return b;}
std::string hex(const unsigned char*b,size_t n){const char*h="0123456789abcdef";std::string s;for(size_t i=0;i<n;i++){s+=h[b[i]>>4];s+=h[b[i]&15];}return s;}
struct Node{std::array<uint32_t,256> next;uint32_t fail=0,mask=0;Node(){next.fill(NIL);}};
struct Automaton{
 std::vector<Node> nodes{1};
 explicit Automaton(const std::vector<std::vector<uint8_t>>&patterns){
  gate(patterns.size()>0&&patterns.size()<=MAX_Q,"query admission");
  size_t sum=0;
  for(size_t k=0;k<patterns.size();k++){
   gate(!patterns[k].empty(),"empty pattern");sum+=patterns[k].size();gate(sum<=MAX_BYTES,"memory admission");
   uint32_t state=0;
   for(uint8_t c:patterns[k]){
    auto edge=nodes[state].next[c];
    if(edge==NIL){gate(nodes.size()<=MAX_BYTES,"node admission");edge=static_cast<uint32_t>(nodes.size());nodes[state].next[c]=edge;nodes.emplace_back();}
    state=edge;
   }
   nodes[state].mask|=uint32_t(1)<<k;
  }
  std::deque<uint32_t>q;
  for(size_t c=0;c<256;c++){uint32_t &next=nodes[0].next[c];if(next==NIL)next=0;else q.push_back(next);}
  while(!q.empty()){
   uint32_t v=q.front();q.pop_front();nodes[v].mask|=nodes[nodes[v].fail].mask;
   for(size_t c=0;c<256;c++){
    auto u=nodes[v].next[c];
    if(u==NIL)nodes[v].next[c]=nodes[nodes[v].fail].next[c];
    else{nodes[u].fail=nodes[nodes[v].fail].next[c];q.push_back(u);}
   }
  }
 }
};
struct File{int fd=-1;explicit File(const std::string&path){fd=open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK);gate(fd>=0,"source refused");}~File(){if(fd>=0)close(fd);}};
struct Sha{
 EVP_MD_CTX*ptr=EVP_MD_CTX_new();
 Sha(){gate(ptr!=nullptr&&EVP_DigestInit_ex(ptr,EVP_sha256(),nullptr)==1,"digest init");}
 ~Sha(){if(ptr)EVP_MD_CTX_free(ptr);}
 void update(const void*p,size_t n){gate(EVP_DigestUpdate(ptr,p,n)==1,"digest update");}
 std::string done(){unsigned char output[EVP_MAX_MD_SIZE];unsigned n=0;gate(EVP_DigestFinal_ex(ptr,output,&n)==1&&n==32,"digest final");return hex(output,n);}
};
struct Count{uint64_t total=0;std::vector<uint64_t>offsets;};
bool same(const struct stat&a,const struct stat&b){
 return a.st_dev==b.st_dev&&a.st_ino==b.st_ino&&a.st_size==b.st_size&&
 a.st_mtim.tv_sec==b.st_mtim.tv_sec&&a.st_mtim.tv_nsec==b.st_mtim.tv_nsec&&
 a.st_ctim.tv_sec==b.st_ctim.tv_sec&&a.st_ctim.tv_nsec==b.st_ctim.tv_nsec;
}
void execute(const std::string&path,uint64_t bytes,const std::string&pin,
 const std::vector<std::vector<uint8_t>>&patterns){
 gate(bytes>0&&bytes<=MAX_SOURCE,"source length bound");
 gate(pin.size()==64&&unhex(pin).size()==32,"bad pin");
 Automaton ac(patterns);
 File input(path);struct stat before{},after{};
 gate(fstat(input.fd,&before)==0&&S_ISREG(before.st_mode),"nonregular source");
 gate(before.st_size>=0&&static_cast<uint64_t>(before.st_size)==bytes,"source geometry");
 Sha sha;std::vector<Count>result(patterns.size());
 std::vector<uint8_t>block(1<<20);
 uint64_t processed=0;uint32_t state=0;
 for(;;){
  ssize_t n=read(input.fd,block.data(),block.size());
  if(n<0&&errno==EINTR)continue;
  gate(n>=0,"read error");if(n==0)break;
  gate(processed+static_cast<uint64_t>(n)<=bytes,"source grew");
  sha.update(block.data(),static_cast<size_t>(n));
  for(ssize_t j=0;j<n;j++){
   state=ac.nodes[state].next[block[static_cast<size_t>(j)]];
   uint32_t mask=ac.nodes[state].mask;
   while(mask){
    unsigned i=static_cast<unsigned>(__builtin_ctz(mask));mask&=mask-1;
    Count&c=result[i];gate(c.total<UINT64_MAX,"count overflow");c.total++;
    if(c.offsets.size()<MAX_HITS){
     uint64_t end=processed+static_cast<uint64_t>(j)+1;
     gate(end>=patterns[i].size(),"invalid suffix");
     c.offsets.push_back(end-patterns[i].size());
    }
   }
  }
  processed+=static_cast<uint64_t>(n);
 }
 gate(processed==bytes&&fstat(input.fd,&after)==0&&same(before,after),"incomplete or changing source");
 gate(sha.done()==pin,"full-source SHA mismatch");
 std::cout<<"{\"profile\":\"CORE_PERF_P03_NATIVE_AC_RESEARCH\",\"source_bytes\":"<<processed
 <<",\"source_reads_bytes\":"<<processed<<",\"source_hash_passes\":1,\"eof\":true"
 <<",\"automaton_nodes\":"<<ac.nodes.size()<<",\"automaton_bytes\":"
 <<ac.nodes.capacity()*sizeof(Node)
 <<",\"signed_root\":false,\"index_semantics_proven\":false,\"results\":[";
 for(size_t i=0;i<patterns.size();i++){
  if(i)std::cout<<",";
  auto &c=result[i];gate(c.offsets.size()==std::min<uint64_t>(c.total,MAX_HITS),"offset count");
  bool limited=c.total>MAX_HITS;
  std::cout<<"{\"pattern_hex\":\""<<hex(patterns[i].data(),patterns[i].size())
  <<"\",\"count\":"<<c.total<<",\"offsets\":[";
  for(size_t j=0;j<c.offsets.size();j++){if(j)std::cout<<",";std::cout<<c.offsets[j];}
  std::cout<<"],\"truncated\":"<<(limited?"true":"false")
  <<",\"status\":\""<<(limited?"UNTRUSTED":c.total?"FOUND":"PROVEN_EMPTY")
  <<"\",\"reason\":\""<<(limited?"LIMITED_COMPLETENESS":"AUTHENTICATED_FULL_SCAN")<<"\"}";
 }
 std::cout<<"]}\n";
}
}
int main(int argc,char**argv){
 try{
  std::string path,pin;uint64_t bytes=0;std::vector<std::vector<uint8_t>>patterns;
  for(int i=1;i<argc;i++){
   gate(i+1<argc,"missing argument");std::string key=argv[i],value=argv[++i];
   if(key=="--source")path=value;
   else if(key=="--source-bytes")bytes=number(value);
   else if(key=="--sha256")pin=value;
   else if(key=="--pattern-hex")patterns.push_back(unhex(value));
   else throw std::runtime_error("unknown option");
  }
  gate(!path.empty()&&!pin.empty(),"missing source/pin");execute(path,bytes,pin,patterns);
  return 0;
 }catch(const std::exception&){std::cerr<<"{\"status\":\"UNTRUSTED\",\"error\":\"P03 AC rejected\"}\n";return 3;}
}
