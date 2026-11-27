#define main accepted_cli_main
#include "mover-cli.cpp"
#undef main

#include <algorithm>
#include <vector>
#include <string>
#include <stdexcept>
static std::string decode_hex(const std::string& hex) {
 if(hex.empty()||hex.size()%2||hex.find_first_not_of("0123456789abcdefABCDEF")!=std::string::npos)throw std::runtime_error("invalid hex");
 std::string q;for(size_t i=0;i<hex.size();i+=2)q.push_back(static_cast<char>(std::stoul(hex.substr(i,2),nullptr,16)));return q;
}
static void emit(const std::string& hex,uint64_t count,std::vector<uint64_t> positions){
 std::cout<<"{\"query_hex\":\""<<hex<<"\",\"count\":"<<count<<",\"offsets\":[";
 for(size_t i=0;i<positions.size();++i)std::cout<<(i?",":"")<<positions[i];std::cout<<"]}"<<std::endl;
}

int main(int argc,char** argv){try {
 if(argc!=3||std::string(argv[1])!="serve")return accepted_cli_main(argc,argv);
 std::ifstream in(argv[2],std::ios::binary);if(!in)return 1;move_r<_locate_move> index;index.load(in);if(!in)return 1;
 std::cout<<"{\"ready\":true}"<<std::endl;
 std::string line;while(std::getline(std::cin,line)){
  if(line=="QUIT")break;size_t tab=line.find('\t');if(tab==std::string::npos)throw std::runtime_error("missing limit");
  auto hex=line.substr(0,tab);auto q=decode_hex(hex);uint64_t limit=std::stoull(line.substr(tab+1));
  auto query=index.query();bool found=true;for(auto i=q.rbegin();i!=q.rend();++i)if(!query.prepend(static_cast<unsigned char>(*i))){found=false;break;}uint64_t count=found?query.num_occ():0;std::vector<uint64_t> offsets;for(uint64_t i=0;i<std::min<uint64_t>(count,limit);++i)offsets.push_back(query.next_occ());emit(hex,count,offsets);
 }
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
