#include <cstddef>
#include <cstdint>
// Unsigned wraparound and reset-at-cut exactly match chunk_versions.split_python.
extern "C" std::size_t glyph_gear_cuts(const unsigned char* data, std::size_t n,
 const std::uint64_t* gear, std::size_t minimum, std::size_t target,
 std::size_t maximum, std::size_t* ends, std::size_t capacity) {
 if (!minimum || !target || (target & (target-1)) || maximum < minimum) return SIZE_MAX;
 std::size_t start=0, count=0; std::uint64_t rolling=0;
 for (std::size_t i=0;i<n;++i) {
  rolling=(rolling<<1)+gear[data[i]];
  const std::size_t size=i+1-start;
  if (size>=maximum || (size>=minimum && (rolling & (target-1))==0)) {
   if (count>=capacity) return SIZE_MAX;
   ends[count++]=i+1;start=i+1;rolling=0;
  }
 }
 if (start<n) { if(count>=capacity)return SIZE_MAX;ends[count++]=n; }
 return count;
}
