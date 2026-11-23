#include <cstdlib>
#include <cstring>
#include <cstdio>
__attribute__((noinline)) void allocate(bool leak) {
    void *p = std::malloc(12345);
    if (!p) std::abort();
    std::memset(p, 42, 12345);
    std::printf("allocated=%p\n", p);
    if (!leak) std::free(p);
}
int main(int argc, char **) { allocate(argc > 1); }
