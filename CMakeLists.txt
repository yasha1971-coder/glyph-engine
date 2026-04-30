cmake_minimum_required(VERSION 3.16)

project(GLYPH_CPP_BACKEND LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)

# ============================================
# build_sa
# ============================================
add_executable(build_sa
    src/build_sa.cpp
    third_party/libsais/libsais/src/libsais.c
)

target_include_directories(build_sa PRIVATE
    third_party/libsais/libsais/include
)

target_compile_options(build_sa PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -funroll-loops>
)

target_link_libraries(build_sa PRIVATE Threads::Threads)

# ============================================
# build_bwt
# ============================================
add_executable(build_bwt
    src/build_bwt.cpp
)

target_compile_options(build_bwt PRIVATE
    -O3
    -march=native
    -funroll-loops
)

target_link_libraries(build_bwt PRIVATE Threads::Threads)

# ============================================
# build_fm
# ============================================
add_executable(build_fm
    src/build_fm.cpp
)

target_compile_options(build_fm PRIVATE
    -O3
    -march=native
    -funroll-loops
)

target_link_libraries(build_fm PRIVATE Threads::Threads)

# ============================================
# query_fm_v1
# ============================================
add_executable(query_fm_v1
    src/query_fm_v1.cpp
)

target_compile_options(query_fm_v1 PRIVATE
    -O3
    -march=native
    -funroll-loops
)

target_link_libraries(query_fm_v1 PRIVATE Threads::Threads)

# ============================================
# build_chunk_map
# ============================================
add_executable(build_chunk_map
    src/build_chunk_map.cpp
)

target_compile_options(build_chunk_map PRIVATE
    -O3
    -march=native
    -funroll-loops
)

target_link_libraries(build_chunk_map PRIVATE Threads::Threads)

# ============================================
# query_fm_batch_v1
# ============================================
add_executable(query_fm_batch_v1
    src/query_fm_batch_v1.cpp
)

target_compile_options(query_fm_batch_v1 PRIVATE
    -O3
    -march=native
    -funroll-loops
)

target_link_libraries(query_fm_batch_v1 PRIVATE Threads::Threads)
add_executable(query_fm_server_v1
    src/query_fm_server_v1.cpp
)

target_compile_options(query_fm_server_v1 PRIVATE
    -O3
    -march=native
    -funroll-loops
)

target_link_libraries(query_fm_server_v1 PRIVATE Threads::Threads)

add_executable(build_sa_u32
    src/build_sa_u32.cpp
    third_party/libsais/libsais/src/libsais64.c
    third_party/libsais/libsais/src/libsais.c
)

target_include_directories(build_sa_u32 PRIVATE
    third_party/libsais/libsais/include
)

target_compile_options(build_sa_u32 PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

target_link_libraries(build_sa_u32 PRIVATE Threads::Threads)

add_executable(build_chunk_map_u24
    src/build_chunk_map_u24.cpp
)

target_compile_options(build_chunk_map_u24 PRIVATE -O3)
target_link_libraries(build_chunk_map_u24 PRIVATE Threads::Threads)

