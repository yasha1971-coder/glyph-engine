#ifndef GLYPH_GLYPH_H
#define GLYPH_GLYPH_H

#include <stdint.h>

#if defined(_WIN32)
  #if defined(GLYPH_BUILDING_LIBRARY)
    #define GLYPH_API __declspec(dllexport)
  #else
    #define GLYPH_API __declspec(dllimport)
  #endif
  #define GLYPH_CALL __cdecl
#else
  #define GLYPH_API __attribute__((visibility("default")))
  #define GLYPH_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define GLYPH_ABI_VERSION_V1 UINT32_C(1)
#define GLYPH_SHA256_SIZE_V1 UINT32_C(32)

#define GLYPH_FALSE_V1 UINT32_C(0)
#define GLYPH_TRUE_V1 UINT32_C(1)

typedef struct glyph_index_v1 glyph_index_v1;

typedef int32_t glyph_status_v1;

enum glyph_status_code_v1 {
    GLYPH_OK = 0,
    GLYPH_E_ARG = 1,
    GLYPH_E_FORMAT = 2,
    GLYPH_E_VERIFY = 3,
    GLYPH_E_VERSION = 4,
    GLYPH_E_IO = 5,
    GLYPH_E_NOMEM = 6,
    GLYPH_E_LIMIT = 7,
    GLYPH_E_TIMEOUT = 8,
    GLYPH_E_BUSY = 9,
    GLYPH_E_CLOSED = 10,
    GLYPH_E_INTERNAL = 11,
    GLYPH_E_UNSUPPORTED = 12
};

enum glyph_open_flag_v1 {
    GLYPH_OPEN_FLAG_NONE_V1 = 0
};

enum glyph_query_flag_v1 {
    GLYPH_QUERY_FLAG_NONE_V1 = 0
};

typedef struct glyph_open_options_v1 {
    uint32_t struct_size;
    uint32_t flags;
    uint64_t max_mapped_bytes;
    uint64_t max_documents;
    uint64_t max_query_bytes;
    uint64_t reserved[5];
} glyph_open_options_v1;

#define GLYPH_OPEN_OPTIONS_V1_INIT \
    { \
        (uint32_t)sizeof(glyph_open_options_v1), \
        UINT32_C(0), \
        UINT64_C(0), \
        UINT64_C(0), \
        UINT64_C(0), \
        { \
            UINT64_C(0), UINT64_C(0), UINT64_C(0), \
            UINT64_C(0), UINT64_C(0) \
        } \
    }

typedef struct glyph_query_options_v1 {
    uint32_t struct_size;
    uint32_t flags;
    uint64_t timeout_ns;
    uint64_t reserved[5];
} glyph_query_options_v1;

#define GLYPH_QUERY_OPTIONS_V1_INIT \
    { \
        (uint32_t)sizeof(glyph_query_options_v1), \
        UINT32_C(0), \
        UINT64_C(0), \
        { \
            UINT64_C(0), UINT64_C(0), UINT64_C(0), \
            UINT64_C(0), UINT64_C(0) \
        } \
    }

typedef struct glyph_coordinate_v1 {
    uint64_t doc_id;
    uint64_t doc_offset;
} glyph_coordinate_v1;

typedef struct glyph_locate_result_v1 {
    uint32_t struct_size;
    uint32_t complete;
    uint64_t total_matches;
    uint64_t returned_matches;
    uint64_t reserved[4];
} glyph_locate_result_v1;

#define GLYPH_LOCATE_RESULT_V1_INIT \
    { \
        (uint32_t)sizeof(glyph_locate_result_v1), \
        GLYPH_FALSE_V1, \
        UINT64_C(0), \
        UINT64_C(0), \
        { \
            UINT64_C(0), UINT64_C(0), \
            UINT64_C(0), UINT64_C(0) \
        } \
    }

typedef struct glyph_index_info_v1 {
    uint32_t struct_size;
    uint32_t flags;
    uint64_t document_count;
    uint64_t total_source_bytes;
    uint8_t corpus_id_sha256[GLYPH_SHA256_SIZE_V1];
    uint8_t runtime_index_id_sha256[GLYPH_SHA256_SIZE_V1];
    uint64_t reserved[4];
} glyph_index_info_v1;

#define GLYPH_INDEX_INFO_V1_INIT \
    { \
        (uint32_t)sizeof(glyph_index_info_v1), \
        UINT32_C(0), \
        UINT64_C(0), \
        UINT64_C(0), \
        { UINT8_C(0) }, \
        { UINT8_C(0) }, \
        { \
            UINT64_C(0), UINT64_C(0), \
            UINT64_C(0), UINT64_C(0) \
        } \
    }

typedef struct glyph_document_info_v1 {
    uint32_t struct_size;
    uint32_t flags;
    uint64_t doc_id;
    uint64_t byte_length;
    uint8_t source_sha256[GLYPH_SHA256_SIZE_V1];
    uint64_t path_length_bytes;
    uint64_t reserved[4];
} glyph_document_info_v1;

#define GLYPH_DOCUMENT_INFO_V1_INIT \
    { \
        (uint32_t)sizeof(glyph_document_info_v1), \
        UINT32_C(0), \
        UINT64_C(0), \
        UINT64_C(0), \
        { UINT8_C(0) }, \
        UINT64_C(0), \
        { \
            UINT64_C(0), UINT64_C(0), \
            UINT64_C(0), UINT64_C(0) \
        } \
    }

GLYPH_API uint32_t GLYPH_CALL
glyph_abi_version_v1(void);

GLYPH_API glyph_status_v1 GLYPH_CALL
glyph_index_open_v1(
    const char *index_directory,
    const glyph_open_options_v1 *options,
    glyph_index_v1 **out_index
);

GLYPH_API glyph_status_v1 GLYPH_CALL
glyph_index_get_info_v1(
    glyph_index_v1 *index,
    glyph_index_info_v1 *out_info
);

GLYPH_API glyph_status_v1 GLYPH_CALL
glyph_document_get_info_v1(
    glyph_index_v1 *index,
    uint64_t doc_id,
    glyph_document_info_v1 *out_info
);

GLYPH_API glyph_status_v1 GLYPH_CALL
glyph_document_path_v1(
    glyph_index_v1 *index,
    uint64_t doc_id,
    uint8_t *buffer,
    uint64_t buffer_capacity,
    uint64_t *out_required_size
);

GLYPH_API glyph_status_v1 GLYPH_CALL
glyph_query_count_v1(
    glyph_index_v1 *index,
    const uint8_t *query,
    uint64_t query_size,
    const glyph_query_options_v1 *options,
    uint64_t *out_count
);

GLYPH_API glyph_status_v1 GLYPH_CALL
glyph_query_locate_v1(
    glyph_index_v1 *index,
    const uint8_t *query,
    uint64_t query_size,
    const glyph_query_options_v1 *options,
    uint64_t max_results,
    glyph_coordinate_v1 *coordinates,
    uint64_t coordinate_capacity,
    glyph_locate_result_v1 *out_result
);

GLYPH_API glyph_status_v1 GLYPH_CALL
glyph_index_close_v1(
    glyph_index_v1 **inout_index
);

#ifdef __cplusplus
}
#endif

#endif
