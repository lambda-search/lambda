
FILE(GLOB_RECURSE FAISS_INDEX_SRC "faiss/*.cc")
FILE(GLOB_RECURSE HNSW_INDEX_SRC "hnsw/*.cc")
FILE(GLOB_RECURSE GRAPH_INDEX_SRC "graph/*.cc")
FILE(GLOB_RECURSE COMMON_SRC "common/*.cc")

set(LAMBDA_SRC
        ann_index.cc
        ${FAISS_INDEX_SRC}
        ${HNSW_INDEX_SRC}
        ${GRAPH_INDEX_SRC}
        ${COMMON_SRC}
        dbd/dbd_index.cc
        common/roaring.c
        )

find_package(OpenMP REQUIRED)
find_package(MKL  REQUIRED)
carbin_cc_library(
        NAMESPACE lambda
        NAME ann
        SOURCES ${LAMBDA_SRC}
        PUBLIC_LINKED_TARGETS ${CARBIN_DYLINK}
        PRIVATE_LINKED_TARGETS lambda::proto OpenMP::OpenMP_CXX ${MKL_LIBRARIES} ${DYNAMIC_LIB}
        PRIVATE_COMPILE_OPTIONS ${CARBIN_DEFAULT_COPTS} -march=native
        PUBLIC
        SHARED
)