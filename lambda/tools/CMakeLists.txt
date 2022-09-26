#/****************************************************************
#* Copyright (c) 2022, liyinbin
#* All rights reserved.
#* Author by liyinbin (jeff.li) lijippy@163.com
#*****************************************************************/



carbin_cc_binary(
        NAMESPACE lambda
        NAME fvecs_to_bin
        SOURCES fvecs_to_bin.cc
        PUBLIC_LINKED_TARGETS ${CARBIN_SYS_DYLINK}  lambda::ann ${DYNAMIC_LIB} OpenMP::OpenMP_CXX ${MKL_LIBRARIES}
        PRIVATE_COMPILE_OPTIONS ${CARBIN_DEFAULT_COPTS}
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE lambda
        NAME fvecs_to_bvecs
        SOURCES fvecs_to_bvecs.cc
        PUBLIC_LINKED_TARGETS ${CARBIN_SYS_DYLINK}  lambda::ann ${DYNAMIC_LIB} OpenMP::OpenMP_CXX ${MKL_LIBRARIES}
        PRIVATE_COMPILE_OPTIONS ${CARBIN_DEFAULT_COPTS}
        PUBLIC
)


carbin_cc_binary(
        NAMESPACE lambda
        NAME ivecs_to_bin
        SOURCES ivecs_to_bin.cc
        PUBLIC_LINKED_TARGETS ${CARBIN_SYS_DYLINK}  lambda::ann ${DYNAMIC_LIB} OpenMP::OpenMP_CXX ${MKL_LIBRARIES}
        PRIVATE_COMPILE_OPTIONS ${CARBIN_DEFAULT_COPTS}
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE lambda
        NAME float_bin_to_int8
        SOURCES float_bin_to_int8.cc
        PUBLIC_LINKED_TARGETS ${CARBIN_SYS_DYLINK}  lambda::ann ${DYNAMIC_LIB} OpenMP::OpenMP_CXX ${MKL_LIBRARIES}
        PRIVATE_COMPILE_OPTIONS ${CARBIN_DEFAULT_COPTS}
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE lambda
        NAME tsv_to_bin
        SOURCES tsv_to_bin.cc
        PUBLIC_LINKED_TARGETS ${CARBIN_SYS_DYLINK}  lambda::ann ${DYNAMIC_LIB} OpenMP::OpenMP_CXX ${MKL_LIBRARIES}
        PRIVATE_COMPILE_OPTIONS ${CARBIN_DEFAULT_COPTS}
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE lambda
        NAME bin_to_tsv
        SOURCES bin_to_tsv.cc
        PUBLIC_LINKED_TARGETS ${CARBIN_SYS_DYLINK}  lambda::ann ${DYNAMIC_LIB} OpenMP::OpenMP_CXX ${MKL_LIBRARIES}
        PRIVATE_COMPILE_OPTIONS ${CARBIN_DEFAULT_COPTS}
        PUBLIC
)
find_package(Boost COMPONENTS program_options)
carbin_cc_binary(
        NAMESPACE lambda
        NAME compute_groundtruth
        SOURCES compute_groundtruth.cc
        PUBLIC_LINKED_TARGETS ${CARBIN_SYS_DYLINK}  lambda::ann ${DYNAMIC_LIB} OpenMP::OpenMP_CXX ${MKL_LIBRARIES} Boost::program_options
        PRIVATE_COMPILE_OPTIONS ${CARBIN_DEFAULT_COPTS}
        PUBLIC
)
#add_executable(rand_data_gen rand_data_gen.cc)
#target_link_libraries(rand_data_gen ${PROJECT_NAME} Boost::program_options)


#[[

add_executable(int8_to_float int8_to_float.cc)
target_link_libraries(int8_to_float ${PROJECT_NAME})

add_executable(int8_to_float_scale int8_to_float_scale.cc)
target_link_libraries(int8_to_float_scale ${PROJECT_NAME})

add_executable(uint8_to_float uint8_to_float.cc)
target_link_libraries(uint8_to_float ${PROJECT_NAME})

add_executable(uint32_to_uint8 uint32_to_uint8.cc)
target_link_libraries(uint32_to_uint8 ${PROJECT_NAME})

add_executable(vector_analysis vector_analysis.cc)
target_link_libraries(vector_analysis ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})

add_executable(gen_random_slice gen_random_slice.cc)
target_link_libraries(gen_random_slice ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})

add_executable(simulate_aggregate_recall simulate_aggregate_recall.cc)

add_executable(calculate_recall calculate_recall.cc)
target_link_libraries(calculate_recall ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})

# This is the only thing outside of DiskANN main source that depends on MKL.
add_executable(compute_groundtruth compute_groundtruth.cc)
target_include_directories(compute_groundtruth PRIVATE ${DISKANN_MKL_INCLUDE_DIRECTORIES})
target_link_libraries(compute_groundtruth ${PROJECT_NAME} ${DISKANN_MKL_LINK_LIBRARIES} ${DISKANN_ASYNC_LIB} Boost::program_options)


add_executable(generate_pq generate_pq.cc)
target_link_libraries(generate_pq ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})


add_executable(partition_data partition_data.cc)
target_link_libraries(partition_data ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})

add_executable(partition_with_ram_budget partition_with_ram_budget.cc)
target_link_libraries(partition_with_ram_budget ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})

add_executable(merge_shards merge_shards.cc)
target_link_libraries(merge_shards ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} ${DISKANN_ASYNC_LIB})

add_executable(create_disk_layout create_disk_layout.cc)
target_link_libraries(create_disk_layout ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})


]]

