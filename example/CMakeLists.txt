

find_package(Boost COMPONENTS program_options)

carbin_cc_binary(
        NAMESPACE lambda
        NAME build_disk_index
        SOURCES build_disk_index.cc
        PUBLIC_LINKED_TARGETS ${CARBIN_SYS_DYLINK}  lambda::ann ${DYNAMIC_LIB} Boost::program_options OpenMP::OpenMP_CXX ${MKL_LIBRARIES}
        PRIVATE_COMPILE_OPTIONS ${CARBIN_DEFAULT_COPTS}
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE lambda
        NAME build_memory_index
        SOURCES build_memory_index.cc
        PUBLIC_LINKED_TARGETS ${CARBIN_SYS_DYLINK}  lambda::ann ${DYNAMIC_LIB} Boost::program_options OpenMP::OpenMP_CXX ${MKL_LIBRARIES}
        PRIVATE_COMPILE_OPTIONS ${CARBIN_DEFAULT_COPTS}
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE lambda
        NAME search_memory_index
        SOURCES search_memory_index.cc
        PUBLIC_LINKED_TARGETS ${CARBIN_SYS_DYLINK}  lambda::ann ${DYNAMIC_LIB} Boost::program_options OpenMP::OpenMP_CXX ${MKL_LIBRARIES}
        PRIVATE_COMPILE_OPTIONS ${CARBIN_DEFAULT_COPTS}
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE lambda
        NAME search_memory_index_dynamic
        SOURCES search_memory_index_dynamic.cc
        PUBLIC_LINKED_TARGETS ${CARBIN_SYS_DYLINK}  lambda::ann ${DYNAMIC_LIB} Boost::program_options OpenMP::OpenMP_CXX ${MKL_LIBRARIES}
        PRIVATE_COMPILE_OPTIONS ${CARBIN_DEFAULT_COPTS}
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE lambda
        NAME search_disk_index
        SOURCES search_disk_index.cc
        PUBLIC_LINKED_TARGETS ${CARBIN_SYS_DYLINK}  lambda::ann ${DYNAMIC_LIB} Boost::program_options OpenMP::OpenMP_CXX ${MKL_LIBRARIES}
        PRIVATE_COMPILE_OPTIONS ${CARBIN_DEFAULT_COPTS}
        PUBLIC
)