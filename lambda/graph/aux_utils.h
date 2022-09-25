/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#pragma once

#include <algorithm>
#include <fcntl.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <tuple>
#ifdef __APPLE__
#else
#include <malloc.h>
#endif

#include <unistd.h>
#include "flare/container/robin_set.h"
#include "utils.h"
#include <flare/base/profile.h>

namespace lambda {
    const size_t MAX_PQ_TRAINING_SET_SIZE = 256000;
    const size_t MAX_SAMPLE_POINTS_FOR_WARMUP = 100000;
    const double PQ_TRAINING_SET_FRACTION = 0.1;
    const double SPACE_FOR_CACHED_NODES_IN_GB = 0.25;
    const double THRESHOLD_FOR_CACHING_IN_GB = 1.0;
    const uint32_t NUM_NODES_TO_CACHE = 250000;
    const uint32_t WARMUP_L = 20;
    const uint32_t NUM_KMEANS_REPS = 12;

    template<typename T>
    class pq_flash_index;

    FLARE_EXPORT double get_memory_budget(const std::string &mem_budget_str);

    FLARE_EXPORT double get_memory_budget(double search_ram_budget_in_gb);

    FLARE_EXPORT flare::result_status add_new_file_to_single_index(std::string index_file,
                                                                   std::string new_file);

    FLARE_EXPORT size_t calculate_num_pq_chunks(double final_index_ram_limit,
                                                size_t points_num,
                                                uint32_t dim);

    FLARE_EXPORT double calculate_recall(
            unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
            unsigned *our_results, unsigned dim_or, unsigned recall_at);

    FLARE_EXPORT double calculate_recall(
            unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
            unsigned *our_results, unsigned dim_or, unsigned recall_at,
            const flare::robin_set<unsigned> &active_tags);

    FLARE_EXPORT double calculate_range_search_recall(
            unsigned num_queries, std::vector<std::vector<uint32_t>> &groundtruth,
            std::vector<std::vector<uint32_t>> &our_results);

    [[nodiscard]] FLARE_EXPORT flare::result_status read_idmap(const std::string &fname,
                                 std::vector<unsigned> &ivecs);

    template<typename T>
    [[nodiscard]] FLARE_EXPORT std::pair<flare::result_status,T*> load_warmup(const std::string &cache_warmup_file,
                                uint64_t &warmup_num, uint64_t warmup_dim,
                                uint64_t warmup_aligned_dim);

    [[nodiscard]] FLARE_EXPORT flare::result_status merge_shards(const std::string &vamana_prefix,
                                  const std::string &vamana_suffix,
                                  const std::string &idmaps_prefix,
                                  const std::string &idmaps_suffix,
                                  const size_t nshards, size_t max_degree,
                                  const std::string &output_vamana,
                                  const std::string &medoids_file);

    template<typename T>
    FLARE_EXPORT std::string preprocess_base_file(
            const std::string &infile, const std::string &indexPrefix,
            lambda::Metric &distMetric);

    template<typename T>
    [[nodiscard]] FLARE_EXPORT flare::result_status build_merged_vamana_index(
            std::string base_file, lambda::Metric _compareMetric, unsigned L,
            unsigned R, double sampling_rate, double ram_budget,
            std::string mem_index_path, std::string medoids_file,
            std::string centroids_file);

    template<typename T>
    FLARE_EXPORT uint32_t optimize_beamwidth(
            std::unique_ptr<lambda::pq_flash_index<T>> &_pFlashIndex, T *tuning_sample,
            uint64_t tuning_sample_num, uint64_t tuning_sample_aligned_dim, uint32_t L,
            uint32_t nthreads, uint32_t start_bw = 2);

    template<typename T>
    [[nodiscard]] FLARE_EXPORT flare::result_status build_disk_index(const char *dataFilePath,
                                      const char *indexFilePath,
                                      const char *indexBuildParameters,
                                      lambda::Metric _compareMetric,
                                      bool use_opq = false);

    template<typename T>
    [[nodiscard]] FLARE_EXPORT flare::result_status create_disk_layout(
            const std::string base_file, const std::string mem_index_file,
            const std::string output_file,
            const std::string reorder_data_file = std::string(""));

}  // namespace lambda
