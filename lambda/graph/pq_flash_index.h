/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#pragma once

#include <cassert>
#include <sstream>
#include <stack>
#include <string>
#include "flare/container/robin_map.h"
#include "flare/container/robin_set.h"

#include "aligned_file_reader.h"
#include "concurrent_queue.h"
#include "neighbor.h"
#include "parameters.h"
#include "percentile_stats.h"
#include "pq_table.h"
#include "utils.h"
#include <flare/base/profile.h>

#define MAX_GRAPH_DEGREE 512
#define MAX_N_CMPS 16384
#define SECTOR_LEN (uint64_t) 4096
#define MAX_N_SECTOR_READS 128
#define MAX_PQ_CHUNKS 256

#define FULL_PRECISION_REORDER_MULTIPLIER 3

namespace lambda {
    template<typename T>
    struct QueryScratch {
        T *coord_scratch = nullptr;  // MUST BE AT LEAST [MAX_N_CMPS * data_dim]
        uint64_t coord_idx = 0;            // index of next [data_dim] scratch to use

        char *sector_scratch =
                nullptr;          // MUST BE AT LEAST [MAX_N_SECTOR_READS * SECTOR_LEN]
        uint64_t sector_idx = 0;  // index of next [SECTOR_LEN] scratch to use

        float *aligned_pqtable_dist_scratch =
                nullptr;  // MUST BE AT LEAST [256 * NCHUNKS]
        float *aligned_dist_scratch =
                nullptr;  // MUST BE AT LEAST diskann MAX_DEGREE
        uint8_t *aligned_pq_coord_scratch =
                nullptr;  // MUST BE AT LEAST  [N_CHUNKS * MAX_DEGREE]
        T *aligned_query_T = nullptr;
        float *aligned_query_float = nullptr;
        float *rotated_query = nullptr;

        flare::robin_set<uint64_t> *visited = nullptr;

        void reset() {
            coord_idx = 0;
            sector_idx = 0;
            visited->clear();  // does not deallocate memory.
        }
    };

    template<typename T>
    struct ThreadData {
        QueryScratch<T> scratch;
        IOContext ctx;
    };

    template<typename T>
    class pq_flash_index {
    public:
        FLARE_EXPORT pq_flash_index(
                std::shared_ptr<AlignedFileReader> &fileReader,
                lambda::Metric metric = lambda::Metric::L2);

        FLARE_EXPORT ~pq_flash_index();

        // load compressed data, and obtains the handle to the disk-resident index
        FLARE_EXPORT flare::result_status load(uint32_t num_threads, const char *index_prefix);

        FLARE_EXPORT void load_cache_list(std::vector<uint32_t> &node_list);

        FLARE_EXPORT flare::result_status generate_cache_list_from_sample_queries(
                std::string sample_bin, uint64_t l_search, uint64_t beamwidth,
                uint64_t num_nodes_to_cache, uint32_t num_threads,
                std::vector<uint32_t> &node_list);

        FLARE_EXPORT void cache_bfs_levels(uint64_t num_nodes_to_cache,
                                           std::vector<uint32_t> &node_list);

        FLARE_EXPORT void cached_beam_search(
                const T *query, const uint64_t k_search, const uint64_t l_search, uint64_t *res_ids,
                float *res_dists, const uint64_t beam_width,
                const bool use_reorder_data = false, query_stats *stats = nullptr);

        FLARE_EXPORT void cached_beam_search(
                const T *query, const uint64_t k_search, const uint64_t l_search, uint64_t *res_ids,
                float *res_dists, const uint64_t beam_width, const uint32_t io_limit,
                const bool use_reorder_data = false, query_stats *stats = nullptr);

        FLARE_EXPORT uint32_t range_search(const T *query1, const double range,
                                       const uint64_t min_l_search,
                                       const uint64_t max_l_search,
                                       std::vector<uint64_t> &indices,
                                       std::vector<float> &distances,
                                       const uint64_t min_beam_width,
                                       query_stats *stats = nullptr);

        std::shared_ptr<AlignedFileReader> &reader;

    protected:
        FLARE_EXPORT void use_medoids_data_as_centroids();

        FLARE_EXPORT void setup_thread_data(uint64_t nthreads);

        FLARE_EXPORT void destroy_thread_data();

    private:
        // index info
        // nhood of node `i` is in sector: [i / nnodes_per_sector]
        // offset in sector: [(i % nnodes_per_sector) * max_node_len]
        // nnbrs of node `i`: *(unsigned*) (buf)
        // nbrs of node `i`: ((unsigned*)buf) + 1
        uint64_t max_node_len = 0, nnodes_per_sector = 0, max_degree = 0;

        // Data used for searching with re-order vectors
        uint64_t ndims_reorder_vecs = 0, reorder_data_start_sector = 0,
                nvecs_per_sector = 0;

        lambda::Metric metric = lambda::Metric::L2;

        // used only for inner product search to re-scale the result value
        // (due to the pre-processing of base during index build)
        float max_base_norm = 0.0f;

        // data info
        uint64_t num_points = 0;
        uint64_t num_frozen_points = 0;
        uint64_t frozen_location = 0;
        uint64_t data_dim = 0;
        uint64_t disk_data_dim = 0;  // will be different from data_dim only if we use
        // PQ for disk data (very large dimensionality)
        uint64_t aligned_dim = 0;
        uint64_t disk_bytes_per_point = 0;

        std::string disk_index_file;
        std::vector<std::pair<uint32_t, uint32_t>> node_visit_counter;

        // PQ data
        // n_chunks = # of chunks ndims is split into
        // data: uint8_t * n_chunks
        // chunk_size = chunk size of each dimension chunk
        // pq_tables = float* [[2^8 * [chunk_size]] * n_chunks]
        uint8_t *data = nullptr;
        uint64_t n_chunks;
        fixed_chunk_pq_table pq_table;

        // distance comparator
        std::shared_ptr<vector_distance<T>> dist_cmp;
        std::shared_ptr<vector_distance<float>> dist_cmp_float;

        // for very large datasets: we use PQ even for the disk resident index
        bool use_disk_index_pq = false;
        uint64_t disk_pq_n_chunks = 0;
        fixed_chunk_pq_table disk_pq_table;

        // medoid/start info

        // graph has one entry point by default,
        // we can optionally have multiple starting points
        uint32_t *medoids = nullptr;
        // defaults to 1
        size_t num_medoids;
        // by default, it is empty. If there are multiple
        // centroids, we pick the medoid corresponding to the
        // closest centroid as the starting point of search
        float *centroid_data = nullptr;

        // nhood_cache
        unsigned *nhood_cache_buf = nullptr;
        flare::robin_map<uint32_t, std::pair<uint32_t, uint32_t *>> nhood_cache;

        // coord_cache
        T *coord_cache_buf = nullptr;
        flare::robin_map<uint32_t, T *> coord_cache;

        // thread-specific scratch
        ConcurrentQueue<ThreadData<T>> thread_data;
        uint64_t max_nthreads;
        bool load_flag = false;
        bool count_visited_nodes = false;
        bool reorder_data_exists = false;
        uint64_t reoreder_data_offset = 0;
    };
}  // namespace lambda
