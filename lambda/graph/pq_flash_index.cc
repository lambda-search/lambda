/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "flare/log/logging.h"
#include "pq_flash_index.h"
#include <flare/base/profile.h>

#ifndef FLARE_PLATFORM_OSX
#include <malloc.h>
#endif

#include "percentile_stats.h"

#include <omp.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iterator>
#include <thread>
#include "lambda/common/vector_distance.h"
#include "parameters.h"
#include <flare/times/time.h>
#include "utils.h"
#include "cosine_similarity.h"
#include "flare/container/robin_set.h"
#include "linux_aligned_file_reader.h"

#define READ_U64(stream, val) stream.read((char *) &val, sizeof(uint64_t))
#define READ_U32(stream, val) stream.read((char *) &val, sizeof(uint32_t))
#define READ_UNSIGNED(stream, val) stream.read((char *) &val, sizeof(unsigned))

// sector # on disk where node_id is present with in the graph part
#define NODE_SECTOR_NO(node_id) (((uint64_t)(node_id)) / nnodes_per_sector + 1)

// obtains region of sector containing node
#define OFFSET_TO_NODE(sector_buf, node_id) \
  ((char *) sector_buf + (((uint64_t) node_id) % nnodes_per_sector) * max_node_len)

// returns region of `node_buf` containing [NNBRS][NBR_ID(uint32_t)]
#define OFFSET_TO_NODE_NHOOD(node_buf) \
  (unsigned *) ((char *) node_buf + disk_bytes_per_point)

// returns region of `node_buf` containing [COORD(T)]
#define OFFSET_TO_NODE_COORDS(node_buf) (T *) (node_buf)

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_NO(id) \
  (((uint64_t)(id)) / nvecs_per_sector + reorder_data_start_sector)

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_OFFSET(id) \
  ((((uint64_t)(id)) % nvecs_per_sector) * data_dim * sizeof(float))

namespace {
    void aggregate_coords(const unsigned *ids, const uint64_t n_ids,
                          const uint8_t *all_coords, const uint64_t ndims, uint8_t *out) {
        for (uint64_t i = 0; i < n_ids; i++) {
            memcpy(out + i * ndims, all_coords + ids[i] * ndims, ndims * sizeof(uint8_t));
        }
    }

    void pq_dist_lookup(const uint8_t *pq_ids, const uint64_t n_pts,
                        const uint64_t pq_nchunks, const float *pq_dists,
                        float *dists_out) {
        _mm_prefetch((char *) dists_out, _MM_HINT_T0);
        _mm_prefetch((char *) pq_ids, _MM_HINT_T0);
        _mm_prefetch((char *) (pq_ids + 64), _MM_HINT_T0);
        _mm_prefetch((char *) (pq_ids + 128), _MM_HINT_T0);
        memset(dists_out, 0, n_pts * sizeof(float));
        for (uint64_t chunk = 0; chunk < pq_nchunks; chunk++) {
            const float *chunk_dists = pq_dists + 256 * chunk;
            if (chunk < pq_nchunks - 1) {
                _mm_prefetch((char *) (chunk_dists + 256), _MM_HINT_T0);
            }
            for (uint64_t idx = 0; idx < n_pts; idx++) {
                uint8_t pq_centerid = pq_ids[pq_nchunks * idx + chunk];
                dists_out[idx] += chunk_dists[pq_centerid];
            }
        }
    }
}  // namespace

namespace lambda {
    template<typename T>
    pq_flash_index<T>::pq_flash_index(std::shared_ptr<AlignedFileReader> &fileReader,
                                      lambda::Metric m)
            : reader(fileReader), metric(m) {
        if (m == lambda::Metric::COSINE || m == lambda::Metric::INNER_PRODUCT) {
            if (std::is_floating_point<T>::value) {
                FLARE_LOG(INFO) << "Cosine metric chosen for (normalized) float data."
                                   "Changing distance to L2 to boost accuracy.";
                m = lambda::Metric::L2;
            } else {
                FLARE_LOG(ERROR) << "WARNING: Cannot normalize integral data types."
                                 << " This may result in erroneous results or poor recall."
                                 << " Consider using L2 distance with integral data types.";
            }
        }

        this->dist_cmp.reset(lambda::get_distance_function<T>(m));
        this->dist_cmp_float.reset(lambda::get_distance_function<float>(m));
    }

    template<typename T>
    pq_flash_index<T>::~pq_flash_index() {
        if (data != nullptr) {
            delete[] data;
        }
        if (centroid_data != nullptr)
            aligned_free(centroid_data);
        // delete backing bufs for nhood and coord cache
        if (nhood_cache_buf != nullptr) {
            delete[] nhood_cache_buf;
            lambda::aligned_free(coord_cache_buf);
        }

        if (load_flag) {
            this->destroy_thread_data();
            reader->close();
        }
    }

    template<typename T>
    void pq_flash_index<T>::setup_thread_data(uint64_t nthreads) {
        FLARE_LOG(INFO) << "Setting up thread-specific contexts for nthreads: "
                        << nthreads;
// omp parallel for to generate unique thread IDs
#pragma omp parallel for num_threads((int) nthreads)
        for (int64_t thread = 0; thread < (int64_t) nthreads; thread++) {
#pragma omp critical
            {
                this->reader->register_thread();
                IOContext &ctx = this->reader->get_ctx();
                QueryScratch<T> scratch;
                uint64_t coord_alloc_size = ROUND_UP(MAX_N_CMPS * this->aligned_dim, 256);
                lambda::alloc_aligned((void **) &scratch.coord_scratch,
                                      coord_alloc_size, 256);
                lambda::alloc_aligned((void **) &scratch.sector_scratch,
                                      (uint64_t) MAX_N_SECTOR_READS * (uint64_t) SECTOR_LEN,
                                      SECTOR_LEN);
                lambda::alloc_aligned(
                        (void **) &scratch.aligned_pq_coord_scratch,
                        (uint64_t) MAX_GRAPH_DEGREE * (uint64_t) MAX_PQ_CHUNKS * sizeof(uint8_t), 256);
                lambda::alloc_aligned((void **) &scratch.aligned_pqtable_dist_scratch,
                                      256 * (uint64_t) MAX_PQ_CHUNKS * sizeof(float), 256);
                lambda::alloc_aligned((void **) &scratch.aligned_dist_scratch,
                                      (uint64_t) MAX_GRAPH_DEGREE * sizeof(float), 256);
                lambda::alloc_aligned((void **) &scratch.aligned_query_T,
                                      this->aligned_dim * sizeof(T), 8 * sizeof(T));
                lambda::alloc_aligned((void **) &scratch.aligned_query_float,
                                      this->aligned_dim * sizeof(float),
                                      8 * sizeof(float));
                scratch.visited = new flare::robin_set<uint64_t>(4096);
                lambda::alloc_aligned((void **) &scratch.rotated_query,
                                      this->aligned_dim * sizeof(float),
                                      8 * sizeof(float));

                memset(scratch.coord_scratch, 0, MAX_N_CMPS * this->aligned_dim);
                memset(scratch.aligned_query_T, 0, this->aligned_dim * sizeof(T));
                memset(scratch.aligned_query_float, 0,
                       this->aligned_dim * sizeof(float));
                memset(scratch.rotated_query, 0, this->aligned_dim * sizeof(float));

                ThreadData<T> data;
                data.ctx = ctx;
                data.scratch = scratch;
                this->thread_data.push(data);
            }
        }
        load_flag = true;
    }

    template<typename T>
    void pq_flash_index<T>::destroy_thread_data() {
        FLARE_LOG(INFO) << "Clearing scratch";
        assert(this->thread_data.size() == this->max_nthreads);
        while (this->thread_data.size() > 0) {
            ThreadData<T> data = this->thread_data.pop();
            while (data.scratch.sector_scratch == nullptr) {
                this->thread_data.wait_for_push_notify();
                data = this->thread_data.pop();
            }
            auto &scratch = data.scratch;
            lambda::aligned_free((void *) scratch.coord_scratch);
            lambda::aligned_free((void *) scratch.sector_scratch);
            lambda::aligned_free((void *) scratch.aligned_pq_coord_scratch);
            lambda::aligned_free((void *) scratch.aligned_pqtable_dist_scratch);
            lambda::aligned_free((void *) scratch.aligned_dist_scratch);
            lambda::aligned_free((void *) scratch.aligned_query_float);
            lambda::aligned_free((void *) scratch.rotated_query);
            lambda::aligned_free((void *) scratch.aligned_query_T);

            delete scratch.visited;
        }
        this->reader->deregister_all_threads();
    }

    template<typename T>
    void pq_flash_index<T>::load_cache_list(std::vector<uint32_t> &node_list) {
        FLARE_LOG(INFO) << "Loading the cache list into memory.." << std::flush;
        uint64_t num_cached_nodes = node_list.size();

        // borrow thread data
        ThreadData<T> this_thread_data = this->thread_data.pop();
        while (this_thread_data.scratch.sector_scratch == nullptr) {
            this->thread_data.wait_for_push_notify();
            this_thread_data = this->thread_data.pop();
        }

        IOContext &ctx = this_thread_data.ctx;

        nhood_cache_buf = new unsigned[num_cached_nodes * (max_degree + 1)];
        memset(nhood_cache_buf, 0, num_cached_nodes * (max_degree + 1));

        uint64_t coord_cache_buf_len = num_cached_nodes * aligned_dim;
        lambda::alloc_aligned((void **) &coord_cache_buf,
                              coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
        memset(coord_cache_buf, 0, coord_cache_buf_len * sizeof(T));

        size_t BLOCK_SIZE = 8;
        size_t num_blocks = DIV_ROUND_UP(num_cached_nodes, BLOCK_SIZE);

        for (uint64_t block = 0; block < num_blocks; block++) {
            uint64_t start_idx = block * BLOCK_SIZE;
            uint64_t end_idx = (std::min)(num_cached_nodes, (block + 1) * BLOCK_SIZE);
            std::vector<AlignedRead> read_reqs;
            std::vector<std::pair<uint32_t, char *>> nhoods;
            for (uint64_t node_idx = start_idx; node_idx < end_idx; node_idx++) {
                AlignedRead read;
                char *buf = nullptr;
                alloc_aligned((void **) &buf, SECTOR_LEN, SECTOR_LEN);
                nhoods.push_back(std::make_pair(node_list[node_idx], buf));
                read.len = SECTOR_LEN;
                read.buf = buf;
                read.offset = NODE_SECTOR_NO(node_list[node_idx]) * SECTOR_LEN;
                read_reqs.push_back(read);
            }

            reader->read(read_reqs, ctx);

            uint64_t node_idx = start_idx;
            for (uint32_t i = 0; i < read_reqs.size(); i++) {
                auto &nhood = nhoods[i];
                char *node_buf = OFFSET_TO_NODE(nhood.second, nhood.first);
                T *node_coords = OFFSET_TO_NODE_COORDS(node_buf);
                T *cached_coords = coord_cache_buf + node_idx * aligned_dim;
                memcpy(cached_coords, node_coords, disk_bytes_per_point);
                coord_cache.insert(std::make_pair(nhood.first, cached_coords));

                // insert node nhood into nhood_cache
                unsigned *node_nhood = OFFSET_TO_NODE_NHOOD(node_buf);

                auto nnbrs = *node_nhood;
                unsigned *nbrs = node_nhood + 1;
                std::pair<uint32_t, unsigned *> cnhood;
                cnhood.first = nnbrs;
                cnhood.second = nhood_cache_buf + node_idx * (max_degree + 1);
                memcpy(cnhood.second, nbrs, nnbrs * sizeof(unsigned));
                nhood_cache.insert(std::make_pair(nhood.first, cnhood));
                aligned_free(nhood.second);
                node_idx++;
            }
        }
        // return thread data
        this->thread_data.push(this_thread_data);
        this->thread_data.push_notify_all();
        FLARE_LOG(INFO) << "..done.";
    }

    template<typename T>
    flare::result_status pq_flash_index<T>::generate_cache_list_from_sample_queries(
            std::string sample_bin, uint64_t l_search, uint64_t beamwidth,
            uint64_t num_nodes_to_cache, uint32_t nthreads,
            std::vector<uint32_t> &node_list) {
        this->count_visited_nodes = true;
        this->node_visit_counter.clear();
        this->node_visit_counter.resize(this->num_points);
        for (uint32_t i = 0; i < node_visit_counter.size(); i++) {
            this->node_visit_counter[i].first = i;
            this->node_visit_counter[i].second = 0;
        }

        size_t sample_num, sample_dim, sample_aligned_dim;
        T *samples;

        if (file_exists(sample_bin)) {
            auto rs = lambda::binary_file::load_aligned_bin<T>(sample_bin, samples, sample_num, sample_dim,
                                                     sample_aligned_dim);
            if(!rs.is_ok()) {
                return rs;
            }
        } else {
            FLARE_LOG(ERROR) << "Sample bin file not found. Not generating cache.";
            return flare::result_status(-1, "Sample bin file not found. Not generating cache.");
        }

        std::vector<uint64_t> tmp_result_ids_64(sample_num, 0);
        std::vector<float> tmp_result_dists(sample_num, 0);

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
        for (int64_t i = 0; i < (int64_t) sample_num; i++) {
            cached_beam_search(samples + (i * sample_aligned_dim), 1, l_search,
                               tmp_result_ids_64.data() + (i * 1),
                               tmp_result_dists.data() + (i * 1), beamwidth);
        }

        std::sort(this->node_visit_counter.begin(), node_visit_counter.end(),
                  [](std::pair<uint32_t, uint32_t> &left, std::pair<uint32_t, uint32_t> &right) {
                      return left.second > right.second;
                  });
        node_list.clear();
        node_list.shrink_to_fit();
        node_list.reserve(num_nodes_to_cache);
        for (uint64_t i = 0; i < num_nodes_to_cache; i++) {
            node_list.push_back(this->node_visit_counter[i].first);
        }
        this->count_visited_nodes = false;

        lambda::aligned_free(samples);
        return flare::result_status::success();
    }

    template<typename T>
    void pq_flash_index<T>::cache_bfs_levels(uint64_t num_nodes_to_cache,
                                             std::vector<uint32_t> &node_list) {
        std::random_device rng;
        std::mt19937 urng(rng());

        node_list.clear();

        // Do not cache more than 10% of the nodes in the index
        uint64_t tenp_nodes = (uint64_t) (std::round(this->num_points * 0.1));
        if (num_nodes_to_cache > tenp_nodes) {
            FLARE_LOG(INFO) << "Reducing nodes to cache from: " << num_nodes_to_cache
                            << " to: " << tenp_nodes
                            << "(10 percent of total nodes:" << this->num_points << ")";
            num_nodes_to_cache = tenp_nodes == 0 ? 1 : tenp_nodes;
        }
        FLARE_LOG(INFO) << "Caching " << num_nodes_to_cache << "...";

        // borrow thread data
        ThreadData<T> this_thread_data = this->thread_data.pop();
        while (this_thread_data.scratch.sector_scratch == nullptr) {
            this->thread_data.wait_for_push_notify();
            this_thread_data = this->thread_data.pop();
        }

        IOContext &ctx = this_thread_data.ctx;

        std::unique_ptr<flare::robin_set<unsigned>> cur_level, prev_level;
        cur_level = std::make_unique<flare::robin_set<unsigned>>();
        prev_level = std::make_unique<flare::robin_set<unsigned>>();

        for (uint64_t miter = 0; miter < num_medoids; miter++) {
            cur_level->insert(medoids[miter]);
        }

        uint64_t lvl = 1;
        uint64_t prev_node_list_size = 0;
        while ((node_list.size() + cur_level->size() < num_nodes_to_cache) &&
               cur_level->size() != 0) {
            // swap prev_level and cur_level
            std::swap(prev_level, cur_level);
            // clear cur_level
            cur_level->clear();

            std::vector<unsigned> nodes_to_expand;

            for (const unsigned &id : *prev_level) {
                if (std::find(node_list.begin(), node_list.end(), id) !=
                    node_list.end()) {
                    continue;
                }
                node_list.push_back(id);
                nodes_to_expand.push_back(id);
            }

            std::shuffle(nodes_to_expand.begin(), nodes_to_expand.end(), urng);

            FLARE_LOG(INFO) << "Level: " << lvl << std::flush;
            bool finish_flag = false;

            size_t BLOCK_SIZE = 1024;
            uint64_t nblocks = DIV_ROUND_UP(nodes_to_expand.size(), BLOCK_SIZE);
            for (size_t block = 0; block < nblocks && !finish_flag; block++) {
                FLARE_LOG(INFO) << "." << std::flush;
                size_t start = block * BLOCK_SIZE;
                size_t end =
                        (std::min)((block + 1) * BLOCK_SIZE, nodes_to_expand.size());
                std::vector<AlignedRead> read_reqs;
                std::vector<std::pair<uint32_t, char *>> nhoods;
                for (size_t cur_pt = start; cur_pt < end; cur_pt++) {
                    char *buf = nullptr;
                    alloc_aligned((void **) &buf, SECTOR_LEN, SECTOR_LEN);
                    nhoods.push_back(std::make_pair(nodes_to_expand[cur_pt], buf));
                    AlignedRead read;
                    read.len = SECTOR_LEN;
                    read.buf = buf;
                    read.offset = NODE_SECTOR_NO(nodes_to_expand[cur_pt]) * SECTOR_LEN;
                    read_reqs.push_back(read);
                }

                // issue read requests
                reader->read(read_reqs, ctx);

                // process each nhood buf
                for (uint32_t i = 0; i < read_reqs.size(); i++) {
                    auto &nhood = nhoods[i];

                    // insert node coord into coord_cache
                    char *node_buf = OFFSET_TO_NODE(nhood.second, nhood.first);
                    unsigned *node_nhood = OFFSET_TO_NODE_NHOOD(node_buf);
                    uint64_t nnbrs = (uint64_t) *node_nhood;
                    unsigned *nbrs = node_nhood + 1;
                    // explore next level
                    for (uint64_t j = 0; j < nnbrs && !finish_flag; j++) {
                        if (std::find(node_list.begin(), node_list.end(), nbrs[j]) ==
                            node_list.end()) {
                            cur_level->insert(nbrs[j]);
                        }
                        if (cur_level->size() + node_list.size() >= num_nodes_to_cache) {
                            finish_flag = true;
                        }
                    }
                    aligned_free(nhood.second);
                }
            }

            FLARE_LOG(INFO) << ". #nodes: " << node_list.size() - prev_node_list_size
                            << ", #nodes thus far: " << node_list.size();
            prev_node_list_size = node_list.size();
            lvl++;
        }

        std::vector<uint32_t> cur_level_node_list;
        for (const unsigned &p : *cur_level)
            cur_level_node_list.push_back(p);

        std::shuffle(cur_level_node_list.begin(), cur_level_node_list.end(), urng);
        size_t residual = num_nodes_to_cache - node_list.size();

        for (size_t i = 0; i < (std::min)(residual, cur_level_node_list.size());
             i++)
            node_list.push_back(cur_level_node_list[i]);

        FLARE_LOG(INFO) << "Level: " << lvl;
        FLARE_LOG(INFO) << ". #nodes: " << node_list.size() - prev_node_list_size
                        << ", #nodes thus far: " << node_list.size();

        // return thread data
        this->thread_data.push(this_thread_data);
        this->thread_data.push_notify_all();

        FLARE_LOG(INFO) << "done";
    }

    template<typename T>
    void pq_flash_index<T>::use_medoids_data_as_centroids() {
        if (centroid_data != nullptr)
            aligned_free(centroid_data);
        alloc_aligned(((void **) &centroid_data),
                      num_medoids * aligned_dim * sizeof(float), 32);
        std::memset(centroid_data, 0, num_medoids * aligned_dim * sizeof(float));

        // borrow ctx
        ThreadData<T> data = this->thread_data.pop();
        while (data.scratch.sector_scratch == nullptr) {
            this->thread_data.wait_for_push_notify();
            data = this->thread_data.pop();
        }
        IOContext &ctx = data.ctx;
        FLARE_LOG(INFO) << "Loading centroid data from medoids vector data of "
                        << num_medoids << " medoid(s)";
        for (uint64_t cur_m = 0; cur_m < num_medoids; cur_m++) {
            auto medoid = medoids[cur_m];
            // read medoid nhood
            char *medoid_buf = nullptr;
            alloc_aligned((void **) &medoid_buf, SECTOR_LEN, SECTOR_LEN);
            std::vector<AlignedRead> medoid_read(1);
            medoid_read[0].len = SECTOR_LEN;
            medoid_read[0].buf = medoid_buf;
            medoid_read[0].offset = NODE_SECTOR_NO(medoid) * SECTOR_LEN;
            reader->read(medoid_read, ctx);

            // all data about medoid
            char *medoid_node_buf = OFFSET_TO_NODE(medoid_buf, medoid);

            // add medoid coords to `coord_cache`
            T *medoid_coords = new T[data_dim];
            T *medoid_disk_coords = OFFSET_TO_NODE_COORDS(medoid_node_buf);
            memcpy(medoid_coords, medoid_disk_coords, disk_bytes_per_point);

            if (!use_disk_index_pq) {
                for (uint32_t i = 0; i < data_dim; i++)
                    centroid_data[cur_m * aligned_dim + i] = medoid_coords[i];
            } else {
                disk_pq_table.inflate_vector((uint8_t *) medoid_coords,
                                             (centroid_data + cur_m * aligned_dim));
            }

            aligned_free(medoid_buf);
            delete[] medoid_coords;
        }

        // return ctx
        this->thread_data.push(data);
        this->thread_data.push_notify_all();
    }

    template<typename T>
    flare::result_status pq_flash_index<T>::load(uint32_t num_threads, const char *index_prefix) {
        std::string pq_table_bin = std::string(index_prefix) + "_pq_pivots.bin";
        std::string pq_compressed_vectors =
                std::string(index_prefix) + "_pq_compressed.bin";
        std::string disk_index_file = std::string(index_prefix) + "_disk.index";
        std::string medoids_file = std::string(disk_index_file) + "_medoids.bin";
        std::string centroids_file =
                std::string(disk_index_file) + "_centroids.bin";

        size_t pq_file_dim, pq_file_num_centroids;
        get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim,
                         METADATA_SIZE);

        this->disk_index_file = disk_index_file;

        if (pq_file_num_centroids != 256) {
            FLARE_LOG(INFO) << "Error. Number of PQ centroids is not 256. Exitting.";
            return flare::result_status(-1, "Error. Number of PQ centroids is not 256. Exitting.");
        }

        this->data_dim = pq_file_dim;
        // will reset later if we use PQ on disk
        this->disk_data_dim = this->data_dim;
        // will change later if we use PQ on disk or if we are using
        // inner product without PQ
        this->disk_bytes_per_point = this->data_dim * sizeof(T);
        this->aligned_dim = ROUND_UP(pq_file_dim, 8);

        size_t nptsuint64_t, nchunksuint64_t;
        auto rs = lambda::binary_file::load_bin<uint8_t>(pq_compressed_vectors, this->data, nptsuint64_t,
                                               nchunksuint64_t);
        if(!rs.is_ok()) {
            return rs;
        }
        this->num_points = nptsuint64_t;
        this->n_chunks = nchunksuint64_t;

        rs = pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunksuint64_t);
        if(!rs.is_ok()) {
            return rs;
        }
        FLARE_LOG(INFO)
                << "Loaded PQ centroids and in-memory compressed vectors. #points: "
                << num_points << " #dim: " << data_dim
                << " #aligned_dim: " << aligned_dim << " #chunks: " << n_chunks;

        if (n_chunks > MAX_PQ_CHUNKS) {
            return flare::result_status(-1, "Error loading index. Ensure that max PQ bytes for in-memory "
                                            "PQ data does not exceed {}", MAX_PQ_CHUNKS);
        }

        std::string disk_pq_pivots_path = this->disk_index_file + "_pq_pivots.bin";
        if (file_exists(disk_pq_pivots_path)) {
            use_disk_index_pq = true;
            // giving 0 chunks to make the pq_table infer from the
            // chunk_offsets file the correct value
            rs = disk_pq_table.load_pq_centroid_bin(disk_pq_pivots_path.c_str(), 0);
            if(!rs.is_ok()) {
                return rs;
            }
            disk_pq_n_chunks = disk_pq_table.get_num_chunks();
            disk_bytes_per_point =
                    disk_pq_n_chunks *
                    sizeof(uint8_t);  // revising disk_bytes_per_point since DISK PQ is used.
            FLARE_LOG(INFO) << "Disk index uses PQ data compressed down to "
                            << disk_pq_n_chunks << " bytes per point.";
        }

        // read index metadata

        std::ifstream index_metadata(disk_index_file, std::ios::binary);

        uint32_t nr, nc;  // metadata itself is stored as bin format (nr is number of
        // metadata, nc should be 1)
        READ_U32(index_metadata, nr);
        READ_U32(index_metadata, nc);

        uint64_t disk_nnodes;
        uint64_t disk_ndims;  // can be disk PQ dim if disk_PQ is set to true
        READ_U64(index_metadata, disk_nnodes);
        READ_U64(index_metadata, disk_ndims);

        if (disk_nnodes != num_points) {
            FLARE_LOG(INFO) << "Mismatch in #points for compressed data file and disk "
                               "index file: "
                            << disk_nnodes << " vs " << num_points;
            return flare::result_status(-1, "Mismatch in #points for compressed data file and disk "
                                            "index file: {} vs {}",disk_nnodes, num_points);
        }

        size_t medoid_id_on_file;
        READ_U64(index_metadata, medoid_id_on_file);
        READ_U64(index_metadata, max_node_len);
        READ_U64(index_metadata, nnodes_per_sector);
        max_degree = ((max_node_len - disk_bytes_per_point) / sizeof(unsigned)) - 1;

        if (max_degree > MAX_GRAPH_DEGREE) {
            std::stringstream stream;
            stream << "Error loading index. Ensure that max graph degree (R) does "
                      "not exceed "
                   << MAX_GRAPH_DEGREE;
            return flare::result_status(-1, "Error loading index. Ensure that max graph degree (R) does "
                                            "not exceed {}", MAX_GRAPH_DEGREE);
        }

        // setting up concept of frozen points in disk index for streaming-DiskANN
        READ_U64(index_metadata, this->num_frozen_points);
        uint64_t file_frozen_id;
        READ_U64(index_metadata, file_frozen_id);
        if (this->num_frozen_points == 1)
            this->frozen_location = file_frozen_id;
        if (this->num_frozen_points == 1) {
            FLARE_LOG(INFO) << " Detected frozen point in index at location "
                            << this->frozen_location
                            << ". Will not output it at search time.";
        }

        READ_U64(index_metadata, this->reorder_data_exists);
        if (this->reorder_data_exists) {
            if (this->use_disk_index_pq == false) {
                return flare::result_status(-1, "Reordering is designed for used with disk PQ compression option");
            }
            READ_U64(index_metadata, this->reorder_data_start_sector);
            READ_U64(index_metadata, this->ndims_reorder_vecs);
            READ_U64(index_metadata, this->nvecs_per_sector);
        }

        FLARE_LOG(INFO) << "Disk-Index File Meta-data: ";
        FLARE_LOG(INFO) << "# nodes per sector: " << nnodes_per_sector;
        FLARE_LOG(INFO) << ", max node len (bytes): " << max_node_len;
        FLARE_LOG(INFO) << ", max node degree: " << max_degree;

        index_metadata.close();
        // open AlignedFileReader handle to index_file
        std::string index_fname(disk_index_file);
        reader->open(index_fname);
        this->setup_thread_data(num_threads);
        this->max_nthreads = num_threads;

        if (file_exists(medoids_file)) {
            size_t tmp_dim;
            auto rs = lambda::binary_file::load_bin<uint32_t>(medoids_file, medoids, num_medoids, tmp_dim);
            if(!rs.is_ok()) {
                return rs;
            }
            if (tmp_dim != 1) {
                return flare::result_status(-1, "Error loading medoids file. Expected bin format of m times "
                                                "1 vector of uint32_t.");
            }
            if (!file_exists(centroids_file)) {
                FLARE_LOG(INFO)
                        << "Centroid data file not found. Using corresponding vectors "
                           "for the medoids ";
                use_medoids_data_as_centroids();
            } else {
                size_t num_centroids, aligned_tmp_dim;

                auto rs = lambda::binary_file::load_aligned_bin<float>(centroids_file, centroid_data,
                                                             num_centroids, tmp_dim,
                                                             aligned_tmp_dim);
                if(!rs.is_ok()) {
                    return rs;
                }
                if (aligned_tmp_dim != aligned_dim || num_centroids != num_medoids) {
                    std::stringstream stream;
                    stream << "Error loading centroids data file. Expected bin format of "
                              "m times data_dim vector of float, where m is number of "
                              "medoids "
                              "in medoids file.";
                    FLARE_LOG(ERROR) << stream.str();
                    return flare::result_status(-1, stream.str());
                }
            }
        } else {
            num_medoids = 1;
            medoids = new uint32_t[1];
            medoids[0] = (uint32_t) (medoid_id_on_file);
            use_medoids_data_as_centroids();
        }

        std::string norm_file = std::string(disk_index_file) + "_max_base_norm.bin";

        if (file_exists(norm_file) && metric == lambda::Metric::INNER_PRODUCT) {
            size_t dumr, dumc;
            float *norm_val;
            auto rs = lambda::binary_file::load_bin<float>(norm_file, norm_val, dumr, dumc);
            if(!rs.is_ok()) {
                return rs;
            }
            this->max_base_norm = norm_val[0];
            FLARE_LOG(INFO) << "Setting re-scaling factor of base vectors to "
                            << this->max_base_norm;
            delete[] norm_val;
        }
        FLARE_LOG(INFO) << "done..";
        return flare::result_status::success();
    }

#ifdef USE_BING_INFRA
    bool getNextCompletedRequest(const IOContext &ctx, size_t size,
                                 int &completedIndex) {
      bool waitsRemaining = false;
      for (int i = 0; i < size; i++) {
        auto ithStatus = (*ctx.m_pRequestsStatus)[i];
        if (ithStatus == IOContext::Status::READ_SUCCESS) {
          completedIndex = i;
          return true;
        } else if (ithStatus == IOContext::Status::READ_WAIT) {
          waitsRemaining = true;
        }
      }
      completedIndex = -1;
      return waitsRemaining;
    }
#endif

    template<typename T>
    void pq_flash_index<T>::cached_beam_search(const T *query1, const uint64_t k_search,
                                               const uint64_t l_search, uint64_t *indices,
                                               float *distances,
                                               const uint64_t beam_width,
                                               const bool use_reorder_data,
                                               query_stats *stats) {
        cached_beam_search(query1, k_search, l_search, indices, distances,
                           beam_width, std::numeric_limits<uint32_t>::max(),
                           use_reorder_data, stats);
    }

    template<typename T>
    void pq_flash_index<T>::cached_beam_search(
            const T *query1, const uint64_t k_search, const uint64_t l_search, uint64_t *indices,
            float *distances, const uint64_t beam_width, const uint32_t io_limit,
            const bool use_reorder_data, query_stats *stats) {
        ThreadData<T> data = this->thread_data.pop();
        while (data.scratch.sector_scratch == nullptr) {
            this->thread_data.wait_for_push_notify();
            data = this->thread_data.pop();
        }

        if (beam_width > MAX_N_SECTOR_READS) {
            void(0);
            /*throw ANNException("Beamwidth can not be higher than MAX_N_SECTOR_READS",
                               -1, __FUNCSIG__, __FILE__, __LINE__);

        */
        }
        // copy query to thread specific aligned and allocated memory (for distance
        // calculations we need aligned data)
        float query_norm = 0;
        const T *query = data.scratch.aligned_query_T;
        const float *query_float = data.scratch.aligned_query_float;
        float *query_rotated = data.scratch.rotated_query;

        for (uint32_t i = 0; i < this->data_dim;
             i++) {  // need to check if this is correct for MIPS search
            if ((i == (this->data_dim - 1)) &&
                (metric == lambda::Metric::INNER_PRODUCT))
                break;
            data.scratch.aligned_query_float[i] = query1[i];
            data.scratch.aligned_query_T[i] = query1[i];
            query_rotated[i] = query1[i];
            query_norm += query1[i] * query1[i];
        }

        // if inner product, we laso normalize the query and set the last coordinate
        // to 0 (this is the extra coordindate used to convert MIPS to L2 search)
        if (metric == lambda::Metric::INNER_PRODUCT) {
            query_norm = std::sqrt(query_norm);
            data.scratch.aligned_query_T[this->data_dim - 1] = 0;
            data.scratch.aligned_query_float[this->data_dim - 1] = 0;
            query_rotated[this->data_dim - 1] = 0;
            for (uint32_t i = 0; i < this->data_dim - 1; i++) {
                data.scratch.aligned_query_T[i] /= query_norm;
                data.scratch.aligned_query_float[i] /= query_norm;
            }
        }

        IOContext &ctx = data.ctx;
        auto query_scratch = &(data.scratch);

        // reset query
        query_scratch->reset();

        // pointers to buffers for data
        T *data_buf = query_scratch->coord_scratch;
        uint64_t &data_buf_idx = query_scratch->coord_idx;
        _mm_prefetch((char *) data_buf, _MM_HINT_T1);

        // sector scratch
        char *sector_scratch = query_scratch->sector_scratch;
        uint64_t &sector_scratch_idx = query_scratch->sector_idx;

        // query <-> PQ chunk centers distances
        pq_table.preprocess_query(query_rotated);  // center the query and rotate if
        // we have a rotation matrix
        float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
        pq_table.populate_chunk_distances(query_rotated, pq_dists);

        // query <-> neighbor list
        float *dist_scratch = query_scratch->aligned_dist_scratch;
        uint8_t *pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;

        // lambda to batch compute query<-> node distances in PQ space
        auto compute_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids,
                                                                const uint64_t n_ids,
                                                                float *dists_out) {
            ::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
                               pq_coord_scratch);
            ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                             dists_out);
        };
        flare::stop_watcher query_timer, io_timer, cpu_timer;
        std::vector<neighbor> retset(l_search + 1);
        flare::robin_set<uint64_t> &visited = *(query_scratch->visited);

        std::vector<neighbor> full_retset;
        full_retset.reserve(4096);
        uint32_t best_medoid = 0;
        float best_dist = (std::numeric_limits<float>::max)();
        std::vector<simple_neighbor> medoid_dists;
        for (uint64_t cur_m = 0; cur_m < num_medoids; cur_m++) {
            float cur_expanded_dist = dist_cmp_float->compare(
                    query_float, centroid_data + aligned_dim * cur_m,
                    (unsigned) aligned_dim);
            if (cur_expanded_dist < best_dist) {
                best_medoid = medoids[cur_m];
                best_dist = cur_expanded_dist;
            }
        }

        compute_dists(&best_medoid, 1, dist_scratch);
        retset[0].id = best_medoid;
        retset[0].distance = dist_scratch[0];
        retset[0].flag = true;
        visited.insert(best_medoid);

        unsigned cur_list_size = 1;

        std::sort(retset.begin(), retset.begin() + cur_list_size);

        unsigned cmps = 0;
        unsigned hops = 0;
        unsigned num_ios = 0;
        unsigned k = 0;

        // cleared every iteration
        std::vector<unsigned> frontier;
        frontier.reserve(2 * beam_width);
        std::vector<std::pair<unsigned, char *>> frontier_nhoods;
        frontier_nhoods.reserve(2 * beam_width);
        std::vector<AlignedRead> frontier_read_reqs;
        frontier_read_reqs.reserve(2 * beam_width);
        std::vector<std::pair<unsigned, std::pair<unsigned, unsigned *>>>
                cached_nhoods;
        cached_nhoods.reserve(2 * beam_width);

        while (k < cur_list_size && num_ios < io_limit) {
            auto nk = cur_list_size;
            // clear iteration state
            frontier.clear();
            frontier_nhoods.clear();
            frontier_read_reqs.clear();
            cached_nhoods.clear();
            sector_scratch_idx = 0;
            // find new beam
            uint32_t marker = k;
            uint32_t num_seen = 0;
            while (marker < cur_list_size && frontier.size() < beam_width &&
                   num_seen < beam_width) {
                if (retset[marker].flag) {
                    num_seen++;
                    auto iter = nhood_cache.find(retset[marker].id);
                    if (iter != nhood_cache.end()) {
                        cached_nhoods.push_back(
                                std::make_pair(retset[marker].id, iter->second));
                        if (stats != nullptr) {
                            stats->n_cache_hits++;
                        }
                    } else {
                        frontier.push_back(retset[marker].id);
                    }
                    retset[marker].flag = false;
                    if (this->count_visited_nodes) {
                        reinterpret_cast<std::atomic<uint32_t> &>(
                                this->node_visit_counter[retset[marker].id].second)
                                .fetch_add(1);
                    }
                }
                marker++;
            }

            // read nhoods of frontier ids
            if (!frontier.empty()) {
                if (stats != nullptr)
                    stats->n_hops++;
                for (uint64_t i = 0; i < frontier.size(); i++) {
                    auto id = frontier[i];
                    std::pair<uint32_t, char *> fnhood;
                    fnhood.first = id;
                    fnhood.second = sector_scratch + sector_scratch_idx * SECTOR_LEN;
                    sector_scratch_idx++;
                    frontier_nhoods.push_back(fnhood);
                    frontier_read_reqs.emplace_back(
                            NODE_SECTOR_NO(((size_t) id)) * SECTOR_LEN, SECTOR_LEN,
                            fnhood.second);
                    if (stats != nullptr) {
                        stats->n_4k++;
                        stats->n_ios++;
                    }
                    num_ios++;
                }
                io_timer.start();
#ifdef USE_BING_INFRA
                reader->read(frontier_read_reqs, ctx, true);  // async reader windows.
#else
                reader->read(frontier_read_reqs, ctx);  // synchronous IO linux
#endif
                if (stats != nullptr) {
                    stats->io_us += (double) io_timer.u_elapsed();
                }
            }

            // process cached nhoods
            for (auto &cached_nhood : cached_nhoods) {
                auto global_cache_iter = coord_cache.find(cached_nhood.first);
                T *node_fp_coords_copy = global_cache_iter->second;
                float cur_expanded_dist;
                if (!use_disk_index_pq) {
                    cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy,
                                                          (unsigned) aligned_dim);
                } else {
                    if (metric == lambda::Metric::INNER_PRODUCT)
                        cur_expanded_dist = disk_pq_table.inner_product(
                                query_float, (uint8_t *) node_fp_coords_copy);
                    else
                        cur_expanded_dist =
                                disk_pq_table.l2_distance(  // disk_pq does not support OPQ yet
                                        query_float, (uint8_t *) node_fp_coords_copy);
                }
                full_retset.push_back(
                        neighbor((unsigned) cached_nhood.first, cur_expanded_dist, true));

                uint64_t nnbrs = cached_nhood.second.first;
                unsigned *node_nbrs = cached_nhood.second.second;

                // compute node_nbrs <-> query dists in PQ space
                cpu_timer.start();
                compute_dists(node_nbrs, nnbrs, dist_scratch);
                if (stats != nullptr) {
                    stats->n_cmps += (double) nnbrs;
                    stats->cpu_us += (double) cpu_timer.u_elapsed();
                }

                // process prefetched nhood
                for (uint64_t m = 0; m < nnbrs; ++m) {
                    unsigned id = node_nbrs[m];
                    if (visited.find(id) != visited.end()) {
                        continue;
                    } else {
                        visited.insert(id);
                        cmps++;
                        float dist = dist_scratch[m];
                        if (dist >= retset[cur_list_size - 1].distance &&
                            (cur_list_size == l_search))
                            continue;
                        neighbor nn(id, dist, true);
                        // Return position in sorted list where nn inserted.
                        auto r = insert_into_pool(retset.data(), cur_list_size, nn);
                        if (cur_list_size < l_search)
                            ++cur_list_size;
                        if (r < nk)
                            // nk logs the best position in the retset that was
                            // updated due to neighbors of n.
                            nk = r;
                    }
                }
            }
#ifdef USE_BING_INFRA
            // process each frontier nhood - compute distances to unvisited nodes
            int completedIndex = -1;
            // If we issued read requests and if a read is complete or there are reads
            // in wait state, then enter the while loop.
            while (frontier_read_reqs.size() > 0 &&
                   getNextCompletedRequest(ctx, frontier_read_reqs.size(),
                                           completedIndex)) {
              if (completedIndex == -1) {  // all reads are waiting
                continue;
              }
              auto &frontier_nhood = frontier_nhoods[completedIndex];
              (*ctx.m_pRequestsStatus)[completedIndex] = IOContext::PROCESS_COMPLETE;
#else
            for (auto &frontier_nhood : frontier_nhoods) {
#endif
                char *node_disk_buf =
                        OFFSET_TO_NODE(frontier_nhood.second, frontier_nhood.first);
                unsigned *node_buf = OFFSET_TO_NODE_NHOOD(node_disk_buf);
                uint64_t nnbrs = (uint64_t) (*node_buf);
                T *node_fp_coords = OFFSET_TO_NODE_COORDS(node_disk_buf);
                //        assert(data_buf_idx < MAX_N_CMPS);
                if (data_buf_idx == MAX_N_CMPS)
                    data_buf_idx = 0;

                T *node_fp_coords_copy = data_buf + (data_buf_idx * aligned_dim);
                data_buf_idx++;
                memcpy(node_fp_coords_copy, node_fp_coords, disk_bytes_per_point);
                float cur_expanded_dist;
                if (!use_disk_index_pq) {
                    cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy,
                                                          (unsigned) aligned_dim);
                } else {
                    if (metric == lambda::Metric::INNER_PRODUCT)
                        cur_expanded_dist = disk_pq_table.inner_product(
                                query_float, (uint8_t *) node_fp_coords_copy);
                    else
                        cur_expanded_dist = disk_pq_table.l2_distance(
                                query_float, (uint8_t *) node_fp_coords_copy);
                }
                full_retset.push_back(
                        neighbor(frontier_nhood.first, cur_expanded_dist, true));
                unsigned *node_nbrs = (node_buf + 1);
                // compute node_nbrs <-> query dist in PQ space
                cpu_timer.start();
                compute_dists(node_nbrs, nnbrs, dist_scratch);
                if (stats != nullptr) {
                    stats->n_cmps += (double) nnbrs;
                    stats->cpu_us += (double) cpu_timer.u_elapsed();
                }

                cpu_timer.start();
                // process prefetch-ed nhood
                for (uint64_t m = 0; m < nnbrs; ++m) {
                    unsigned id = node_nbrs[m];
                    if (visited.find(id) != visited.end()) {
                        continue;
                    } else {
                        visited.insert(id);
                        cmps++;
                        float dist = dist_scratch[m];
                        if (stats != nullptr) {
                            stats->n_cmps++;
                        }
                        if (dist >= retset[cur_list_size - 1].distance &&
                            (cur_list_size == l_search))
                            continue;
                        neighbor nn(id, dist, true);
                        auto r = insert_into_pool(
                                retset.data(), cur_list_size,
                                nn);  // Return position in sorted list where nn inserted.
                        if (cur_list_size < l_search)
                            ++cur_list_size;
                        if (r < nk)
                            nk = r;  // nk logs the best position in the retset that was
                        // updated due to neighbors of n.
                    }
                }

                if (stats != nullptr) {
                    stats->cpu_us += (double) cpu_timer.u_elapsed();
                }
            }

            // update best inserted position
            if (nk <= k)
                k = nk;  // k is the best position in retset updated in this round.
            else
                ++k;

            hops++;
        }

        // re-sort by distance
        std::sort(full_retset.begin(), full_retset.end(),
                  [](const neighbor &left, const neighbor &right) {
                      return left.distance < right.distance;
                  });

        if (use_reorder_data) {
            if (!(this->reorder_data_exists)) {
                /*throw ANNException(
                        "Requested use of reordering data which does not exist in index "
                        "file",
                        -1, __FUNCSIG__, __FILE__, __LINE__);*/
            }

            std::vector<AlignedRead> vec_read_reqs;

            if (full_retset.size() > k_search * FULL_PRECISION_REORDER_MULTIPLIER)
                full_retset.erase(
                        full_retset.begin() + k_search * FULL_PRECISION_REORDER_MULTIPLIER,
                        full_retset.end());

            for (size_t i = 0; i < full_retset.size(); ++i) {
                vec_read_reqs.emplace_back(
                        VECTOR_SECTOR_NO(((size_t) full_retset[i].id)) * SECTOR_LEN,
                        SECTOR_LEN, sector_scratch + i * SECTOR_LEN);

                if (stats != nullptr) {
                    stats->n_4k++;
                    stats->n_ios++;
                }
            }

            io_timer.start();
#ifdef USE_BING_INFRA
            reader->read(vec_read_reqs, ctx, false);  // sync reader windows.
#else
            reader->read(vec_read_reqs, ctx);  // synchronous IO linux
#endif
            if (stats != nullptr) {
                stats->io_us += (double) io_timer.u_elapsed();
            }

            for (size_t i = 0; i < full_retset.size(); ++i) {
                auto id = full_retset[i].id;
                auto location =
                        (sector_scratch + i * SECTOR_LEN) + VECTOR_SECTOR_OFFSET(id);
                full_retset[i].distance =
                        dist_cmp->compare(query, (T *) location, this->data_dim);
            }

            std::sort(full_retset.begin(), full_retset.end(),
                      [](const neighbor &left, const neighbor &right) {
                          return left.distance < right.distance;
                      });
        }

        // copy k_search values
        for (uint64_t i = 0; i < k_search; i++) {
            indices[i] = full_retset[i].id;
            if (distances != nullptr) {
                distances[i] = full_retset[i].distance;
                if (metric == lambda::Metric::INNER_PRODUCT) {
                    // flip the sign to convert min to max
                    distances[i] = (-distances[i]);
                    // rescale to revert back to original norms (cancelling the effect of
                    // base and query pre-processing)
                    if (max_base_norm != 0)
                        distances[i] *= (max_base_norm * query_norm);
                }
            }
        }

        this->thread_data.push(data);
        this->thread_data.push_notify_all();

        if (stats != nullptr) {
            stats->total_us = (double) query_timer.u_elapsed();
        }
    }

    // range search returns results of all neighbors within distance of range.
    // indices and distances need to be pre-allocated of size l_search and the
    // return value is the number of matching hits.
    template<typename T>
    uint32_t pq_flash_index<T>::range_search(const T *query1, const double range,
                                             const uint64_t min_l_search,
                                             const uint64_t max_l_search,
                                             std::vector<uint64_t> &indices,
                                             std::vector<float> &distances,
                                             const uint64_t min_beam_width,
                                             query_stats *stats) {
        uint32_t res_count = 0;

        bool stop_flag = false;

        uint32_t l_search = min_l_search;  // starting size of the candidate list
        while (!stop_flag) {
            indices.resize(l_search);
            distances.resize(l_search);
            uint64_t cur_bw =
                    min_beam_width > (l_search / 5) ? min_beam_width : l_search / 5;
            cur_bw = (cur_bw > 100) ? 100 : cur_bw;
            for (auto &x : distances)
                x = std::numeric_limits<float>::max();
            this->cached_beam_search(query1, l_search, l_search, indices.data(),
                                     distances.data(), cur_bw, false, stats);
            for (uint32_t i = 0; i < l_search; i++) {
                if (distances[i] > (float) range) {
                    res_count = i;
                    break;
                } else if (i == l_search - 1)
                    res_count = l_search;
            }
            if (res_count < (uint32_t) (l_search / 2.0))
                stop_flag = true;
            l_search = l_search * 2;
            if (l_search > max_l_search)
                stop_flag = true;
        }
        indices.resize(res_count);
        distances.resize(res_count);
        return res_count;
    }

    // instantiations
    template
    class pq_flash_index<uint8_t>;

    template
    class pq_flash_index<int8_t>;

    template
    class pq_flash_index<float>;

}  // namespace lambda
