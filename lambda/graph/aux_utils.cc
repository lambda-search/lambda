/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include <algorithm>
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <flare/files/filesystem.h>
#include <flare/files/sequential_read_file.h>
#include <flare/files/sequential_write_file.h>


#if defined(RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && \
    defined(DISKANN_BUILD)
#include "gperftools/malloc_extension.h"
#endif

#include "flare/log/logging.h"
#include "aux_utils.h"
#include "index.h"
#include "mkl.h"
#include "omp.h"
#include "partition_and_pq.h"
#include "percentile_stats.h"
#include "pq_flash_index.h"
#include "flare/container/robin_set.h"
#include "lambda/graph/binary_file.h"
#include "lambda/common/memory.h"

#include "lambda/graph/utils.h"

namespace lambda {

    flare::result_status add_new_file_to_single_index(std::string index_file,
                                                      std::string new_file) {
        std::unique_ptr<uint64_t[]> metadata;
        size_t nr, nc;
        auto rs = lambda::binary_file::load_bin<uint64_t>(index_file, metadata, nr, nc);
        if (!rs.is_ok()) {
            return rs;
        }
        if (nc != 1) {
            std::stringstream stream;
            stream << "Error, index file specified does not have correct metadata. ";
            return flare::result_status(-1, stream.str());
        }
        size_t index_ending_offset = metadata[nr - 1];
        uint64_t read_blk_size = 64 * 1024 * 1024;
        flare::sequential_write_file writer;
        rs = writer.open(index_file);
        if(!rs.is_ok()) {
            return rs;
        }
        std::error_code ec;
        uint64_t check_file_size = flare::file_size(index_file, ec);
        if (check_file_size != index_ending_offset) {
            std::stringstream stream;
            stream << "Error, index file specified does not have correct metadata "
                      "(last entry must match the filesize). \n";
            return flare::result_status(-1, stream.str());
        }

        flare::sequential_read_file reader;
        rs = reader.open(new_file);
        if(!rs.is_ok()) {
            return rs;
        }
        size_t fsize = flare::file_size(new_file, ec);
        if (fsize == 0) {
            std::stringstream stream;
            stream << "Error, new file specified is empty. Not appending. \n";
            return flare::result_status(-1, stream.str());
        }

        size_t num_blocks = DIV_ROUND_UP(fsize, read_blk_size);
        char *dump = new char[read_blk_size];
        for (uint64_t i = 0; i < num_blocks; i++) {
            size_t cur_block_size = read_blk_size > fsize - (i * read_blk_size)
                                    ? fsize - (i * read_blk_size)
                                    : read_blk_size;
            reader.read(dump, cur_block_size);
            writer.write(dump, cur_block_size);
        }
        //    reader.close();
        //    writer.close();

        delete[] dump;
        std::vector<uint64_t> new_meta;
        for (uint64_t i = 0; i < nr; i++)
            new_meta.push_back(metadata[i]);
        new_meta.push_back(metadata[nr - 1] + fsize);

        rs = lambda::binary_file::save_bin<uint64_t>(index_file, new_meta.data(), new_meta.size(), 1);
        return rs;
    }

    double get_memory_budget(double search_ram_budget) {
        double final_index_ram_limit = search_ram_budget;
        if (search_ram_budget - SPACE_FOR_CACHED_NODES_IN_GB >
            THRESHOLD_FOR_CACHING_IN_GB) {  // slack for space used by cached
            // nodes
            final_index_ram_limit = search_ram_budget - SPACE_FOR_CACHED_NODES_IN_GB;
        }
        return final_index_ram_limit * 1024 * 1024 * 1024;
    }

    double get_memory_budget(const std::string &mem_budget_str) {
        double search_ram_budget = atof(mem_budget_str.c_str());
        return get_memory_budget(search_ram_budget);
    }

    size_t calculate_num_pq_chunks(double final_index_ram_limit,
                                   size_t points_num, uint32_t dim,
                                   const std::vector<std::string> &param_list) {
        size_t num_pq_chunks =
                (size_t) (std::floor)(uint64_t(final_index_ram_limit / (double) points_num));
        FLARE_LOG(INFO) << "Calculated num_pq_chunks :" << num_pq_chunks;
        if (param_list.size() >= 6) {
            float compress_ratio = (float) atof(param_list[5].c_str());
            if (compress_ratio > 0 && compress_ratio <= 1) {
                size_t chunks_by_cr = (size_t) (std::floor)(compress_ratio * dim);

                if (chunks_by_cr > 0 && chunks_by_cr < num_pq_chunks) {
                    FLARE_LOG(INFO) << "Compress ratio:" << compress_ratio
                                    << " new #pq_chunks:" << chunks_by_cr;
                    num_pq_chunks = chunks_by_cr;
                } else {
                    FLARE_LOG(INFO) << "Compress ratio: " << compress_ratio
                                    << " #new pq_chunks: " << chunks_by_cr
                                    << " is either zero or greater than num_pq_chunks: "
                                    << num_pq_chunks << ". num_pq_chunks is unchanged. ";
                }
            } else {
                FLARE_LOG(ERROR) << "Compression ratio: " << compress_ratio
                                 << " should be in (0,1]";
            }
        }

        num_pq_chunks = num_pq_chunks <= 0 ? 1 : num_pq_chunks;
        num_pq_chunks = num_pq_chunks > dim ? dim : num_pq_chunks;
        num_pq_chunks =
                num_pq_chunks > MAX_PQ_CHUNKS ? MAX_PQ_CHUNKS : num_pq_chunks;

        FLARE_LOG(INFO) << "Compressing " << dim << "-dimensional data into "
                        << num_pq_chunks << " bytes per vector.";
        return num_pq_chunks;
    }

    double calculate_recall(unsigned num_queries, unsigned *gold_std,
                            float *gs_dist, unsigned dim_gs,
                            unsigned *our_results, unsigned dim_or,
                            unsigned recall_at) {
        double total_recall = 0;
        std::set<unsigned> gt, res;

        for (size_t i = 0; i < num_queries; i++) {
            gt.clear();
            res.clear();
            unsigned *gt_vec = gold_std + dim_gs * i;
            unsigned *res_vec = our_results + dim_or * i;
            size_t tie_breaker = recall_at;
            if (gs_dist != nullptr) {
                tie_breaker = recall_at - 1;
                float *gt_dist_vec = gs_dist + dim_gs * i;
                while (tie_breaker < dim_gs &&
                       gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
                    tie_breaker++;
            }

            gt.insert(gt_vec, gt_vec + tie_breaker);
            res.insert(res_vec,
                       res_vec + recall_at);  // change to recall_at for recall k@k or
            // dim_or for k@dim_or
            unsigned cur_recall = 0;
            for (auto &v : gt) {
                if (res.find(v) != res.end()) {
                    cur_recall++;
                }
            }
            total_recall += cur_recall;
        }
        return total_recall / (num_queries) * (100.0 / recall_at);
    }

    double calculate_recall(unsigned num_queries, unsigned *gold_std,
                            float *gs_dist, unsigned dim_gs,
                            unsigned *our_results, unsigned dim_or,
                            unsigned recall_at,
                            const flare::robin_set<unsigned> &active_tags) {
        double total_recall = 0;
        std::set<unsigned> gt, res;
        bool printed = false;
        for (size_t i = 0; i < num_queries; i++) {
            gt.clear();
            res.clear();
            unsigned *gt_vec = gold_std + dim_gs * i;
            unsigned *res_vec = our_results + dim_or * i;
            size_t tie_breaker = recall_at;
            unsigned active_points_count = 0;
            unsigned cur_counter = 0;
            while (active_points_count < recall_at && cur_counter < dim_gs) {
                if (active_tags.find(*(gt_vec + cur_counter)) != active_tags.end()) {
                    active_points_count++;
                }
                cur_counter++;
            }
            if (active_tags.empty())
                cur_counter = recall_at;

            if ((active_points_count < recall_at && !active_tags.empty()) &&
                !printed) {
                FLARE_LOG(INFO) << "Warning: Couldn't find enough closest neighbors "
                                << active_points_count << "/" << recall_at
                                << " from "
                                   "truthset for query # "
                                << i << ". Will result in under-reported value of recall.";
                printed = true;
            }
            if (gs_dist != nullptr) {
                tie_breaker = cur_counter - 1;
                float *gt_dist_vec = gs_dist + dim_gs * i;
                while (tie_breaker < dim_gs &&
                       gt_dist_vec[tie_breaker] == gt_dist_vec[cur_counter - 1])
                    tie_breaker++;
            }

            gt.insert(gt_vec, gt_vec + tie_breaker);
            res.insert(res_vec, res_vec + recall_at);
            unsigned cur_recall = 0;
            for (auto &v : res) {
                if (gt.find(v) != gt.end()) {
                    cur_recall++;
                }
            }
            total_recall += cur_recall;
        }
        return ((double) (total_recall / (num_queries))) *
               ((double) (100.0 / recall_at));
    }

    double calculate_range_search_recall(
            unsigned num_queries, std::vector<std::vector<uint32_t>> &groundtruth,
            std::vector<std::vector<uint32_t>> &our_results) {
        double total_recall = 0;
        std::set<unsigned> gt, res;

        for (size_t i = 0; i < num_queries; i++) {
            gt.clear();
            res.clear();

            gt.insert(groundtruth[i].begin(), groundtruth[i].end());
            res.insert(our_results[i].begin(), our_results[i].end());
            unsigned cur_recall = 0;
            for (auto &v : gt) {
                if (res.find(v) != res.end()) {
                    cur_recall++;
                }
            }
            if (gt.size() != 0)
                total_recall += ((100.0 * cur_recall) / gt.size());
            else
                total_recall += 100;
        }
        return total_recall / (num_queries);
    }

    template<typename T>
    T *generateRandomWarmup(uint64_t warmup_num, uint64_t warmup_dim,
                            uint64_t warmup_aligned_dim) {
        T *warmup = nullptr;
        warmup_num = 100000;
        FLARE_LOG(INFO) << "Generating random warmup file with dim " << warmup_dim
                        << " and aligned dim " << warmup_aligned_dim << std::flush;
        lambda::alloc_aligned(((void **) &warmup),
                              warmup_num * warmup_aligned_dim * sizeof(T),
                              8 * sizeof(T));
        std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(-128, 127);
        for (uint32_t i = 0; i < warmup_num; i++) {
            for (uint32_t d = 0; d < warmup_dim; d++) {
                warmup[i * warmup_aligned_dim + d] = (T) dis(gen);
            }
        }
        FLARE_LOG(INFO) << "..done";
        return warmup;
    }


    template<typename T>
    std::pair<flare::result_status, T *> load_warmup(const std::string &cache_warmup_file, size_t &warmup_num,
                                                     uint64_t warmup_dim, uint64_t warmup_aligned_dim) {
        T *warmup = nullptr;
        size_t file_dim, file_aligned_dim;

        if (file_exists(cache_warmup_file)) {
            auto rs = lambda::binary_file::load_aligned_bin<T>(cache_warmup_file, warmup, warmup_num,
                                                               file_dim, file_aligned_dim);
            if (!rs.is_ok()) {
                return {rs, nullptr};
            }
            if (file_dim != warmup_dim || file_aligned_dim != warmup_aligned_dim) {
                std::stringstream stream;
                stream << "Mismatched dimensions in sample file. file_dim = "
                       << file_dim << " file_aligned_dim: " << file_aligned_dim
                       << " index_dim: " << warmup_dim
                       << " index_aligned_dim: " << warmup_aligned_dim;
                return {flare::result_status(-1, stream.str()), nullptr};
            }
        } else {
            warmup =
                    generateRandomWarmup<T>(warmup_num, warmup_dim, warmup_aligned_dim);
        }
        return {flare::result_status::success(), warmup};
    }

    /***************************************************
        Support for Merging Many Vamana Indices
     ***************************************************/

    flare::result_status read_idmap(const std::string &fname, std::vector<unsigned> &ivecs) {
        uint32_t npts32, dim;
        std::error_code ec;
        size_t actual_file_size = flare::file_size(fname, ec);
        std::ifstream reader(fname.c_str(), std::ios::binary);
        reader.read((char *) &npts32, sizeof(uint32_t));
        reader.read((char *) &dim, sizeof(uint32_t));
        if (dim != 1 || actual_file_size != ((size_t) npts32) * sizeof(uint32_t) +
                                            2 * sizeof(uint32_t)) {
            std::stringstream stream;
            stream << "Error reading idmap file. Check if the file is bin file with "
                      "1 dimensional data. Actual: "
                   << actual_file_size
                   << ", expected: " << (size_t) npts32 + 2 * sizeof(uint32_t);

            return flare::result_status(-1, stream.str());
        }
        ivecs.resize(npts32);
        reader.read((char *) ivecs.data(), ((size_t) npts32) * sizeof(uint32_t));
        reader.close();
        return flare::result_status::success();
    }

    flare::result_status merge_shards(const std::string &vamana_prefix,
                     const std::string &vamana_suffix,
                     const std::string &idmaps_prefix,
                     const std::string &idmaps_suffix, const size_t nshards,
                     size_t max_degree, const std::string &output_vamana,
                     const std::string &medoids_file) {
        // Read ID maps
        std::vector<std::string> vamana_names(nshards);
        std::vector<std::vector<unsigned>> idmaps(nshards);
        for (uint64_t shard = 0; shard < nshards; shard++) {
            vamana_names[shard] =
                    vamana_prefix + std::to_string(shard) + vamana_suffix;
            auto rs = read_idmap(idmaps_prefix + std::to_string(shard) + idmaps_suffix,
                       idmaps[shard]);
            if(!rs.is_ok()) {
                return rs;
            }
        }

        // find max node id
        uint64_t nnodes = 0;
        uint64_t nelems = 0;
        for (auto &idmap : idmaps) {
            for (auto &id : idmap) {
                nnodes = std::max(nnodes, (uint64_t) id);
            }
            nelems += idmap.size();
        }
        nnodes++;
        FLARE_LOG(INFO) << "# nodes: " << nnodes << ", max. degree: " << max_degree;

        // compute inverse map: node -> shards
        std::vector<std::pair<unsigned, unsigned>> node_shard;
        node_shard.reserve(nelems);
        for (uint64_t shard = 0; shard < nshards; shard++) {
            FLARE_LOG(INFO) << "Creating inverse map -- shard #" << shard;
            for (uint64_t idx = 0; idx < idmaps[shard].size(); idx++) {
                uint64_t node_id = idmaps[shard][idx];
                node_shard.push_back(std::make_pair((uint32_t) node_id, (uint32_t) shard));
            }
        }
        std::sort(node_shard.begin(), node_shard.end(),
                  [](const auto &left, const auto &right) {
                      return left.first < right.first || (left.first == right.first &&
                                                          left.second < right.second);
                  });
        FLARE_LOG(INFO) << "Finished computing node -> shards map";

        // create cached vamana readers
        std::vector<flare::sequential_read_file> vamana_readers(nshards);
        for (uint64_t i = 0; i < nshards; i++) {
            auto rs = vamana_readers[i].open(vamana_names[i]);
            if(!rs.is_ok()) {
                return rs;
            }
            size_t expected_file_size;
            vamana_readers[i].read((char *) &expected_file_size, sizeof(uint64_t));
        }

        size_t vamana_metadata_size =
                sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint32_t) +
                sizeof(uint64_t);  // expected file size + max degree + medoid_id +
        // frozen_point info

        // create cached vamana writers
        flare::sequential_write_file merged_vamana_writer;
        auto rs = merged_vamana_writer.open(output_vamana);
        if(!rs.is_ok()) {
            return rs;
        }
        size_t merged_index_size =
                vamana_metadata_size;  // we initialize the size of the merged index to
        // the metadata size
        size_t merged_index_frozen = 0;
        merged_vamana_writer.write(
                (char *) &merged_index_size,
                sizeof(uint64_t));  // we will overwrite the index size at the end

        unsigned output_width = max_degree;
        unsigned max_input_width = 0;
        // read width from each vamana to advance buffer by sizeof(unsigned) bytes
        for (auto &reader : vamana_readers) {
            unsigned input_width;
            reader.read((char *) &input_width, sizeof(unsigned));
            max_input_width =
                    input_width > max_input_width ? input_width : max_input_width;
        }

        FLARE_LOG(INFO) << "Max input width: " << max_input_width
                        << ", output width: " << output_width;

        merged_vamana_writer.write((char *) &output_width, sizeof(unsigned));
        std::ofstream medoid_writer(medoids_file.c_str(), std::ios::binary);
        uint32_t nshards_u32 = (uint32_t) nshards;
        uint32_t one_val = 1;
        medoid_writer.write((char *) &nshards_u32, sizeof(uint32_t));
        medoid_writer.write((char *) &one_val, sizeof(uint32_t));

        uint64_t vamana_index_frozen =
                0;  // as of now the functionality to merge many overlapping vamana
        // indices is supported only for bulk indices without frozen point.
        // Hence the final index will also not have any frozen points.
        for (uint64_t shard = 0; shard < nshards; shard++) {
            unsigned medoid;
            // read medoid
            vamana_readers[shard].read((char *) &medoid, sizeof(unsigned));
            vamana_readers[shard].read((char *) &vamana_index_frozen, sizeof(uint64_t));
            assert(vamana_index_frozen == false);
            // rename medoid
            medoid = idmaps[shard][medoid];

            medoid_writer.write((char *) &medoid, sizeof(uint32_t));
            // write renamed medoid
            if (shard == (nshards - 1))  //--> uncomment if running hierarchical
                merged_vamana_writer.write((char *) &medoid, sizeof(unsigned));
        }
        merged_vamana_writer.write((char *) &merged_index_frozen, sizeof(uint64_t));
        medoid_writer.close();

        FLARE_LOG(INFO) << "Starting merge";

        // Gopal. random_shuffle() is deprecated.
        std::random_device rng;
        std::mt19937 urng(rng());

        std::vector<bool> nhood_set(nnodes, 0);
        std::vector<unsigned> final_nhood;

        unsigned nnbrs = 0, shard_nnbrs = 0;
        unsigned cur_id = 0;
        for (const auto &id_shard : node_shard) {
            unsigned node_id = id_shard.first;
            unsigned shard_id = id_shard.second;
            if (cur_id < node_id) {
                // Gopal. random_shuffle() is deprecated.
                std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
                nnbrs =
                        (unsigned) (std::min)(final_nhood.size(), (size_t) max_degree);
                // write into merged ofstream
                merged_vamana_writer.write((char *) &nnbrs, sizeof(unsigned));
                merged_vamana_writer.write((char *) final_nhood.data(),
                                           nnbrs * sizeof(unsigned));
                merged_index_size += (sizeof(unsigned) + nnbrs * sizeof(unsigned));
                if (cur_id % 499999 == 1) {
                    FLARE_LOG(INFO) << "." << std::flush;
                }
                cur_id = node_id;
                nnbrs = 0;
                for (auto &p : final_nhood)
                    nhood_set[p] = 0;
                final_nhood.clear();
            }
            // read from shard_id ifstream
            vamana_readers[shard_id].read((char *) &shard_nnbrs, sizeof(unsigned));
            std::vector<unsigned> shard_nhood(shard_nnbrs);
            vamana_readers[shard_id].read((char *) shard_nhood.data(),
                                          shard_nnbrs * sizeof(unsigned));

            // rename nodes
            for (uint64_t j = 0; j < shard_nnbrs; j++) {
                if (nhood_set[idmaps[shard_id][shard_nhood[j]]] == 0) {
                    nhood_set[idmaps[shard_id][shard_nhood[j]]] = 1;
                    final_nhood.emplace_back(idmaps[shard_id][shard_nhood[j]]);
                }
            }
        }

        // Gopal. random_shuffle() is deprecated.
        std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
        nnbrs = (unsigned) (std::min)(final_nhood.size(), (size_t) max_degree);
        // write into merged ofstream
        merged_vamana_writer.write((char *) &nnbrs, sizeof(unsigned));
        merged_vamana_writer.write((char *) final_nhood.data(),
                                   nnbrs * sizeof(unsigned));
        merged_index_size += (sizeof(unsigned) + nnbrs * sizeof(unsigned));
        for (auto &p : final_nhood)
            nhood_set[p] = 0;
        final_nhood.clear();

        FLARE_LOG(INFO) << "Expected size: " << merged_index_size;

        merged_vamana_writer.reset();
        merged_vamana_writer.write((char *) &merged_index_size, sizeof(uint64_t));

        FLARE_LOG(INFO) << "Finished merge";
        return flare::result_status::success();
    }

    template<typename T>
    flare::result_status build_merged_vamana_index(std::string base_file,
                                  lambda::Metric compareMetric, unsigned L,
                                  unsigned R, double sampling_rate,
                                  double ram_budget, std::string mem_index_path,
                                  std::string medoids_file,
                                  std::string centroids_file) {
        size_t base_num, base_dim;
        lambda::get_bin_metadata(base_file, base_num, base_dim);

        double full_index_ram =
                estimate_ram_usage(base_num, base_dim, sizeof(T), R);
        if (full_index_ram < ram_budget * 1024 * 1024 * 1024) {
            FLARE_LOG(INFO) << "Full index fits in RAM budget, should consume at most "
                            << full_index_ram / (1024 * 1024 * 1024)
                            << "GiBs, so building in one shot";
            lambda::Parameters paras;
            paras.Set<unsigned>("L", (unsigned) L);
            paras.Set<unsigned>("R", (unsigned) R);
            paras.Set<unsigned>("C", 750);
            paras.Set<float>("alpha", 1.2f);
            paras.Set<unsigned>("num_rnds", 2);
            paras.Set<bool>("saturate_graph", 1);
            paras.Set<std::string>("save_path", mem_index_path);

            std::unique_ptr<lambda::Index<T>> _pvamanaIndex =
                    std::unique_ptr<lambda::Index<T>>(new lambda::Index<T>(
                            compareMetric, base_dim, base_num, false, false));
            _pvamanaIndex->build(base_file.c_str(), base_num, paras);

            _pvamanaIndex->save(mem_index_path.c_str());
            std::remove(medoids_file.c_str());
            std::remove(centroids_file.c_str());
            return flare::result_status::success();
        }
        std::string merged_index_prefix = mem_index_path + "_tempFiles";
        int num_parts;
        auto rs = partition_with_ram_budget<T>(base_file, sampling_rate, ram_budget,
                                     2 * R / 3, merged_index_prefix, 2, &num_parts);
        if(!rs.is_ok()) {
            return rs;
        }

        std::string cur_centroid_filepath = merged_index_prefix + "_centroids.bin";
        std::rename(cur_centroid_filepath.c_str(), centroids_file.c_str());

        for (int p = 0; p < num_parts; p++) {
            std::string shard_base_file =
                    merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";

            std::string shard_ids_file = merged_index_prefix + "_subshard-" +
                                         std::to_string(p) + "_ids_uint32.bin";

            rs = retrieve_shard_data_from_ids<T>(base_file, shard_ids_file,
                                            shard_base_file);
            if(!rs.is_ok()) {
                return rs;
            }

            std::string shard_index_file =
                    merged_index_prefix + "_subshard-" + std::to_string(p) + "_mem.index";

            lambda::Parameters paras;
            paras.Set<unsigned>("L", L);
            paras.Set<unsigned>("R", (2 * (R / 3)));
            paras.Set<unsigned>("C", 750);
            paras.Set<float>("alpha", 1.2f);
            paras.Set<unsigned>("num_rnds", 2);
            paras.Set<bool>("saturate_graph", 0);
            paras.Set<std::string>("save_path", shard_index_file);

            size_t shard_base_dim, shard_base_pts;
            get_bin_metadata(shard_base_file, shard_base_pts, shard_base_dim);
            std::unique_ptr<lambda::Index<T>> _pvamanaIndex =
                    std::unique_ptr<lambda::Index<T>>(
                            new lambda::Index<T>(compareMetric, shard_base_dim,
                                                 shard_base_pts, false));  // TODO: Single?
            _pvamanaIndex->build(shard_base_file.c_str(), shard_base_pts, paras);
            _pvamanaIndex->save(shard_index_file.c_str());
            std::remove(shard_base_file.c_str());
        }

        rs = lambda::merge_shards(merged_index_prefix + "_subshard-", "_mem.index",
                             merged_index_prefix + "_subshard-", "_ids_uint32.bin",
                             num_parts, R, mem_index_path, medoids_file);
        if(!rs.is_ok()) {
            return rs;
        }
        // delete tempFiles
        for (int p = 0; p < num_parts; p++) {
            std::string shard_base_file =
                    merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";
            std::string shard_id_file = merged_index_prefix + "_subshard-" +
                                        std::to_string(p) + "_ids_uint32.bin";
            std::string shard_index_file =
                    merged_index_prefix + "_subshard-" + std::to_string(p) + "_mem.index";
            std::string shard_index_file_data = shard_index_file + ".data";

            std::remove(shard_base_file.c_str());
            std::remove(shard_id_file.c_str());
            std::remove(shard_index_file.c_str());
            std::remove(shard_index_file_data.c_str());
        }
        return flare::result_status::success();
    }

    // General purpose support for DiskANN interface

    // optimizes the beamwidth to maximize QPS for a given L_search subject to
    // 99.9 latency not blowing up
    template<typename T>
    uint32_t optimize_beamwidth(
            std::unique_ptr<lambda::pq_flash_index<T>> &pFlashIndex, T *tuning_sample,
            uint64_t tuning_sample_num, uint64_t tuning_sample_aligned_dim, uint32_t L,
            uint32_t nthreads, uint32_t start_bw) {
        uint32_t cur_bw = start_bw;
        double max_qps = 0;
        uint32_t best_bw = start_bw;
        bool stop_flag = false;

        while (!stop_flag) {
            std::vector<uint64_t> tuning_sample_result_ids_64(tuning_sample_num, 0);
            std::vector<float> tuning_sample_result_dists(tuning_sample_num, 0);
            lambda::QueryStats *stats = new lambda::QueryStats[tuning_sample_num];

            auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
            for (int64_t i = 0; i < (int64_t) tuning_sample_num; i++) {
                pFlashIndex->cached_beam_search(
                        tuning_sample + (i * tuning_sample_aligned_dim), 1, L,
                        tuning_sample_result_ids_64.data() + (i * 1),
                        tuning_sample_result_dists.data() + (i * 1), cur_bw, false,
                        stats + i);
            }
            auto e = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = e - s;
            double qps =
                    (1.0f * (float) tuning_sample_num) / (1.0f * (float) diff.count());

            double lat_999 = lambda::get_percentile_stats<float>(
                    stats, tuning_sample_num, 0.999f,
                    [](const lambda::QueryStats &stats) { return stats.total_us; });

            double mean_latency = lambda::get_mean_stats<float>(
                    stats, tuning_sample_num,
                    [](const lambda::QueryStats &stats) { return stats.total_us; });

            if (qps > max_qps && lat_999 < (15000) + mean_latency * 2) {
                max_qps = qps;
                best_bw = cur_bw;
                cur_bw = (uint32_t) (std::ceil)((float) cur_bw * 1.1f);
            } else {
                stop_flag = true;
            }
            if (cur_bw > 64)
                stop_flag = true;

            delete[] stats;
        }
        return best_bw;
    }

    template<typename T>
    flare::result_status create_disk_layout(const std::string base_file,
                                            const std::string mem_index_file,
                                            const std::string output_file,
                                            const std::string reorder_data_file) {
        unsigned npts, ndims;

        flare::sequential_read_file base_reader;
        auto rs = base_reader.open(base_file);
        if(!rs.is_ok()) {
            return rs;
        }
        base_reader.read((char *) &npts, sizeof(uint32_t));
        base_reader.read((char *) &ndims, sizeof(uint32_t));

        size_t npts_64, ndims_64;
        npts_64 = npts;
        ndims_64 = ndims;

        // Check if we need to append data for re-ordering
        bool append_reorder_data = false;
        std::ifstream reorder_data_reader;

        unsigned npts_reorder_file = 0, ndims_reorder_file = 0;
        if (reorder_data_file != std::string("")) {
            append_reorder_data = true;
            std::error_code ec;
            size_t reorder_data_file_size = flare::file_size(reorder_data_file, ec);
            reorder_data_reader.exceptions(std::ofstream::failbit |
                                           std::ofstream::badbit);

            try {
                reorder_data_reader.open(reorder_data_file, std::ios::binary);
                reorder_data_reader.read((char *) &npts_reorder_file, sizeof(unsigned));
                reorder_data_reader.read((char *) &ndims_reorder_file,
                                         sizeof(unsigned));
                if (npts_reorder_file != npts)
                    return flare::result_status(-1,
                                                "Mismatch in num_points between reorder data file and base file");
                if (reorder_data_file_size != 8 + sizeof(float) *
                                                  (size_t) npts_reorder_file *
                                                  (size_t) ndims_reorder_file)
                    return flare::result_status(-1, "Discrepancy in reorder data file size ");
            } catch (std::system_error &e) {
                return flare::result_status(-2, reorder_data_file);
            }
        }

        // create cached reader + writer
        std::error_code ec;
        size_t actual_file_size = flare::file_size(mem_index_file, ec);
        FLARE_LOG(INFO) << "Vamana index file size=" << actual_file_size;
        std::ifstream vamana_reader(mem_index_file, std::ios::binary);
        flare::sequential_write_file diskann_writer;
        rs = diskann_writer.open(output_file);
        if(!rs.is_ok()) {
            return rs;
        }

        // metadata: width, medoid
        unsigned width_u32, medoid_u32;
        size_t index_file_size;

        vamana_reader.read((char *) &index_file_size, sizeof(uint64_t));
        if (index_file_size != actual_file_size) {
            std::stringstream stream;
            stream << "Vamana Index file size does not match expected size per "
                      "meta-data."
                   << " file size from file: " << index_file_size
                   << " actual file size: " << actual_file_size;

            return flare::result_status(-1, stream.str());
        }
        uint64_t vamana_frozen_num = false, vamana_frozen_loc = 0;

        vamana_reader.read((char *) &width_u32, sizeof(unsigned));
        vamana_reader.read((char *) &medoid_u32, sizeof(unsigned));
        vamana_reader.read((char *) &vamana_frozen_num, sizeof(uint64_t));
        // compute
        uint64_t medoid, max_node_len, nnodes_per_sector;
        npts_64 = (uint64_t) npts;
        medoid = (uint64_t) medoid_u32;
        if (vamana_frozen_num == 1)
            vamana_frozen_loc = medoid;
        max_node_len =
                (((uint64_t) width_u32 + 1) * sizeof(unsigned)) + (ndims_64 * sizeof(T));
        nnodes_per_sector = SECTOR_LEN / max_node_len;

        FLARE_LOG(INFO) << "medoid: " << medoid << "B";
        FLARE_LOG(INFO) << "max_node_len: " << max_node_len << "B";
        FLARE_LOG(INFO) << "nnodes_per_sector: " << nnodes_per_sector << "B";

        // SECTOR_LEN buffer for each sector
        std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(SECTOR_LEN);
        std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(max_node_len);
        unsigned &nnbrs = *(unsigned *) (node_buf.get() + ndims_64 * sizeof(T));
        unsigned *nhood_buf =
                (unsigned *) (node_buf.get() + (ndims_64 * sizeof(T)) +
                              sizeof(unsigned));

        // number of sectors (1 for meta data)
        uint64_t n_sectors = ROUND_UP(npts_64, nnodes_per_sector) / nnodes_per_sector;
        uint64_t n_reorder_sectors = 0;
        uint64_t n_data_nodes_per_sector = 0;

        if (append_reorder_data) {
            n_data_nodes_per_sector =
                    SECTOR_LEN / (ndims_reorder_file * sizeof(float));
            n_reorder_sectors =
                    ROUND_UP(npts_64, n_data_nodes_per_sector) / n_data_nodes_per_sector;
        }
        uint64_t disk_index_file_size =
                (n_sectors + n_reorder_sectors + 1) * SECTOR_LEN;

        std::vector<uint64_t> output_file_meta;
        output_file_meta.push_back(npts_64);
        output_file_meta.push_back(ndims_64);
        output_file_meta.push_back(medoid);
        output_file_meta.push_back(max_node_len);
        output_file_meta.push_back(nnodes_per_sector);
        output_file_meta.push_back(vamana_frozen_num);
        output_file_meta.push_back(vamana_frozen_loc);
        output_file_meta.push_back((uint64_t) append_reorder_data);
        if (append_reorder_data) {
            output_file_meta.push_back(n_sectors + 1);
            output_file_meta.push_back(ndims_reorder_file);
            output_file_meta.push_back(n_data_nodes_per_sector);
        }
        output_file_meta.push_back(disk_index_file_size);

        diskann_writer.write(sector_buf.get(), SECTOR_LEN);

        std::unique_ptr<T[]> cur_node_coords = std::make_unique<T[]>(ndims_64);
        FLARE_LOG(INFO) << "# sectors: " << n_sectors;
        uint64_t cur_node_id = 0;
        for (uint64_t sector = 0; sector < n_sectors; sector++) {
            if (sector % 100000 == 0) {
                FLARE_LOG(INFO) << "Sector #" << sector << "written";
            }
            memset(sector_buf.get(), 0, SECTOR_LEN);
            for (uint64_t sector_node_id = 0;
                 sector_node_id < nnodes_per_sector && cur_node_id < npts_64;
                 sector_node_id++) {
                memset(node_buf.get(), 0, max_node_len);
                // read cur node's nnbrs
                vamana_reader.read((char *) &nnbrs, sizeof(unsigned));

                // sanity checks on nnbrs
                assert(nnbrs > 0);
                assert(nnbrs <= width_u32);

                // read node's nhood
                vamana_reader.read((char *) nhood_buf,
                                   (std::min)(nnbrs, width_u32) * sizeof(unsigned));
                if (nnbrs > width_u32) {
                    vamana_reader.seekg((nnbrs - width_u32) * sizeof(unsigned),
                                        vamana_reader.cur);
                }

                // write coords of node first
                //  T *node_coords = data + ((uint64_t) ndims_64 * cur_node_id);
                base_reader.read((char *) cur_node_coords.get(), sizeof(T) * ndims_64);
                memcpy(node_buf.get(), cur_node_coords.get(), ndims_64 * sizeof(T));

                // write nnbrs
                *(unsigned *) (node_buf.get() + ndims_64 * sizeof(T)) =
                        (std::min)(nnbrs, width_u32);

                // write nhood next
                memcpy(node_buf.get() + ndims_64 * sizeof(T) + sizeof(unsigned),
                       nhood_buf, (std::min)(nnbrs, width_u32) * sizeof(unsigned));

                // get offset into sector_buf
                char *sector_node_buf =
                        sector_buf.get() + (sector_node_id * max_node_len);

                // copy node buf into sector_node_buf
                memcpy(sector_node_buf, node_buf.get(), max_node_len);
                cur_node_id++;
            }
            // flush sector to disk
            diskann_writer.write(sector_buf.get(), SECTOR_LEN);
        }
        if (append_reorder_data) {
            FLARE_LOG(INFO) << "Index written. Appending reorder data...";

            auto vec_len = ndims_reorder_file * sizeof(float);
            std::unique_ptr<char[]> vec_buf = std::make_unique<char[]>(vec_len);

            for (uint64_t sector = 0; sector < n_reorder_sectors; sector++) {
                if (sector % 100000 == 0) {
                    FLARE_LOG(INFO) << "Reorder data Sector #" << sector << "written";
                }

                memset(sector_buf.get(), 0, SECTOR_LEN);

                for (uint64_t sector_node_id = 0;
                     sector_node_id < n_data_nodes_per_sector &&
                     sector_node_id < npts_64;
                     sector_node_id++) {
                    memset(vec_buf.get(), 0, vec_len);
                    reorder_data_reader.read(vec_buf.get(), vec_len);

                    // copy node buf into sector_node_buf
                    memcpy(sector_buf.get() + (sector_node_id * vec_len), vec_buf.get(),
                           vec_len);
                }
                // flush sector to disk
                diskann_writer.write(sector_buf.get(), SECTOR_LEN);
            }
        }
        diskann_writer.close();
        rs = lambda::binary_file::save_bin<uint64_t>(output_file, output_file_meta.data(),
                                                          output_file_meta.size(), 1, 0);
        FLARE_LOG(INFO) << "Output disk index file written to " << output_file;
        return rs;
    }

    template<typename T>
    flare::result_status build_disk_index(const char *dataFilePath, const char *indexFilePath,
                                          const char *indexBuildParameters,
                                          lambda::Metric compareMetric, bool use_opq) {
        std::stringstream parser;
        parser << std::string(indexBuildParameters);
        std::string cur_param;
        std::vector<std::string> param_list;
        while (parser >> cur_param) {
            param_list.push_back(cur_param);
        }
        if (param_list.size() != 5 && param_list.size() != 6 &&
            param_list.size() != 7) {
            FLARE_LOG(INFO)
                    << "Correct usage of parameters is R (max degree) "
                       "L (indexing list size, better if >= R)"
                       "B (RAM limit of final index in GB)"
                       "M (memory limit while indexing)"
                       "T (number of threads for indexing)"
                       "B' (PQ bytes for disk index: optional parameter for "
                       "very large dimensional data)"
                       "reorder (set true to include full precision in data file"
                       ": optional paramter, use only when using disk PQ";
            return flare::result_status(-1, "");
        }

        if (!std::is_same<T, float>::value &&
            compareMetric == lambda::Metric::INNER_PRODUCT) {
            std::stringstream stream;
            FLARE_LOG(ERROR) << "DiskANN currently only supports floating point data for Max "
                                "Inner Product Search. ";
            //throw lambda::ANNException(stream.str(), -1);
            return flare::result_status(-1, stream.str());
        }

        uint32_t disk_pq_dims = 0;
        bool use_disk_pq = false;

        // if there is a 6th parameter, it means we compress the disk index
        // vectors also using PQ data (for very large dimensionality data). If the
        // provided parameter is 0, it means we store full vectors.
        if (param_list.size() == 6 || param_list.size() == 7) {
            disk_pq_dims = atoi(param_list[5].c_str());
            use_disk_pq = true;
            if (disk_pq_dims == 0)
                use_disk_pq = false;
        }

        bool reorder_data = false;
        if (param_list.size() == 7) {
            if (1 == atoi(param_list[6].c_str())) {
                reorder_data = true;
            }
        }

        std::string base_file(dataFilePath);
        std::string data_file_to_use = base_file;
        std::string index_prefix_path(indexFilePath);
        std::string pq_pivots_path = index_prefix_path + "_pq_pivots.bin";
        std::string pq_compressed_vectors_path =
                index_prefix_path + "_pq_compressed.bin";
        std::string mem_index_path = index_prefix_path + "_mem.index";
        std::string disk_index_path = index_prefix_path + "_disk.index";
        std::string medoids_path = disk_index_path + "_medoids.bin";
        std::string centroids_path = disk_index_path + "_centroids.bin";
        std::string sample_base_prefix = index_prefix_path + "_sample";
        // optional, used if disk index file must store pq data
        std::string disk_pq_pivots_path =
                index_prefix_path + "_disk.index_pq_pivots.bin";
        // optional, used if disk index must store pq data
        std::string disk_pq_compressed_vectors_path =
                index_prefix_path + "_disk.index_pq_compressed.bin";

        // output a new base file which contains extra dimension with sqrt(1 -
        // ||x||^2/M^2) for every x, M is max norm of all points. Extra space on
        // disk needed!
        if (compareMetric == lambda::Metric::INNER_PRODUCT) {
            FLARE_LOG(INFO) << "Using Inner Product search, so need to pre-process base "
                               "data into temp file. Please ensure there is additional "
                               "(n*(d+1)*4) bytes for storing pre-processed base vectors, "
                               "apart from the intermin indices and final index.";
            std::string prepped_base = index_prefix_path + "_prepped_base.bin";
            data_file_to_use = prepped_base;
            float max_norm_of_base =
                    lambda::prepare_base_for_inner_products<T>(base_file, prepped_base);
            std::string norm_file = disk_index_path + "_max_base_norm.bin";
            auto rs = lambda::binary_file::save_bin<float>(norm_file, &max_norm_of_base, 1, 1);
            if (!rs.is_ok()) {
                return rs;
            }
        }

        unsigned R = (unsigned) atoi(param_list[0].c_str());
        unsigned L = (unsigned) atoi(param_list[1].c_str());

        double final_index_ram_limit = get_memory_budget(param_list[2]);
        if (final_index_ram_limit <= 0) {
            FLARE_LOG(ERROR) << "Insufficient memory budget (or string was not in right "
                                "format). Should be > 0.";
            return flare::result_status(-1, "");
        }
        double indexing_ram_budget = (float) atof(param_list[3].c_str());
        if (indexing_ram_budget <= 0) {
            FLARE_LOG(ERROR) << "Not building index. Please provide more RAM budget";
            return flare::result_status(-1, "");
        }
        uint32_t num_threads = (uint32_t) atoi(param_list[4].c_str());

        if (num_threads != 0) {
            omp_set_num_threads(num_threads);
            mkl_set_num_threads(num_threads);
        }

        FLARE_LOG(INFO) << "Starting index build: R=" << R << " L=" << L
                        << " Query RAM budget: " << final_index_ram_limit
                        << " Indexing ram budget: " << indexing_ram_budget
                        << " T: " << num_threads;

        auto s = std::chrono::high_resolution_clock::now();

        size_t points_num, dim;

        lambda::get_bin_metadata(data_file_to_use.c_str(), points_num, dim);

        size_t num_pq_chunks =
                (size_t) (std::floor)(uint64_t(final_index_ram_limit / points_num));

        num_pq_chunks = num_pq_chunks <= 0 ? 1 : num_pq_chunks;
        num_pq_chunks = num_pq_chunks > dim ? dim : num_pq_chunks;
        num_pq_chunks =
                num_pq_chunks > MAX_PQ_CHUNKS ? MAX_PQ_CHUNKS : num_pq_chunks;

        FLARE_LOG(INFO) << "Compressing " << dim << "-dimensional data into "
                        << num_pq_chunks << " bytes per vector.";

        size_t train_size, train_dim;
        float *train_data;

        double p_val = ((double) MAX_PQ_TRAINING_SET_SIZE / (double) points_num);
        // generates random sample and sets it to train_data and updates
        // train_size
        gen_random_slice<T>(data_file_to_use.c_str(), p_val, train_data, train_size,
                            train_dim);

        if (use_disk_pq) {
            if (disk_pq_dims > dim)
                disk_pq_dims = dim;

            FLARE_LOG(INFO) << "Compressing base for disk-PQ into " << disk_pq_dims
                            << " chunks ";
            auto rs = generate_pq_pivots(train_data, train_size, (uint32_t) dim, 256,
                               (uint32_t) disk_pq_dims, NUM_KMEANS_REPS,
                               disk_pq_pivots_path, false);
            if(!rs.is_ok()) {
                return rs;
            }
            if (compareMetric == lambda::Metric::INNER_PRODUCT) {
                rs = generate_pq_data_from_pivots<float>(
                        data_file_to_use.c_str(), 256, (uint32_t) disk_pq_dims,
                        disk_pq_pivots_path, disk_pq_compressed_vectors_path);
            } else {
                rs = generate_pq_data_from_pivots<T>(
                        data_file_to_use.c_str(), 256, (uint32_t) disk_pq_dims,
                        disk_pq_pivots_path, disk_pq_compressed_vectors_path);
            }
            if(!rs.is_ok()) {
                return rs;
            }
        }
        FLARE_LOG(INFO) << "Training data loaded of size " << train_size;

        // don't translate data to make zero mean for PQ compression. We must not
        // translate for inner product search.
        bool make_zero_mean = true;
        if (compareMetric == lambda::Metric::INNER_PRODUCT)
            make_zero_mean = false;
        if (use_opq)  // we also do not center the data for OPQ
            make_zero_mean = false;

        flare::result_status rs;
        if (!use_opq) {
            rs = generate_pq_pivots(train_data, train_size, (uint32_t) dim, 256,
                               (uint32_t) num_pq_chunks, NUM_KMEANS_REPS,
                               pq_pivots_path, make_zero_mean);
        } else {
            rs = generate_opq_pivots(train_data, train_size, (uint32_t) dim, 256,
                                (uint32_t) num_pq_chunks, pq_pivots_path, make_zero_mean);
        }
        if(!rs.is_ok()) {
            return rs;
        }
        rs = generate_pq_data_from_pivots<T>(data_file_to_use.c_str(), 256,
                                        (uint32_t) num_pq_chunks, pq_pivots_path,
                                        pq_compressed_vectors_path, use_opq);
        if(!rs.is_ok()) {
            return rs;
        }

        delete[] train_data;

        train_data = nullptr;
// Gopal. Splitting diskann_dll into separate DLLs for search and build.
// This code should only be available in the "build" DLL.
#if defined(RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && \
    defined(DISKANN_BUILD)
        MallocExtension::instance()->ReleaseFreeMemory();
#endif

        rs = lambda::build_merged_vamana_index<T>(
                data_file_to_use.c_str(), lambda::Metric::L2, L, R, p_val,
                indexing_ram_budget, mem_index_path, medoids_path, centroids_path);
        if(!rs.is_ok()) {
            return rs;
        }

        if (!use_disk_pq) {
            rs = lambda::create_disk_layout<T>(data_file_to_use.c_str(), mem_index_path,
                                          disk_index_path);
        } else {
            if (!reorder_data)
                rs = lambda::create_disk_layout<uint8_t>(disk_pq_compressed_vectors_path,
                                                    mem_index_path, disk_index_path);
            else
                rs = lambda::create_disk_layout<uint8_t>(disk_pq_compressed_vectors_path,
                                                    mem_index_path, disk_index_path,
                                                    data_file_to_use.c_str());
        }
        if(!rs.is_ok()) {
            return rs;
        }

        double ten_percent_points = std::ceil(points_num * 0.1);
        double num_sample_points = ten_percent_points > MAX_SAMPLE_POINTS_FOR_WARMUP
                                   ? MAX_SAMPLE_POINTS_FOR_WARMUP
                                   : ten_percent_points;
        double sample_sampling_rate = num_sample_points / points_num;
        gen_random_slice<T>(data_file_to_use.c_str(), sample_base_prefix,
                            sample_sampling_rate);

        std::remove(mem_index_path.c_str());
        if (use_disk_pq)
            std::remove(disk_pq_compressed_vectors_path.c_str());

        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        FLARE_LOG(INFO) << "Indexing time: " << diff.count();

        return flare::result_status::success();
    }

    template FLARE_EXPORT flare::result_status create_disk_layout<int8_t>(
            const std::string base_file, const std::string mem_index_file,
            const std::string output_file, const std::string reorder_data_file);

    template FLARE_EXPORT flare::result_status create_disk_layout<uint8_t>(
            const std::string base_file, const std::string mem_index_file,
            const std::string output_file, const std::string reorder_data_file);

    template FLARE_EXPORT flare::result_status create_disk_layout<float>(
            const std::string base_file, const std::string mem_index_file,
            const std::string output_file, const std::string reorder_data_file);

    template FLARE_EXPORT std::pair<flare::result_status, int8_t *> load_warmup<int8_t>(
            const std::string &cache_warmup_file, size_t &warmup_num,
            uint64_t warmup_dim, uint64_t warmup_aligned_dim);

    template FLARE_EXPORT std::pair<flare::result_status, uint8_t *> load_warmup<uint8_t>(
            const std::string &cache_warmup_file, size_t &warmup_num,
            uint64_t warmup_dim, uint64_t warmup_aligned_dim);

    template FLARE_EXPORT std::pair<flare::result_status, float *> load_warmup<float>(
            const std::string &cache_warmup_file, size_t &warmup_num,
            uint64_t warmup_dim, uint64_t warmup_aligned_dim);


    template FLARE_EXPORT uint32_t optimize_beamwidth<int8_t>(
            std::unique_ptr<lambda::pq_flash_index<int8_t>> &pFlashIndex,
            int8_t *tuning_sample, uint64_t tuning_sample_num,
            uint64_t tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads,
            uint32_t start_bw);

    template FLARE_EXPORT uint32_t optimize_beamwidth<uint8_t>(
            std::unique_ptr<lambda::pq_flash_index<uint8_t>> &pFlashIndex,
            uint8_t *tuning_sample, uint64_t tuning_sample_num,
            uint64_t tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads,
            uint32_t start_bw);

    template FLARE_EXPORT uint32_t optimize_beamwidth<float>(
            std::unique_ptr<lambda::pq_flash_index<float>> &pFlashIndex,
            float *tuning_sample, uint64_t tuning_sample_num,
            uint64_t tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads,
            uint32_t start_bw);

    template FLARE_EXPORT flare::result_status build_disk_index<int8_t>(
            const char *dataFilePath, const char *indexFilePath,
            const char *indexBuildParameters, lambda::Metric compareMetric,
            bool use_opq);

    template FLARE_EXPORT flare::result_status build_disk_index<uint8_t>(
            const char *dataFilePath, const char *indexFilePath,
            const char *indexBuildParameters, lambda::Metric compareMetric,
            bool use_opq);

    template FLARE_EXPORT flare::result_status build_disk_index<float>(
            const char *dataFilePath, const char *indexFilePath,
            const char *indexBuildParameters, lambda::Metric compareMetric,
            bool use_opq);

    template FLARE_EXPORT flare::result_status build_merged_vamana_index<int8_t>(
            std::string base_file, lambda::Metric compareMetric, unsigned L,
            unsigned R, double sampling_rate, double ram_budget,
            std::string mem_index_path, std::string medoids_path,
            std::string centroids_file);

    template FLARE_EXPORT flare::result_status build_merged_vamana_index<float>(
            std::string base_file, lambda::Metric compareMetric, unsigned L,
            unsigned R, double sampling_rate, double ram_budget,
            std::string mem_index_path, std::string medoids_path,
            std::string centroids_file);

    template FLARE_EXPORT flare::result_status build_merged_vamana_index<uint8_t>(
            std::string base_file, lambda::Metric compareMetric, unsigned L,
            unsigned R, double sampling_rate, double ram_budget,
            std::string mem_index_path, std::string medoids_path,
            std::string centroids_file);
};  // namespace lambda
