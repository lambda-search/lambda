/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include "lambda/graph/binary_file.h"

#define NUM_PQ_CENTROIDS 256

namespace lambda {
    class fixed_chunk_pq_table {
        float *tables = nullptr;  // pq_tables = float array of size [256 * ndims]
        uint64_t ndims = 0;         // ndims = true dimension of vectors
        uint64_t n_chunks = 0;
        bool use_rotation = false;
        uint32_t *chunk_offsets = nullptr;
        float *centroid = nullptr;
        float *tables_tr = nullptr;  // same as pq_tables, but col-major
        float *rotmat_tr = nullptr;

    public:
        fixed_chunk_pq_table() {
        }

        virtual ~fixed_chunk_pq_table() {
            if (tables != nullptr)
                delete[] tables;
            if (tables_tr != nullptr)
                delete[] tables_tr;
            if (chunk_offsets != nullptr)
                delete[] chunk_offsets;
            if (centroid != nullptr)
                delete[] centroid;
            if (rotmat_tr != nullptr)
                delete[] rotmat_tr;
        }


        flare::result_status load_pq_centroid_bin(const char *pq_table_file, size_t num_chunks) {

            size_t nr, nc;
            std::string rotmat_file =
                    std::string(pq_table_file) + "_rotation_matrix.bin";
            std::unique_ptr<uint64_t[]> file_offset_data;
            auto rs = lambda::binary_file::load_bin<uint64_t>(pq_table_file, file_offset_data, nr, nc);
            if(!rs.is_ok()) {
                return rs;
            }

            if (nr != 4) {
                FLARE_LOG(INFO) << "Error reading pq_pivots file " << pq_table_file
                                << ". Offsets dont contain correct metadata, # offsets = "
                                << nr << ", but expecting " << 4;
                return flare::result_status(-1,
                        "Error reading pq_pivots file at offsets data.");
            }

            FLARE_LOG(INFO) << "Offsets: " << file_offset_data[0] << " "
                            << file_offset_data[1] << " " << file_offset_data[2] << " "
                            << file_offset_data[3];

            rs = lambda::binary_file::load_bin<float>(pq_table_file, tables, nr, nc,
                                    file_offset_data[0]);
            if(!rs.is_ok()) {
                return rs;
            }
            if ((nr != NUM_PQ_CENTROIDS)) {
                FLARE_LOG(INFO) << "Error reading pq_pivots file " << pq_table_file
                                << ". file_num_centers  = " << nr << " but expecting "
                                << NUM_PQ_CENTROIDS << " centers";
                return flare::result_status(-1,
                        "Error reading pq_pivots file at pivots data.");
            }

            this->ndims = nc;

            rs = lambda::binary_file::load_bin<float>(pq_table_file, centroid, nr, nc,
                                    file_offset_data[1]);
            if(!rs.is_ok()) {
                return rs;
            }
            if ((nr != this->ndims) || (nc != 1)) {
                FLARE_LOG(ERROR) << "Error reading centroids from pq_pivots file "
                                 << pq_table_file << ". file_dim  = " << nr
                                 << ", file_cols = " << nc << " but expecting "
                                 << this->ndims << " entries in 1 dimension.";
                return flare::result_status(-1,
                        "Error reading pq_pivots file at centroid data.");
            }

            rs = lambda::binary_file::load_bin<uint32_t>(pq_table_file, chunk_offsets, nr, nc,
                                       file_offset_data[2]);
            if(!rs.is_ok()) {
                return rs;
            }

            if (nc != 1 || (nr != num_chunks + 1 && num_chunks != 0)) {
                FLARE_LOG(ERROR) << "Error loading chunk offsets file. numc: " << nc
                                 << " (should be 1). numr: " << nr << " (should be "
                                 << num_chunks + 1 << " or 0 if we need to infer)";
                return flare::result_status(-1, "Error loading chunk offsets file");
            }

            this->n_chunks = nr - 1;
            FLARE_LOG(INFO) << "Loaded PQ Pivots: #ctrs: " << NUM_PQ_CENTROIDS
                            << ", #dims: " << this->ndims
                            << ", #chunks: " << this->n_chunks;

            if (file_exists(rotmat_file)) {
                rs = lambda::binary_file::load_bin<float>(rotmat_file, rotmat_tr, nr, nc);
                if(!rs.is_ok()) {
                    return rs;
                }
                if (nr != this->ndims || nc != this->ndims) {
                    FLARE_LOG(ERROR) << "Error loading rotation matrix file";
                    return flare::result_status(-1, "Error loading rotation matrix file");
                }
                use_rotation = true;
            }

            // alloc and compute transpose
            tables_tr = new float[256 * this->ndims];
            for (uint64_t i = 0; i < 256; i++) {
                for (uint64_t j = 0; j < this->ndims; j++) {
                    tables_tr[j * 256 + i] = tables[i * this->ndims + j];
                }
            }
            return flare::result_status::success();
        }

        uint32_t
        get_num_chunks() {
            return static_cast<uint32_t>(n_chunks);
        }

        void preprocess_query(float *query_vec) {
            for (uint32_t d = 0; d < ndims; d++) {
                query_vec[d] -= centroid[d];
            }
            std::vector<float> tmp(ndims, 0);
            if (use_rotation) {
                for (uint32_t d = 0; d < ndims; d++) {
                    for (uint32_t d1 = 0; d1 < ndims; d1++) {
                        tmp[d] += query_vec[d1] * rotmat_tr[d1 * ndims + d];
                    }
                }
                std::memcpy(query_vec, tmp.data(), ndims * sizeof(float));
            }
        }

        // assumes pre-processed query
        void populate_chunk_distances(const float *query_vec, float *dist_vec) {
            memset(dist_vec, 0, 256 * n_chunks * sizeof(float));
            // chunk wise distance computation
            for (uint64_t chunk = 0; chunk < n_chunks; chunk++) {
                // sum (q-c)^2 for the dimensions associated with this chunk
                float *chunk_dists = dist_vec + (256 * chunk);
                for (uint64_t j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
                    const float *centers_dim_vec = tables_tr + (256 * j);
                    for (uint64_t idx = 0; idx < 256; idx++) {
                        double diff = centers_dim_vec[idx] - (query_vec[j]);
                        chunk_dists[idx] += (float) (diff * diff);
                    }
                }
            }
        }

        float l2_distance(const float *query_vec, uint8_t *base_vec) {
            float res = 0;
            for (uint64_t chunk = 0; chunk < n_chunks; chunk++) {
                for (uint64_t j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
                    const float *centers_dim_vec = tables_tr + (256 * j);
                    float diff = centers_dim_vec[base_vec[chunk]] - (query_vec[j]);
                    res += diff * diff;
                }
            }
            return res;
        }

        float inner_product(const float *query_vec, uint8_t *base_vec) {
            float res = 0;
            for (uint64_t chunk = 0; chunk < n_chunks; chunk++) {
                for (uint64_t j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
                    const float *centers_dim_vec = tables_tr + (256 * j);
                    float diff = centers_dim_vec[base_vec[chunk]] *
                                 query_vec[j];  // assumes centroid is 0 to
                    // prevent translation errors
                    res += diff;
                }
            }
            return -res;  // returns negative value to simulate distances (max -> min
            // conversion)
        }

        // assumes no rotation is involved
        void inflate_vector(uint8_t *base_vec, float *out_vec) {
            for (uint64_t chunk = 0; chunk < n_chunks; chunk++) {
                for (uint64_t j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
                    const float *centers_dim_vec = tables_tr + (256 * j);
                    out_vec[j] = centers_dim_vec[base_vec[chunk]] + centroid[j];
                }
            }
        }

        void populate_chunk_inner_products(const float *query_vec, float *dist_vec) {
            memset(dist_vec, 0, 256 * n_chunks * sizeof(float));
            // chunk wise distance computation
            for (uint64_t chunk = 0; chunk < n_chunks; chunk++) {
                // sum (q-c)^2 for the dimensions associated with this chunk
                float *chunk_dists = dist_vec + (256 * chunk);
                for (uint64_t j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
                    const float *centers_dim_vec = tables_tr + (256 * j);
                    for (uint64_t idx = 0; idx < 256; idx++) {
                        double prod =
                                centers_dim_vec[idx] * query_vec[j];  // assumes that we are not
                        // shifting the vectors to
                        // mean zero, i.e., centroid
                        // array should be all zeros
                        chunk_dists[idx] -=
                                (float) prod;  // returning negative to keep the search code clean
                        // (max inner product vs min distance)
                    }
                }
            }
        }
    };
}  // namespace lambda
