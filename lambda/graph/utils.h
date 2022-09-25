/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#pragma once

#include <immintrin.h>
#include <fcntl.h>
#include <errno.h>
#include <cassert>
#include <cstdlib>
#include <limits.h>

#include <string>
#include <memory>
#include <set>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <sys/stat.h>
#include <time.h>
#include <vector>
#include <melon/base/profile.h>
#include <melon/base/result_status.h>
#include "lambda/common/math_utils.h"
#include <melon/files/sequential_read_file.h>
#include <melon/files/sequential_write_file.h>

#ifndef MELON_PLATFORM_OSX
#include <malloc.h>
#endif

#include <unistd.h>
#include "lambda/common/vector_distance.h"
#include "melon/log/logging.h"
#include <melon/base/profile.h>

namespace lambda {

#define METADATA_SIZE \
  4096  // all metadata of individual sub-component files is written in first
// 4KB for unified files

#define BUFFER_SIZE_FOR_CACHED_IO (uint64_t) 1024 * (uint64_t) 1048576

    inline bool file_exists(const std::string &name, bool dirCheck = false) {
        int val;
        struct stat buffer;
        val = stat(name.c_str(), &buffer);

        if (val != 0) {
            switch (errno) {
                case EINVAL:
                    MELON_LOG(INFO) << "Invalid argument passed to stat()";
                    break;
                case ENOENT:
                    // file is not existing, not an issue, so we won't cout anything.
                    break;
                default:
                    MELON_LOG(INFO) << "Unexpected error in stat():" << errno;
                    break;
            }
            return false;
        } else {
            // the file entry exists. If reqd, check if this is a directory.
            return dirCheck ? buffer.st_mode & S_IFDIR : true;
        }
    }

    enum Metric {
        L2 = 0, INNER_PRODUCT = 1, COSINE = 2, FAST_L2 = 3, PQ = 4
    };



    inline void check_stop(std::string arnd) {
        int brnd;
        MELON_LOG(INFO) << arnd;
        std::cin >> brnd;
    }



    inline void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size,
                          unsigned N) {
        for (unsigned i = 0; i < size; ++i) {
            addr[i] = rng() % (N - size);
        }

        std::sort(addr, addr + size);
        for (unsigned i = 1; i < size; ++i) {
            if (addr[i] <= addr[i - 1]) {
                addr[i] = addr[i - 1] + 1;
            }
        }
        unsigned off = rng() % N;
        for (unsigned i = 0; i < size; ++i) {
            addr[i] = (addr[i] + off) % N;
        }
    }

    // get_bin_metadata functions START
    inline void get_bin_metadata_impl(std::basic_istream<char> &reader,
                                      size_t &nrows, size_t &ncols,
                                      size_t offset = 0) {
        int nrows_32, ncols_32;
        reader.seekg(offset, reader.beg);
        reader.read((char *) &nrows_32, sizeof(int));
        reader.read((char *) &ncols_32, sizeof(int));
        nrows = nrows_32;
        ncols = ncols_32;
    }

    inline void get_bin_metadata(const std::string &bin_file, size_t &nrows,
                                 size_t &ncols, size_t offset = 0) {
        std::ifstream reader(bin_file.c_str(), std::ios::binary);
        get_bin_metadata_impl(reader, nrows, ncols, offset);
    }
    // get_bin_metadata functions END

    template<typename T>
    inline std::string getValues(T *data, size_t num) {
        std::stringstream stream;
        stream << "[";
        for (size_t i = 0; i < num; i++) {
            stream << std::to_string(data[i]) << ",";
        }
        stream << "]\n";

        return stream.str();
    }


    inline void wait_for_keystroke() {
        int a;
        std::cout << "Press any number to continue..\n";
        std::cin >> a;
    }

    inline melon::result_status load_truthset(const std::string &bin_file, uint32_t *&ids,
                                              float *&dists, size_t &npts, size_t &dim) {
        melon::sequential_read_file reader;
        auto rs = reader.open(bin_file);
        if(!rs.is_ok()) {
            return rs;
        }
        MELON_LOG(INFO) << "Reading truthset file " << bin_file.c_str() << " ...";
        std::error_code ec;
        size_t actual_file_size = melon::file_size(bin_file, ec);

        int npts_i32, dim_i32;
        reader.read((char *) &npts_i32, sizeof(int));
        reader.read((char *) &dim_i32, sizeof(int));
        npts = (unsigned) npts_i32;
        dim = (unsigned) dim_i32;

        MELON_LOG(INFO) << "Metadata: #pts = " << npts << ", #dims = " << dim
                        << "... ";

        int truthset_type = -1;  // 1 means truthset has ids and distances, 2 means
        // only ids, -1 is error
        size_t expected_file_size_with_dists =
                2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

        if (actual_file_size == expected_file_size_with_dists)
            truthset_type = 1;

        size_t expected_file_size_just_ids =
                npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

        if (actual_file_size == expected_file_size_just_ids)
            truthset_type = 2;

        if (truthset_type == -1) {
            std::stringstream stream;
            stream << "Error. File size mismatch. File should have bin format, with "
                      "npts followed by ngt followed by npts*ngt ids and optionally "
                      "followed by npts*ngt distance values; actual size: "
                   << actual_file_size
                   << ", expected: " << expected_file_size_with_dists << " or "
                   << expected_file_size_just_ids;
            MELON_LOG(INFO) << stream.str();
            return melon::result_status(-1, stream.str());
        }

        ids = new uint32_t[npts * dim];
        reader.read((char *) ids, npts * dim * sizeof(uint32_t));

        if (truthset_type == 1) {
            dists = new float[npts * dim];
            reader.read((char *) dists, npts * dim * sizeof(float));
        }
        return melon::result_status::success();
    }

    inline melon::result_status prune_truthset_for_range(
            const std::string &bin_file, float range,
            std::vector<std::vector<uint32_t>> &groundtruth, size_t &npts) {
        melon::sequential_read_file reader;
        auto rs = reader.open(bin_file);
        if(!rs.is_ok()) {
            return rs;
        }
        MELON_LOG(INFO) << "Reading truthset file " << bin_file.c_str() << "... ";
        size_t actual_file_size = melon::file_size(bin_file);

        int npts_i32, dim_i32;
        reader.read((char *) &npts_i32, sizeof(int));
        reader.read((char *) &dim_i32, sizeof(int));
        npts = (unsigned) npts_i32;
        uint64_t dim = (unsigned) dim_i32;
        uint32_t *ids;
        float *dists;

        MELON_LOG(INFO) << "Metadata: #pts = " << npts << ", #dims = " << dim << "... ";

        int truthset_type = -1;  // 1 means truthset has ids and distances, 2 means
        // only ids, -1 is error
        size_t expected_file_size_with_dists =
                2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

        if (actual_file_size == expected_file_size_with_dists)
            truthset_type = 1;

        if (truthset_type == -1) {
            std::stringstream stream;
            stream << "Error. File size mismatch. File should have bin format, with "
                      "npts followed by ngt followed by npts*ngt ids and optionally "
                      "followed by npts*ngt distance values; actual size: "
                   << actual_file_size
                   << ", expected: " << expected_file_size_with_dists;
            MELON_LOG(INFO) << stream.str();
            return melon::result_status(-1, stream.str());
        }

        ids = new uint32_t[npts * dim];
        reader.read((char *) ids, npts * dim * sizeof(uint32_t));

        if (truthset_type == 1) {
            dists = new float[npts * dim];
            reader.read((char *) dists, npts * dim * sizeof(float));
        }
        float min_dist = std::numeric_limits<float>::max();
        float max_dist = 0;
        groundtruth.resize(npts);
        for (uint32_t i = 0; i < npts; i++) {
            groundtruth[i].clear();
            for (uint32_t j = 0; j < dim; j++) {
                if (dists[i * dim + j] <= range) {
                    groundtruth[i].emplace_back(ids[i * dim + j]);
                }
                min_dist =
                        min_dist > dists[i * dim + j] ? dists[i * dim + j] : min_dist;
                max_dist =
                        max_dist < dists[i * dim + j] ? dists[i * dim + j] : max_dist;
            }
            // std::cout<<groundtruth[i].size() << " " ;
        }
        std::cout << "Min dist: " << min_dist << ", Max dist: " << max_dist;
        delete[] ids;
        delete[] dists;
        return melon::result_status::success();
    }

    inline melon::result_status load_range_truthset(const std::string &bin_file,
                                                    std::vector<std::vector<uint32_t>> &groundtruth,
                                                    uint64_t &gt_num) {
        melon::sequential_read_file reader;
        auto rs = reader.open(bin_file);
        if(!rs.is_ok()) {
            return rs;
        }
        MELON_LOG(INFO) << "Reading truthset file " << bin_file.c_str() << "... ";
        std::error_code ec;
        size_t actual_file_size = melon::file_size(bin_file, ec);
        if(ec) {
            return melon::result_status::from_error_code(ec);
        }

        int npts_u32, total_u32;
        reader.read((char *) &npts_u32, sizeof(int));
        reader.read((char *) &total_u32, sizeof(int));

        gt_num = (uint64_t) npts_u32;
        uint64_t total_res = (uint64_t) total_u32;

        MELON_LOG(INFO) << "Metadata: #pts = " << gt_num
                        << ", #total_results = " << total_res << "...";

        size_t expected_file_size =
                2 * sizeof(uint32_t) + gt_num * sizeof(uint32_t) + total_res * sizeof(uint32_t);

        if (actual_file_size != expected_file_size) {
            std::stringstream stream;
            stream << "Error. File size mismatch in range truthset. actual size: "
                   << actual_file_size << ", expected: " << expected_file_size;
            MELON_LOG(INFO) << stream.str();
            return melon::result_status(-1, stream.str());
        }
        groundtruth.clear();
        groundtruth.resize(gt_num);
        std::vector<uint32_t> gt_count(gt_num);

        reader.read((char *) gt_count.data(), sizeof(uint32_t) * gt_num);

        std::vector<uint32_t> gt_stats(gt_count);
        std::sort(gt_stats.begin(), gt_stats.end());

        MELON_LOG(INFO) << "GT count percentiles:";
        for (uint32_t p = 0; p < 100; p += 5)
            MELON_LOG(INFO)
                    << "percentile " << p << ": "
                    << gt_stats[static_cast<size_t>(std::floor((p / 100.0) * gt_num))];
        MELON_LOG(INFO) << "percentile 100"
                        << ": " << gt_stats[gt_num - 1];

        for (uint32_t i = 0; i < gt_num; i++) {
            groundtruth[i].clear();
            groundtruth[i].resize(gt_count[i]);
            if (gt_count[i] != 0)
                reader.read((char *) groundtruth[i].data(), sizeof(uint32_t) * gt_count[i]);
        }
        return melon::result_status::success();
    }

    template<typename InType, typename OutType>
    void convert_types(const InType *srcmat, OutType *destmat, size_t npts,
                       size_t dim) {
#pragma omp parallel for schedule(static, 65536)
        for (int64_t i = 0; i < (int64_t) npts; i++) {
            for (uint64_t j = 0; j < dim; j++) {
                destmat[i * dim + j] = (OutType) srcmat[i * dim + j];
            }
        }
    }

    // this function will take in_file of n*d dimensions and save the output as a
    // floating point matrix
    // with n*(d+1) dimensions. All vectors are scaled by a large value M so that
    // the norms are <=1 and the final coordinate is set so that the resulting
    // norm (in d+1 coordinates) is equal to 1 this is a classical transformation
    // from MIPS to L2 search from "On Symmetric and Asymmetric LSHs for Inner
    // Product Search" by Neyshabur and Srebro

    template<typename T>
    float prepare_base_for_inner_products(const std::string in_file,
                                          const std::string out_file) {
        MELON_LOG(DEBUG) << "Pre-processing base file by adding extra coordinate";
        std::ifstream in_reader(in_file.c_str(), std::ios::binary);
        std::ofstream out_writer(out_file.c_str(), std::ios::binary);
        uint64_t npts, in_dims, out_dims;
        float max_norm = 0;

        uint32_t npts32, dims32;
        in_reader.read((char *) &npts32, sizeof(uint32_t));
        in_reader.read((char *) &dims32, sizeof(uint32_t));

        npts = npts32;
        in_dims = dims32;
        out_dims = in_dims + 1;
        uint32_t outdims32 = (uint32_t) out_dims;

        out_writer.write((char *) &npts32, sizeof(uint32_t));
        out_writer.write((char *) &outdims32, sizeof(uint32_t));

        size_t BLOCK_SIZE = 100000;
        size_t block_size = npts <= BLOCK_SIZE ? npts : BLOCK_SIZE;
        std::unique_ptr<T[]> in_block_data =
                std::make_unique<T[]>(block_size * in_dims);
        std::unique_ptr<float[]> out_block_data =
                std::make_unique<float[]>(block_size * out_dims);

        std::memset(out_block_data.get(), 0, sizeof(float) * block_size * out_dims);
        uint64_t num_blocks = DIV_ROUND_UP(npts, block_size);

        std::vector<float> norms(npts, 0);

        for (uint64_t b = 0; b < num_blocks; b++) {
            uint64_t start_id = b * block_size;
            uint64_t end_id = (b + 1) * block_size < npts ? (b + 1) * block_size : npts;
            uint64_t block_pts = end_id - start_id;
            in_reader.read((char *) in_block_data.get(),
                           block_pts * in_dims * sizeof(T));
            for (uint64_t p = 0; p < block_pts; p++) {
                for (uint64_t j = 0; j < in_dims; j++) {
                    norms[start_id + p] +=
                            in_block_data[p * in_dims + j] * in_block_data[p * in_dims + j];
                }
                max_norm =
                        max_norm > norms[start_id + p] ? max_norm : norms[start_id + p];
            }
        }

        max_norm = std::sqrt(max_norm);

        in_reader.seekg(2 * sizeof(uint32_t), std::ios::beg);
        for (uint64_t b = 0; b < num_blocks; b++) {
            uint64_t start_id = b * block_size;
            uint64_t end_id = (b + 1) * block_size < npts ? (b + 1) * block_size : npts;
            uint64_t block_pts = end_id - start_id;
            in_reader.read((char *) in_block_data.get(),
                           block_pts * in_dims * sizeof(T));
            for (uint64_t p = 0; p < block_pts; p++) {
                for (uint64_t j = 0; j < in_dims; j++) {
                    out_block_data[p * out_dims + j] =
                            in_block_data[p * in_dims + j] / max_norm;
                }
                float res = 1 - (norms[start_id + p] / (max_norm * max_norm));
                res = res <= 0 ? 0 : std::sqrt(res);
                out_block_data[p * out_dims + out_dims - 1] = res;
            }
            out_writer.write((char *) out_block_data.get(),
                             block_pts * out_dims * sizeof(float));
        }
        out_writer.close();
        return max_norm;
    }

    // plain saves data as npts X ndims array into filename
    template<typename T>
    melon::result_status save_Tvecs(const char *filename, T *data, size_t npts, size_t ndims) {
        std::string fname(filename);

        // create cached ofstream with 64MB cache
        melon::sequential_write_file writer;

        auto rs = writer.open(fname);
        if(!rs.is_ok()) {
            return rs;
        }

        unsigned dims_u32 = (unsigned) ndims;

        // start writing
        for (uint64_t i = 0; i < npts; i++) {
            // write dims in u32
            writer.write((char *) &dims_u32, sizeof(unsigned));

            // get cur point in data
            T *cur_pt = data + i * ndims;
            writer.write((char *) cur_pt, ndims * sizeof(T));
        }
        return melon::result_status::success();
    }

    template<typename T>
    inline uint64_t save_data_in_base_dimensions(const std::string &filename,
                                                 T *data, size_t npts,
                                                 size_t ndims, size_t aligned_dim,
                                                 size_t offset = 0) {
        melon::sequential_write_file writer;  //(filename, std::ios::binary | std::ios::out);
        writer.open(filename, false);
        int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
        uint64_t bytes_written = 2 * sizeof(uint32_t) + npts * ndims * sizeof(T);
        writer.reset(offset);
        writer.write((char *) &npts_i32, sizeof(int));
        writer.write((char *) &ndims_i32, sizeof(int));
        for (size_t i = 0; i < npts; i++) {
            writer.write((char *) (data + i * aligned_dim), ndims * sizeof(T));
        }
        writer.close();
        return bytes_written;
    }

    template<typename T>
    inline melon::result_status copy_aligned_data_from_file(const char *bin_file, T *&data,
                                                            size_t &npts, size_t &dim,
                                                            const size_t &rounded_dim,
                                                            size_t offset = 0) {
        if (data == nullptr) {
            MELON_LOG(ERROR) << "Memory was not allocated for " << data
                             << " before calling the load function. Exiting...";
            return melon::result_status(-1, "Null pointer passed to copy_aligned_data_from_file function");
        }
        std::ifstream reader;
        reader.exceptions(std::ios::badbit | std::ios::failbit);
        reader.open(bin_file, std::ios::binary);
        reader.seekg(offset, reader.beg);

        int npts_i32, dim_i32;
        reader.read((char *) &npts_i32, sizeof(int));
        reader.read((char *) &dim_i32, sizeof(int));
        npts = (unsigned) npts_i32;
        dim = (unsigned) dim_i32;

        for (size_t i = 0; i < npts; i++) {
            reader.read((char *) (data + i * rounded_dim), dim * sizeof(T));
            memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
        }
        return melon::result_status::success();
    }



    // NOTE: Implementation in utils.cpp.
    void block_convert(std::ofstream &writr, std::ifstream &readr,
                       float *read_buf, uint64_t npts, uint64_t ndims);

    MELON_EXPORT void normalize_data_file(const std::string &inFileName,
                                          const std::string &outFileName);

    template<typename T>
    vector_distance<T> *get_distance_function(Metric m);



    inline melon::result_status validate_index_file_size(std::ifstream &in) {
        if (!in.is_open())
            return melon::result_status(-1,
                                        "Index file size check called on unopened file stream");
        in.seekg(0, in.end);
        size_t actual_file_size = in.tellg();
        in.seekg(0, in.beg);
        size_t expected_file_size;
        in.read((char *) &expected_file_size, sizeof(uint64_t));
        in.seekg(0, in.beg);
        if (actual_file_size != expected_file_size) {
            MELON_LOG(ERROR) << "Index file size error. Expected size (metadata): "
                             << expected_file_size
                             << ", actual file size : " << actual_file_size << ".";
            return melon::result_status(-1, "bad file size");
        }
        return melon::result_status::success();
    }




// need to check and change this
    inline bool avx2Supported() {
        return true;
    }

    inline void printProcessMemory(const char *) {
    }

    inline size_t
    getMemoryUsage() {  // for non-windows, we have not implemented this function
        return 0;
    }


    extern bool AvxSupportedCPU;
    extern bool Avx2SupportedCPU;

}  // namespace lambda
