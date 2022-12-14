//
// Created by liyinbin on 2022/9/23.
//

#ifndef LAMBDA_GRAPH_BINARY_FILE_H_
#define LAMBDA_GRAPH_BINARY_FILE_H_

#include <melon/log/logging.h>
#include <melon/base/result_status.h>
#include <melon/files/sequential_read_file.h>
#include <melon/files/sequential_write_file.h>
#include <melon/files/random_write_file.h>
#include <melon/base/math.h>
#include "lambda/common/memory.h"
#include "lambda/common/math_utils.h"

namespace lambda {


    class binary_file {
    public:
        template<typename T>
        [[nodiscard]] static melon::result_status load_bin(const std::string &bin_file, T *&data, size_t &npts,
                                                           size_t &dim, size_t offset = 0) {
            MELON_LOG(INFO) << "Reading bin file " << bin_file.c_str() << " ...";
            melon::sequential_read_file file;
            auto rs = file.open(bin_file);
            if(!rs.is_ok()) {
                return rs;
            }

            rs = file.skip(offset);
            if(!rs.is_ok()) {
                return rs;
            }
            std::string header;
            rs = file.read(&header, 2 * sizeof(int));
            if(!rs.is_ok()) {
                return rs;
            }
            if(header.size() != 2* sizeof(int)) {
                return melon::result_status(-1, "bad binary file format with header size: {}", header.size());
            }
            int npts_i32, dim_i32;
            npts_i32 = *((int*)header.data());
            dim_i32 = *((int*)header.data() + sizeof(int));
            npts = (unsigned) npts_i32;
            dim = (unsigned) dim_i32;
            MELON_LOG(INFO) << "Metadata: #pts = " << npts << ", #dims = " << dim << "...";
            melon::cord_buf content;
            rs = file.read(&content, npts * dim * sizeof(T));
            if(!rs.is_ok()) {
                return rs;
            }
            data = new T[npts * dim];
            content.cutn(data, npts * dim * sizeof(T));
            MELON_LOG(INFO) << "done.";
            return melon::result_status::success();
        }

        template<typename T>
        [[nodiscard]] static melon::result_status load_bin(const std::string &bin_file, std::unique_ptr<T[]> &data,
                                                           size_t &npts, size_t &dim, size_t offset = 0) {
            T *ptr;
            auto rs = load_bin<T>(bin_file, ptr, npts, dim, offset);
            if (!rs.is_ok()) {
                return rs;
            }
            data.reset(ptr);
            return melon::result_status::success();
        }

        template<typename T>
        [[nodiscard]] static melon::result_status save_bin(const std::string &filename, T *data, size_t npts,
                                 size_t ndims, size_t offset = 0, size_t *has_written = nullptr) {
            {
                melon::random_write_file file;
                auto rs = file.open(filename, false);
                if(!rs.is_ok()) {
                    return rs;
                }
                MELON_LOG(INFO) << "Writing bin: " << filename;
                size_t offsset_start = offset;
                uint32_t npts_i32 = (uint32_t) npts, ndims_i32 = (uint32_t) ndims;
                size_t bytes_written = npts * ndims * sizeof(T) + 2 * sizeof(uint32_t);
                rs = file.write(offsset_start, std::string_view((const char *) &npts_i32, sizeof(uint32_t)));
                if(!rs.is_ok()) {
                    return rs;
                }
                offsset_start += sizeof(uint32_t);
                rs = file.write(offsset_start, std::string_view((const char *) &ndims_i32, sizeof(uint32_t)));
                if(!rs.is_ok()) {
                    return rs;
                }
                offsset_start += sizeof(uint32_t);

                MELON_LOG(INFO) << "bin: #pts = " << npts << ", #dims = " << ndims
                                << ", size = " << bytes_written << "B";

                rs = file.write(offsset_start, std::string_view((char *) data, npts * ndims * sizeof(T)));
                if(!rs.is_ok()) {
                    return rs;
                }
                if(has_written) {
                    *has_written = bytes_written;
                }
                file.flush();
            }
            MELON_LOG(INFO) << "Finished writing bin.";
            return melon::result_status::success();
        }

        template<typename T>
        [[nodiscard]] static melon::result_status load_aligned_bin(const std::string &bin_file, T *&data,
                                                     size_t &npts, size_t &dim, size_t &rounded_dim) {
            std::error_code ec;
            melon::sequential_read_file file;
            auto rs = file.open(bin_file);
            if(!rs.is_ok()) {
                return rs;
            }
            std::string header;
            rs = file.read(&header, 2 * sizeof(int));
            if(!rs.is_ok()) {
                return rs;
            }
            if(header.size() != 2* sizeof(int)) {
                return melon::result_status(-1, "bad binary file format with header size: {}", header.size());
            }
            int npts_i32, dim_i32;
            npts_i32 = *((int*)header.data());
            dim_i32 = *((int*)(header.data() + sizeof(int)));
            npts = (unsigned) npts_i32;
            dim = (unsigned) dim_i32;
            auto actual_file_size = melon::file_size(bin_file, ec);
            size_t expected_actual_file_size =
                    npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
            if (actual_file_size != expected_actual_file_size) {
                std::stringstream stream;
                stream << "Error. File: "<<bin_file<<" size mismatch. Actual size is " << actual_file_size
                       << " while expected size is  " << expected_actual_file_size
                       << " npts = " << npts << " dim = " << dim
                       << " size of <T>= " << sizeof(T);
                MELON_LOG(ERROR) << stream.str();
                return melon::result_status(-1, stream.str());
            }
            rounded_dim = ROUND_UP(dim, 8);
            MELON_LOG(INFO) << "Metadata: #pts = " << npts << ", #dims = " << dim
                            << ", aligned_dim = " << rounded_dim << "... " << std::flush;
            size_t allocSize = npts * rounded_dim * sizeof(T);
            MELON_LOG(INFO) << "allocating aligned memory of " << allocSize
                            << " bytes... " << std::flush;
            alloc_aligned(((void **) &data), allocSize, 8 * sizeof(T));
            MELON_LOG(INFO) << "done. Copying data to mem_aligned buffer..."
                            << std::flush;
            melon::cord_buf buf;
            for (size_t i = 0; i < npts; i++) {
                buf.clear();
                rs = file.read(&buf,  dim * sizeof(T));
                if(!rs.is_ok()) {
                    return rs;
                }
                buf.cutn((char *) (data + i * rounded_dim), dim * sizeof(T));
                memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
            }
            MELON_LOG(INFO) << " done.";
            return melon::result_status::success();
        }

    };
}  // namespace lambda

#endif // LAMBDA_GRAPH_BINARY_FILE_H_
