/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "utils.h"

#include <stdio.h>

namespace lambda {

bool Avx2SupportedCPU = true;
bool AvxSupportedCPU = false;

    // Get the right distance function for the given metric.
    template<>
    vector_distance<float> *get_distance_function(lambda::Metric m) {
        if (m == lambda::Metric::L2) {
            if (Avx2SupportedCPU) {
                MELON_LOG(INFO) << "L2: Using AVX2 distance computation DistanceL2Float";
                return new DistanceL2Float();
            } else if (AvxSupportedCPU) {
                MELON_LOG(INFO)
                        << "L2: AVX2 not supported. Using AVX distance computation";
                return new AVXDistanceL2Float();
            } else {
                MELON_LOG(INFO) << "L2: Older CPU. Using slow distance computation";
                return new SlowDistanceL2Float();
            }
        } else if (m == lambda::Metric::COSINE) {
            MELON_LOG(INFO) << "Cosine: Using either AVX or AVX2 implementation";
            return new DistanceCosineFloat();
        } else if (m == lambda::Metric::INNER_PRODUCT) {
            MELON_LOG(INFO) << "Inner product: Using AVX2 implementation "
                            "AVXDistanceInnerProductFloat";
            return new AVXDistanceInnerProductFloat();
        } else if (m == lambda::Metric::FAST_L2) {
            MELON_LOG(INFO) << "Fast_L2: Using AVX2 implementation with norm "
                            "memoization DistanceFastL2<float>";
            return new DistanceFastL2<float>();
        } else {
            std::stringstream stream;
            stream << "Only L2, cosine, and inner product supported for floating "
                      "point vectors as of now. Email "
                      "{gopalsr, harshasi, rakri}@microsoft.com if you need support "
                      "for any other metric.";
            MELON_LOG(ERROR) << stream.str();
            /*throw lambda::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                       __LINE__);*/
            return nullptr;
        }
        return nullptr;
    }

    template<>
    lambda::vector_distance<int8_t> *get_distance_function(lambda::Metric m) {
        if (m == lambda::Metric::L2) {
            if (Avx2SupportedCPU) {
                MELON_LOG(INFO) << "Using AVX2 distance computation DistanceL2Int8.";
                return new lambda::DistanceL2Int8();
            } else if (AvxSupportedCPU) {
                MELON_LOG(INFO) << "AVX2 not supported. Using AVX distance computation";
                return new lambda::AVXDistanceL2Int8();
            } else {
                MELON_LOG(INFO) << "Older CPU. Using slow distance computation "
                                "SlowDistanceL2Int<int8_t>.";
                return new lambda::SlowDistanceL2Int<int8_t>();
            }
        } else if (m == lambda::Metric::COSINE) {
            MELON_LOG(INFO) << "Using either AVX or AVX2 for Cosine similarity "
                            "DistanceCosineInt8.";
            return new lambda::DistanceCosineInt8();
        } else {
            std::stringstream stream;
            stream << "Only L2 and cosine supported for signed byte vectors as of "
                      "now. Email "
                      "{gopalsr, harshasi, rakri}@microsoft.com if you need support "
                      "for any other metric.";
            MELON_LOG(ERROR) << stream.str();
            /*throw lambda::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                       __LINE__);*/
            return nullptr;
        }
    }

    template<>
    lambda::vector_distance<uint8_t> *get_distance_function(lambda::Metric m) {
        if (m == lambda::Metric::L2) {
            return new lambda::DistanceL2UInt8();
        } else if (m == lambda::Metric::COSINE) {
            MELON_LOG(INFO)
                    << "AVX/AVX2 distance function not defined for Uint8. Using "
                       "slow version SlowDistanceCosineUint8() "
                       "Contact gopalsr@microsoft.com if you need AVX/AVX2 support.";
            return new lambda::SlowDistanceCosineUInt8();
        } else {
            std::stringstream stream;
            stream << "Only L2 and cosine supported for unsigned byte vectors as of "
                      "now. Email "
                      "{gopalsr, harshasi, rakri}@microsoft.com if you need support "
                      "for any other metric.";
            MELON_LOG(ERROR) << stream.str();
            assert(false);
           /* throw lambda::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                       __LINE__);*/
        }
    }

    void block_convert(std::ofstream &writr, std::ifstream &readr,
                       float *read_buf, uint64_t npts, uint64_t ndims) {
        readr.read((char *) read_buf, npts * ndims * sizeof(float));
        uint32_t ndims_u32 = (uint32_t) ndims;
#pragma omp parallel for
        for (int64_t i = 0; i < (int64_t) npts; i++) {
            float norm_pt = std::numeric_limits<float>::epsilon();
            for (uint32_t dim = 0; dim < ndims_u32; dim++) {
                norm_pt +=
                        *(read_buf + i * ndims + dim) * *(read_buf + i * ndims + dim);
            }
            norm_pt = std::sqrt(norm_pt);
            for (uint32_t dim = 0; dim < ndims_u32; dim++) {
                *(read_buf + i * ndims + dim) = *(read_buf + i * ndims + dim) / norm_pt;
            }
        }
        writr.write((char *) read_buf, npts * ndims * sizeof(float));
    }

    void normalize_data_file(const std::string &inFileName,
                             const std::string &outFileName) {
        std::ifstream readr(inFileName, std::ios::binary);
        std::ofstream writr(outFileName, std::ios::binary);

        int npts_s32, ndims_s32;
        readr.read((char *) &npts_s32, sizeof(int32_t));
        readr.read((char *) &ndims_s32, sizeof(int32_t));

        writr.write((char *) &npts_s32, sizeof(int32_t));
        writr.write((char *) &ndims_s32, sizeof(int32_t));

        uint64_t npts = (uint64_t) npts_s32, ndims = (uint64_t) ndims_s32;
        MELON_LOG(INFO) << "Normalizing FLOAT vectors in file: " << inFileName;
        MELON_LOG(INFO) << "Dataset: #pts = " << npts << ", # dims = " << ndims;

        uint64_t blk_size = 131072;
        uint64_t nblks = ROUND_UP(npts, blk_size) / blk_size;
        MELON_LOG(INFO) << "# blks: " << nblks;

        float *read_buf = new float[npts * ndims];
        for (uint64_t i = 0; i < nblks; i++) {
            uint64_t cblk_size = std::min(npts - i * blk_size, blk_size);
            block_convert(writr, readr, read_buf, cblk_size, ndims);
        }
        delete[] read_buf;

        MELON_LOG(INFO) << "Wrote normalized points to file: " << outFileName;
    }

}  // namespace lambda