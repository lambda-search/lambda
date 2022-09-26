/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include <iostream>
#include "lambda/graph/utils.h"

void block_convert_float(std::ifstream &reader, std::ofstream &writer,
                         uint64_t npts, uint64_t ndims) {
    auto read_buf = new float[npts * (ndims + 1)];

    auto cursor = read_buf;
    float val;

    for (uint64_t i = 0; i < npts; i++) {
        for (uint64_t d = 0; d < ndims; ++d) {
            reader >> val;
            *cursor = val;
            cursor++;
        }
    }
    writer.write((char *) read_buf, npts * ndims * sizeof(float));
    delete[] read_buf;
}

void block_convert_int8(std::ifstream &reader, std::ofstream &writer, uint64_t npts,
                        uint64_t ndims) {
    auto read_buf = new int8_t[npts * (ndims + 1)];

    auto cursor = read_buf;
    int val;

    for (uint64_t i = 0; i < npts; i++) {
        for (uint64_t d = 0; d < ndims; ++d) {
            reader >> val;
            *cursor = (int8_t) val;
            cursor++;
        }
    }
    writer.write((char *) read_buf, npts * ndims * sizeof(uint8_t));
    delete[] read_buf;
}

void block_convert_uint8(std::ifstream &reader, std::ofstream &writer,
                         uint64_t npts, uint64_t ndims) {
    auto read_buf = new uint8_t[npts * (ndims + 1)];

    auto cursor = read_buf;
    int val;

    for (uint64_t i = 0; i < npts; i++) {
        for (uint64_t d = 0; d < ndims; ++d) {
            reader >> val;
            *cursor = (uint8_t) val;
            cursor++;
        }
    }
    writer.write((char *) read_buf, npts * ndims * sizeof(uint8_t));
    delete[] read_buf;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        std::cout << argv[0]
                  << "<float/int8/uint8> input_filename.tsv output_filename.bin "
                     "dim num_pts>"
                  << std::endl;
        exit(-1);
    }

    if (std::string(argv[1]) != std::string("float") &&
        std::string(argv[1]) != std::string("int8") &&
        std::string(argv[1]) != std::string("uint8")) {
        std::cout << "Unsupported type. float, int8 and uint8 types are supported."
                  << std::endl;
    }

    uint64_t ndims = atoi(argv[4]);
    uint64_t npts = atoi(argv[5]);

    std::ifstream reader(argv[2], std::ios::binary | std::ios::ate);
    //  uint64_t          fsize = reader.tellg();
    reader.seekg(0, std::ios::beg);
    reader.seekg(0, std::ios::beg);

    uint64_t blk_size = 131072;
    uint64_t nblks = ROUND_UP(npts, blk_size) / blk_size;
    std::cout << "# blks: " << nblks << std::endl;
    std::ofstream writer(argv[3], std::ios::binary);
    auto npts_s32 = (uint32_t) npts;
    auto ndims_s32 = (uint32_t) ndims;
    writer.write((char *) &npts_s32, sizeof(uint32_t));
    writer.write((char *) &ndims_s32, sizeof(uint32_t));

    for (uint64_t i = 0; i < nblks; i++) {
        uint64_t cblk_size = std::min(npts - i * blk_size, blk_size);
        if (std::string(argv[1]) == std::string("float")) {
            block_convert_float(reader, writer, cblk_size, ndims);
        } else if (std::string(argv[1]) == std::string("int8")) {
            block_convert_int8(reader, writer, cblk_size, ndims);
        } else if (std::string(argv[1]) == std::string("uint8")) {
            block_convert_uint8(reader, writer, cblk_size, ndims);
        }
        std::cout << "Block #" << i << " written" << std::endl;
    }

    reader.close();
    writer.close();
}
