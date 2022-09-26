/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include <iostream>
#include "lambda/graph/utils.h"

template<class T>
void block_convert(std::ofstream &writer, std::ifstream &reader, T *read_buf,
                   uint64_t npts, uint64_t ndims) {
    reader.read((char *) read_buf, npts * ndims * sizeof(float));

    for (uint64_t i = 0; i < npts; i++) {
        for (uint64_t d = 0; d < ndims; d++) {
            writer << read_buf[d + i * ndims];
            if (d < ndims - 1)
                writer << "\t";
            else
                writer << "\n";
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cout << argv[0] << " <float/int8/uint8> input_bin output_tsv"
                  << std::endl;
        exit(-1);
    }
    std::string type_string(argv[1]);
    if ((type_string != std::string("float")) &&
        (type_string != std::string("int8")) &&
        (type_string != std::string("uin8"))) {
        std::cerr << "Error: type not supported. Use float/int8/uint8" << std::endl;
    }

    std::ifstream reader(argv[2], std::ios::binary);
    uint32_t npts_u32;
    uint32_t ndims_u32;
    reader.read((char *) &npts_u32, sizeof(int32_t));
    reader.read((char *) &ndims_u32, sizeof(int32_t));
    size_t npts = npts_u32;
    size_t ndims = ndims_u32;
    std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims
              << std::endl;

    uint64_t blk_size = 131072;
    uint64_t nblks = ROUND_UP(npts, blk_size) / blk_size;

    std::ofstream writer(argv[3]);
    char *read_buf = new char[blk_size * ndims * 4];
    for (uint64_t i = 0; i < nblks; i++) {
        uint64_t cblk_size = std::min(npts - i * blk_size, blk_size);
        if (type_string == std::string("float"))
            block_convert<float>(writer, reader, (float *) read_buf, cblk_size, ndims);
        else if (type_string == std::string("int8"))
            block_convert<int8_t>(writer, reader, (int8_t *) read_buf, cblk_size,
                                  ndims);
        else if (type_string == std::string("uint8"))
            block_convert<uint8_t>(writer, reader, (uint8_t *) read_buf, cblk_size,
                                   ndims);
        std::cout << "Block #" << i << " written" << std::endl;
    }

    delete[] read_buf;

    writer.close();
    reader.close();
}
