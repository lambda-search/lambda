/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include <iostream>
#include "lambda/graph/utils.h"

void block_convert(std::ifstream &reader, std::ofstream &writer,
                   float *read_buf, float *write_buf, uint64_t npts, uint64_t ndims) {
    reader.read((char *) read_buf,
                npts * (ndims * sizeof(float) + sizeof(unsigned)));
    for (uint64_t i = 0; i < npts; i++) {
        memcpy(write_buf + i * ndims, (read_buf + i * (ndims + 1)) + 1,
               ndims * sizeof(float));
    }
    writer.write((char *) write_buf, npts * ndims * sizeof(float));
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cout << argv[0] << " input_fvecs output_bin" << std::endl;
        exit(-1);
    }
    std::ifstream reader(argv[1], std::ios::binary | std::ios::ate);
    uint64_t fsize = reader.tellg();
    reader.seekg(0, std::ios::beg);

    unsigned ndims_u32;
    reader.read((char *) &ndims_u32, sizeof(unsigned));
    reader.seekg(0, std::ios::beg);
    uint64_t ndims = (uint64_t) ndims_u32;
    uint64_t npts = fsize / ((ndims + 1) * sizeof(float));
    std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims
              << std::endl;

    uint64_t blk_size = 131072;
    uint64_t nblks = ROUND_UP(npts, blk_size) / blk_size;
    std::cout << "# blks: " << nblks << std::endl;
    std::ofstream writer(argv[2], std::ios::binary);
    int npts_s32 = (int32_t) npts;
    int ndims_s32 = (int32_t) ndims;
    writer.write((char *) &npts_s32, sizeof(int32_t));
    writer.write((char *) &ndims_s32, sizeof(int32_t));
    float *read_buf = new float[npts * (ndims + 1)];
    float *write_buf = new float[npts * ndims];
    for (uint64_t i = 0; i < nblks; i++) {
        uint64_t cblk_size = std::min(npts - i * blk_size, blk_size);
        block_convert(reader, writer, read_buf, write_buf, cblk_size, ndims);
        std::cout << "Block #" << i << " written" << std::endl;
    }

    delete[] read_buf;
    delete[] write_buf;

    reader.close();
    writer.close();
}
