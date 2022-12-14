/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include <iostream>
#include "utils.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << argv[0] << " input_uint32_bin output_int8_bin" << std::endl;
    exit(-1);
  }

  uint32_t* input;
  size_t    npts, nd;
  diskann::load_bin<uint32_t>(argv[1], input, npts, nd);
  uint8_t* output = new uint8_t[npts * nd];
  diskann::convert_types<uint32_t, uint8_t>(input, output, npts, nd);
  diskann::save_bin<uint8_t>(argv[2], output, npts, nd);
  delete[] output;
  delete[] input;
}
