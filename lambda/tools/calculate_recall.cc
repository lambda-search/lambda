/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "aux_utils.h"
#include "utils.h"

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << argv[0] << " <ground_truth_bin> <our_results_bin>  <r> "
              << std::endl;
    return -1;
  }
  unsigned* gold_std = NULL;
  float*    gs_dist = nullptr;
  unsigned* our_results = NULL;
  float*    or_dist = nullptr;
  size_t    points_num, points_num_gs, points_num_or;
  size_t    dim_gs;
  size_t    dim_or;
  diskann::load_truthset(argv[1], gold_std, gs_dist, points_num_gs, dim_gs);
  diskann::load_truthset(argv[2], our_results, or_dist, points_num_or, dim_or);

  if (points_num_gs != points_num_or) {
    std::cout
        << "Error. Number of queries mismatch in ground truth and our results"
        << std::endl;
    return -1;
  }
  points_num = points_num_gs;

  uint32_t recall_at = std::atoi(argv[3]);

  if ((dim_or < recall_at) || (recall_at > dim_gs)) {
    std::cout << "ground truth has size " << dim_gs << "; our set has "
              << dim_or << " points. Asking for recall " << recall_at
              << std::endl;
    return -1;
  }
  std::cout << "Calculating recall@" << recall_at << std::endl;
  float recall_val = diskann::calculate_recall(
      points_num, gold_std, gs_dist, dim_gs, our_results, dim_or, recall_at);

  //  double avg_recall = (recall*1.0)/(points_num*1.0);
  std::cout << "Avg. recall@" << recall_at << " is " << recall_val << "\n";
}
