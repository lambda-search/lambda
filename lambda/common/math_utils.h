/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <vector>



// taken from
// https://github.com/Microsoft/BLAS-on-flash/blob/master/include/utils.h
// round up X to the nearest multiple of Y
#define ROUND_UP(X, Y) \
  ((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))

#define DIV_ROUND_UP(X, Y) (((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0))

// round down X to the nearest multiple of Y
#define ROUND_DOWN(X, Y) (((uint64_t)(X) / (Y)) * (Y))

// alignment tests
#define IS_ALIGNED(X, Y) ((uint64_t)(X) % (uint64_t)(Y) == 0)
#define IS_512_ALIGNED(X) IS_ALIGNED(X, 512)
#define IS_4096_ALIGNED(X) IS_ALIGNED(X, 4096)

namespace lambda::math_utils {

    float calc_distance(float *vec_1, float *vec_2, size_t dim);

    // compute l2-squared norms of data stored in row major num_points * dim,
    // needs
    // to be pre-allocated
    void compute_vecs_l2sq(float *vecs_l2sq, float *data, const size_t num_points,
                           const size_t dim);

    void rotate_data_randomly(float *data, size_t num_points, size_t dim,
                              float *rot_mat, float *&new_mat,
                              bool transpose_rot = false);

    // calculate closest center to data of num_points * dim (row major)
    // centers is num_centers * dim (row major)
    // data_l2sq has pre-computed squared norms of data
    // centers_l2sq has pre-computed squared norms of centers
    // pre-allocated center_index will contain id of k nearest centers
    // pre-allocated dist_matrix shound be num_points * num_centers and contain
    // squared distances

    // Ideally used only by compute_closest_centers
    void compute_closest_centers_in_block(
            const float *const data, const size_t num_points, const size_t dim,
            const float *const centers, const size_t num_centers,
            const float *const docs_l2sq, const float *const centers_l2sq,
            uint32_t *center_index, float *const dist_matrix, size_t k = 1);

    // Given data in num_points * new_dim row major
    // Pivots stored in full_pivot_data as k * new_dim row major
    // Calculate the closest pivot for each point and store it in vector
    // closest_centers_ivf (which needs to be allocated outside)
    // Additionally, if inverted index is not null (and pre-allocated), it will
    // return inverted index for each center Additionally, if pts_norms_squared is
    // not null, then it will assume that point norms are pre-computed and use
    // those
    // values

    void compute_closest_centers(float *data, size_t num_points, size_t dim,
                                 float *pivot_data, size_t num_centers, size_t k,
                                 uint32_t *closest_centers_ivf,
                                 std::vector<size_t> *inverted_index = NULL,
                                 float *pts_norms_squared = NULL);

    // if to_subtract is 1, will subtract nearest center from each row. Else will
    // add. Output will be in data_load iself.
    // Nearest centers need to be provided in closst_centers.

    void process_residuals(float *data_load, size_t num_points, size_t dim,
                           float *cur_pivot_data, size_t num_centers,
                           uint32_t *closest_centers, bool to_subtract);

}  // namespace lambda::math_utils

namespace lambda::kmeans {

    // run Lloyds one iteration
    // Given data in row major num_points * dim, and centers in row major
    // num_centers * dim
    // And squared lengths of data points, output the closest center to each data
    // point, update centers, and also return inverted index.
    // If closest_centers == NULL, will allocate memory and return.
    // Similarly, if closest_docs == NULL, will allocate memory and return.

    float lloyds_iter(float *data, size_t num_points, size_t dim, float *centers,
                      size_t num_centers, float *docs_l2sq,
                      std::vector<size_t> *closest_docs,
                      uint32_t *&closest_center);

    // Run Lloyds until max_reps or stopping criterion
    // If you pass NULL for closest_docs and closest_center, it will NOT return
    // the results, else it will assume appriate allocation as closest_docs = new
    // vector<size_t> [num_centers], and closest_center = new size_t[num_points]
    // Final centers are output in centers as row major num_centers * dim
    //
    float run_lloyds(float *data, size_t num_points, size_t dim, float *centers,
                     const size_t num_centers, const size_t max_reps,
                     std::vector<size_t> *closest_docs, uint32_t *closest_center);

    // assumes already memory allocated for pivot_data as new
    // float[num_centers*dim] and select randomly num_centers points as pivots
    void selecting_pivots(float *data, size_t num_points, size_t dim,
                          float *pivot_data, size_t num_centers);

    void kmeanspp_selecting_pivots(float *data, size_t num_points, size_t dim,
                                   float *pivot_data, size_t num_centers);
}  // namespace lambda::kmeans
