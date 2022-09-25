/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "lambda/common/vector_distance.h"
#include "parameters.h"

namespace lambda {
    struct query_stats {
        double total_us = 0;  // total time to process query in micros
        double io_us = 0;     // total time spent in IO
        double cpu_us = 0;    // total time spent in CPU

        unsigned n_4k = 0;          // # of 4kB reads
        unsigned n_8k = 0;          // # of 8kB reads
        unsigned n_12k = 0;         // # of 12kB reads
        unsigned n_ios = 0;         // total # of IOs issued
        unsigned read_size = 0;     // total # of bytes read
        unsigned n_cmps_saved = 0;  // # cmps saved
        unsigned n_cmps = 0;        // # cmps
        unsigned n_cache_hits = 0;  // # cache_hits
        unsigned n_hops = 0;        // # search hops
    };

    template<typename T>
    inline T get_percentile_stats(
            query_stats *stats, uint64_t len, float percentile,
            const std::function<T(const query_stats &)> &member_fn) {
        std::vector<T> vals(len);
        for (uint64_t i = 0; i < len; i++) {
            vals[i] = member_fn(stats[i]);
        }

        std::sort(vals.begin(), vals.end(),
                  [](const T &left, const T &right) { return left < right; });

        auto retval = vals[(uint64_t) (percentile * len)];
        vals.clear();
        return retval;
    }

    template<typename T>
    inline double get_mean_stats(
            query_stats *stats, uint64_t len,
            const std::function<T(const query_stats &)> &member_fn) {
        double avg = 0;
        for (uint64_t i = 0; i < len; i++) {
            avg += (double) member_fn(stats[i]);
        }
        return avg / len;
    }
}  // namespace lambda
