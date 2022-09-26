/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include <atomic>
#include <cstring>
#include <iomanip>
#include <omp.h>
#include <lambda/graph/pq_flash_index.h>
#include <set>
#include <string.h>
#include <time.h>
#include "lambda/graph/aux_utils.h"
#include "lambda/graph/index.h"
#include "lambda/common/math_utils.h"
#include "lambda/graph/binary_file.h"
#include "lambda/graph/partition_and_pq.h"
#include "lambda/graph/utils.h"
#include "lambda/graph/percentile_stats.h"
#include <boost/program_options.hpp>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "lambda/graph/linux_aligned_file_reader.h"

#define WARMUP false

namespace po = boost::program_options;

void print_stats(std::string category, std::vector<float> percentiles,
                 std::vector<float> results) {
    std::cout << std::setw(20) << category << ": " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++) {
        std::cout << std::setw(8) << percentiles[s] << "%";
    }
    std::cout << std::endl;
    std::cout << std::setw(22) << " " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++) {
        std::cout << std::setw(9) << results[s];
    }
    std::cout << std::endl;
}

template<typename T>
int search_disk_index(
        lambda::Metric &metric, const std::string &index_path_prefix,
        const std::string &result_output_prefix, const std::string &query_file,
        std::string &gt_file, const unsigned num_threads, const unsigned recall_at,
        const unsigned beamwidth, const unsigned num_nodes_to_cache,
        const uint32_t search_io_limit, const std::vector<unsigned> &Lvec,
        const bool use_reorder_data = false) {
    std::cout << "Search parameters: #threads: " << num_threads << ", ";
    if (beamwidth <= 0)
        std::cout << "beamwidth to be optimized for each L value" << std::flush;
    else
        std::cout << " beamwidth: " << beamwidth << std::flush;
    if (search_io_limit == std::numeric_limits<uint32_t>::max())
        std::cout << "." << std::endl;
    else
        std::cout << ", io_limit: " << search_io_limit << "." << std::endl;

    std::string warmup_query_file = index_path_prefix + "_sample_data.bin";

    // load query bin
    T *query = nullptr;
    unsigned *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    lambda::binary_file::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                                query_aligned_dim);

    bool calc_recall_flag = false;
    if (gt_file != std::string("null") && gt_file != std::string("NULL") &&
        melon::exists(gt_file)) {
        lambda::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num) {
            std::cout
                    << "Error. Mismatch in number of queries and ground truth data"
                    << std::endl;
        }
        calc_recall_flag = true;
    }

    std::shared_ptr<lambda::AlignedFileReader> reader = nullptr;
    reader.reset(new lambda::LinuxAlignedFileReader());

    std::unique_ptr<lambda::pq_flash_index<T>> _pFlashIndex(
            new lambda::pq_flash_index<T>(reader, metric));

    auto res = _pFlashIndex->load(num_threads, index_path_prefix.c_str());

    if (!res.is_ok()) {
        return -1;
    }
    // cache bfs levels
    std::vector<uint32_t> node_list;
    std::cout << "Caching " << num_nodes_to_cache
                 << " BFS nodes around medoid(s)" << std::endl;
    //_pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
    if (num_nodes_to_cache > 0)
        _pFlashIndex->generate_cache_list_from_sample_queries(
                warmup_query_file, 15, 6, num_nodes_to_cache, num_threads, node_list);
    _pFlashIndex->load_cache_list(node_list);
    node_list.clear();
    node_list.shrink_to_fit();

    omp_set_num_threads(num_threads);

    uint64_t warmup_L = 20;
    size_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
    T *warmup = nullptr;

    if (WARMUP) {
        if (melon::exists(warmup_query_file)) {
            lambda::binary_file::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num,
                                        warmup_dim, warmup_aligned_dim);
        } else {
            warmup_num = (std::min)((uint32_t) 150000, (uint32_t) 15000 * num_threads);
            warmup_dim = query_dim;
            warmup_aligned_dim = query_aligned_dim;
            lambda::alloc_aligned(((void **) &warmup),
                                  warmup_num * warmup_aligned_dim * sizeof(T),
                                  8 * sizeof(T));
            std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-128, 127);
            for (uint32_t i = 0; i < warmup_num; i++) {
                for (uint32_t d = 0; d < warmup_dim; d++) {
                    warmup[i * warmup_aligned_dim + d] = (T) dis(gen);
                }
            }
        }
        std::cout << "Warming up index... " << std::flush;
        std::vector<uint64_t> warmup_result_ids_64(warmup_num, 0);
        std::vector<float> warmup_result_dists(warmup_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t) warmup_num; i++) {
            _pFlashIndex->cached_beam_search(warmup + (i * warmup_aligned_dim), 1,
                                             warmup_L,
                                             warmup_result_ids_64.data() + (i * 1),
                                             warmup_result_dists.data() + (i * 1), 4);
        }
        std::cout << "..done" << std::endl;
    }

    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout.precision(2);

    std::string recall_string = "Recall@" + std::to_string(recall_at);
    std::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth"
                 << std::setw(16) << "QPS" << std::setw(16) << "Mean Latency"
                 << std::setw(16) << "99.9 Latency" << std::setw(16)
                 << "Mean IOs" << std::setw(16) << "CPU (s)";
    if (calc_recall_flag) {
        std::cout << std::setw(16) << recall_string << std::endl;
    } else
        std::cout << std::endl;
    std::cout
            << "==============================================================="
               "======================================================="
            << std::endl;

    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());

    uint32_t optimized_beamwidth = 2;

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
        uint64_t L = Lvec[test_id];

        if (L < recall_at) {
            std::cout << "Ignoring search with L:" << L
                         << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        if (beamwidth <= 0) {
            std::cout << "Tuning beamwidth.." << std::endl;
            optimized_beamwidth =
                    optimize_beamwidth(_pFlashIndex, warmup, warmup_num,
                                       warmup_aligned_dim, L, optimized_beamwidth);
        } else
            optimized_beamwidth = beamwidth;

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);

        auto stats = new lambda::query_stats[query_num];

        std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
        auto s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t) query_num; i++) {
            _pFlashIndex->cached_beam_search(
                    query + (i * query_aligned_dim), recall_at, L,
                    query_result_ids_64.data() + (i * recall_at),
                    query_result_dists[test_id].data() + (i * recall_at),
                    optimized_beamwidth, search_io_limit, use_reorder_data, stats + i);
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        float qps = (1.0 * query_num) / (1.0 * diff.count());

        lambda::convert_types<uint64_t, uint32_t>(query_result_ids_64.data(),
                                                  query_result_ids[test_id].data(),
                                                  query_num, recall_at);

        auto mean_latency = lambda::get_mean_stats<float>(
                stats, query_num,
                [](const lambda::query_stats &stats) { return stats.total_us; });

        auto latency_999 = lambda::get_percentile_stats<float>(
                stats, query_num, 0.999,
                [](const lambda::query_stats &stats) { return stats.total_us; });

        auto mean_ios = lambda::get_mean_stats<unsigned>(
                stats, query_num,
                [](const lambda::query_stats &stats) { return stats.n_ios; });

        auto mean_cpuus = lambda::get_mean_stats<float>(
                stats, query_num,
                [](const lambda::query_stats &stats) { return stats.cpu_us; });

        float recall = 0;
        if (calc_recall_flag) {
            recall = lambda::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
                                              query_result_ids[test_id].data(),
                                              recall_at, recall_at);
        }

        std::cout << std::setw(6) << L << std::setw(12) << optimized_beamwidth
                     << std::setw(16) << qps << std::setw(16) << mean_latency
                     << std::setw(16) << latency_999 << std::setw(16) << mean_ios
                     << std::setw(16) << mean_cpuus;
        if (calc_recall_flag) {
            std::cout << std::setw(16) << recall << std::endl;
        } else
            std::cout << std::endl;
        delete[] stats;
    }

    std::cout << "Done searching. Now saving results " << std::endl;
    uint64_t test_id = 0;
    for (auto L : Lvec) {
        if (L < recall_at)
            continue;

        std::string cur_result_path =
                result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
        lambda::binary_file::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(),
                               query_num, recall_at);

        cur_result_path =
                result_output_prefix + "_" + std::to_string(L) + "_dists_float.bin";
        lambda::binary_file::save_bin<float>(cur_result_path,
                                query_result_dists[test_id++].data(), query_num,
                                recall_at);
    }

    lambda::aligned_free(query);
    if (warmup != nullptr)
        lambda::aligned_free(warmup);
    return 0;
}

int main(int argc, char **argv) {
    std::string data_type, dist_fn, index_path_prefix, result_path_prefix,
            query_file, gt_file;
    unsigned num_threads, K, W, num_nodes_to_cache, search_io_limit;
    std::vector<unsigned> Lvec;
    bool use_reorder_data = false;

    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type",
                           po::value<std::string>(&data_type)->required(),
                           "data type <int8/uint8/float>");
        desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                           "distance function <l2/mips/fast_l2>");
        desc.add_options()("index_path_prefix",
                           po::value<std::string>(&index_path_prefix)->required(),
                           "Path prefix to the index");
        desc.add_options()("result_path",
                           po::value<std::string>(&result_path_prefix)->required(),
                           "Path prefix for saving results of the queries");
        desc.add_options()("query_file",
                           po::value<std::string>(&query_file)->required(),
                           "Query file in binary format");
        desc.add_options()(
                "gt_file",
                po::value<std::string>(&gt_file)->default_value(std::string("null")),
                "ground truth file for the queryset");
        desc.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                           "Number of neighbors to be returned");
        desc.add_options()("search_list,L",
                           po::value<std::vector<unsigned>>(&Lvec)->multitoken(),
                           "List of L values of search");
        desc.add_options()("beamwidth,W", po::value<uint32_t>(&W)->default_value(2),
                           "Beamwidth for search. Set 0 to optimize internally.");
        desc.add_options()(
                "num_nodes_to_cache",
                po::value<uint32_t>(&num_nodes_to_cache)->default_value(0),
                "Beamwidth for search");
        desc.add_options()("search_io_limit",
                           po::value<uint32_t>(&search_io_limit)
                                   ->default_value(std::numeric_limits<uint32_t>::max()),
                           "Max #IOs for search");
        desc.add_options()(
                "num_threads,T",
                po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                "Number of threads used for building index (defaults to "
                "omp_get_num_procs())");
        desc.add_options()("use_reorder_data",
                           po::bool_switch()->default_value(false),
                           "Include full precision data in the index. Use only in "
                           "conjuction with compressed data on SSD.");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
        if (vm["use_reorder_data"].as<bool>())
            use_reorder_data = true;
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    lambda::Metric metric;
    if (dist_fn == std::string("mips")) {
        metric = lambda::Metric::INNER_PRODUCT;
    } else if (dist_fn == std::string("l2")) {
        metric = lambda::Metric::L2;
    } else if (dist_fn == std::string("cosine")) {
        metric = lambda::Metric::COSINE;
    } else {
        std::cout << "Unsupported distance function. Currently only L2/ Inner "
                     "Product/Cosine are supported."
                  << std::endl;
        return -1;
    }

    if ((data_type != std::string("float")) &&
        (metric == lambda::Metric::INNER_PRODUCT)) {
        std::cout << "Currently support only floating point data for Inner Product."
                  << std::endl;
        return -1;
    }

    if (use_reorder_data && data_type != std::string("float")) {
        std::cout << "Error: Reorder data for reordering currently only "
                     "supported for float data type."
                  << std::endl;
        return -1;
    }

    try {
        if (data_type == std::string("float"))
            return search_disk_index<float>(metric, index_path_prefix,
                                            result_path_prefix, query_file, gt_file,
                                            num_threads, K, W, num_nodes_to_cache,
                                            search_io_limit, Lvec, use_reorder_data);
        else if (data_type == std::string("int8"))
            return search_disk_index<int8_t>(metric, index_path_prefix,
                                             result_path_prefix, query_file, gt_file,
                                             num_threads, K, W, num_nodes_to_cache,
                                             search_io_limit, Lvec, use_reorder_data);
        else if (data_type == std::string("uint8"))
            return search_disk_index<uint8_t>(
                    metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                    num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                    use_reorder_data);
        else {
            std::cerr << "Unsupported data type. Use float or int8 or uint8"
                      << std::endl;
            return -1;
        }
    } catch (const std::exception &e) {
        std::cout << std::string(e.what()) << std::endl;
        std::cerr << "Index search failed." << std::endl;
        return -1;
    }
}
