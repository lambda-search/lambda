/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include <type_traits>
#include <omp.h>
#include <atomic>
#include <random>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <melon/container/robin_set.h>
#include <melon/files/filesystem.h>
#include <melon/container/robin_map.h>
#include <melon/times/time.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include "melon/log/logging.h"
#include "aligned_file_reader.h"
#include "lambda/common/math_utils.h"
#include "parameters.h"
#include "partition_and_pq.h"
#include "utils.h"
#include <melon/base/profile.h>
#include "lambda/graph/binary_file.h"
#include "lambda/common/memory.h"

#if defined(RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && \
    defined(DISKANN_BUILD)
#include "gperftools/malloc_extension.h"
#endif

#include "melon/container/dynamic_bitset.h"
#include "index.h"

#define MAX_POINTS_FOR_USING_BITSET 10000000

namespace lambda {
    template<typename T>
    inline T diskann_max(T left, T right) {
        return left > right ? left : right;
    }

    // QueryScratch functions
    template<typename T>
    in_mem_query_scratch<T>::in_mem_query_scratch() {
        search_l = indexing_l = r = 0;
        // pointers are initialized in the header itself.
    }

    template<typename T>
    melon::result_status in_mem_query_scratch<T>::setup(uint32_t search_l, uint32_t indexing_l,
                                                     uint32_t r, size_t dim) {
        if (search_l == 0 || indexing_l == 0 || r == 0 || dim == 0) {
            std::stringstream ss;
            ss << "In in_mem_query_scratch, one of search_l = " << search_l
               << ", indexing_l = " << indexing_l << ", dim = " << dim
               << " or r = " << r << " is zero.\n";
            return melon::result_status(-1, ss.str());
        }
        indices = new uint32_t[search_l];     // only used by search
        interim_dists = new float[search_l];  // only used by search
        memset(indices, 0, sizeof(uint32_t) * search_l);
        memset(interim_dists, 0, sizeof(float) * search_l);
        this->search_l = search_l;
        this->indexing_l = indexing_l;
        this->r = r;

        auto aligned_dim = ROUND_UP(dim, 8);
        size_t allocSize = aligned_dim * sizeof(T);
        alloc_aligned(((void **) &aligned_query), allocSize, 8 * sizeof(T));
        memset(aligned_query, 0, aligned_dim * sizeof(T));

        auto l_to_use = std::max(search_l, indexing_l);

        _des = new std::vector<unsigned>();
        _des->reserve(2 * r);
        _pool = new std::vector<neighbor>();
        _pool->reserve(l_to_use * 10);
        _visited = new melon::robin_set<unsigned>();
        _visited->reserve(l_to_use * 2);
        _best_l_nodes = new std::vector<neighbor>();
        _best_l_nodes->resize(l_to_use + 1);
        _inserted_into_pool_rs = new melon::robin_set<unsigned>();
        _inserted_into_pool_rs->reserve(l_to_use * 20);
        _inserted_into_pool_bs = new melon::dynamic_bitset<>();
        return melon::result_status::success();
    }

    template<typename T>
    void in_mem_query_scratch<T>::clear() {
        memset(indices, 0, sizeof(uint32_t) * search_l);
        memset(interim_dists, 0, sizeof(float) * search_l);
        _pool->clear();
        _visited->clear();
        _des->clear();
        _inserted_into_pool_rs->clear();
        _inserted_into_pool_bs->reset();
    }

    template<typename T>
    void in_mem_query_scratch<T>::resize_for_query(uint32_t new_search_l) {
        if (search_l < new_search_l) {
            if (indices != nullptr) {
                delete[] indices;
            }
            indices = new uint32_t[new_search_l];

            if (interim_dists != nullptr) {
                delete[] interim_dists;
            }
            interim_dists = new float[new_search_l];
            search_l = new_search_l;
        }
    }

    template<typename T>
    void in_mem_query_scratch<T>::destroy() {
        if (indices != nullptr) {
            delete[] indices;
            indices = nullptr;
        }
        if (interim_dists != nullptr) {
            delete[] interim_dists;
            interim_dists = nullptr;
        }
        if (_pool != nullptr) {
            delete _pool;
            _pool = nullptr;
        }
        if (_visited != nullptr) {
            delete _visited;
            _visited = nullptr;
        }
        if (_des != nullptr) {
            delete _des;
            _des = nullptr;
        }
        if (_best_l_nodes != nullptr) {
            delete _best_l_nodes;
            _best_l_nodes = nullptr;
        }
        if (aligned_query != nullptr) {
            aligned_free(aligned_query);
            aligned_query = nullptr;
        }

        if (_inserted_into_pool_rs != nullptr) {
            delete _inserted_into_pool_rs;
            _inserted_into_pool_rs = nullptr;
        }
        if (_inserted_into_pool_bs != nullptr) {
            delete _inserted_into_pool_bs;
            _inserted_into_pool_bs = nullptr;
        }

        search_l = indexing_l = r = 0;
    }

    // Class to avoid the hassle of pushing and popping the query scratch.
    template<typename T>
    class ScratchStoreManager {
    public:
        lambda::in_mem_query_scratch<T> _scratch;
        ConcurrentQueue<in_mem_query_scratch<T>> &_query_scratch;

        ScratchStoreManager(ConcurrentQueue<in_mem_query_scratch<T>> &query_scratch)
                : _query_scratch(query_scratch) {
            _scratch = query_scratch.pop();
            while (_scratch.indices == nullptr) {
                query_scratch.wait_for_push_notify();
                _scratch = query_scratch.pop();
            }
        }

        in_mem_query_scratch<T> scratch_space() {
            return _scratch;
        }

        ~ScratchStoreManager() {
            _scratch.clear();
            _query_scratch.push(_scratch);
            _query_scratch.push_notify_all();
        }

    private:
        ScratchStoreManager(const ScratchStoreManager<T> &);

        ScratchStoreManager<T> &operator=(const ScratchStoreManager<T> &);
    };

    // Initialize an index with metric m, load the data of type T with filename
    // (bin), and initialize max_points
    template<typename T, typename TagT>
    Index<T, TagT>::Index(Metric m, const size_t dim, const size_t max_points,
                          const bool dynamic_index, const Parameters &indexParams,
                          const Parameters &searchParams, const bool enable_tags,
                          const bool support_eager_delete,
                          const bool concurrent_consolidate)
            : Index(m, dim, max_points, dynamic_index, enable_tags,
                    support_eager_delete, concurrent_consolidate) {
        _indexingQueueSize = indexParams.Get<uint32_t>("L");
        _indexingRange = indexParams.Get<uint32_t>("R");
        _indexingMaxC = indexParams.Get<uint32_t>("C");
        _indexingAlpha = indexParams.Get<float>("alpha");

        uint32_t num_threads_srch = searchParams.Get<uint32_t>("num_threads");
        uint32_t num_threads_indx = indexParams.Get<uint32_t>("num_threads");
        uint32_t num_scratch_spaces = num_threads_srch + num_threads_indx;
        uint32_t search_l = searchParams.Get<uint32_t>("L");

        _status = initialize_query_scratch(num_scratch_spaces, search_l, _indexingQueueSize,
                                 _indexingRange, dim);
    }

    template<typename T, typename TagT>
    Index<T, TagT>::Index(Metric m, const size_t dim, const size_t max_points,
                          const bool dynamic_index, const bool enable_tags,
                          const bool support_eager_delete,
                          const bool concurrent_consolidate)
            : _dist_metric(m), _dim(dim), _max_points(max_points),
              _dynamic_index(dynamic_index), _enable_tags(enable_tags),
              _support_eager_delete(support_eager_delete),
              _conc_consolidate(concurrent_consolidate) {
        if (dynamic_index && !enable_tags) {
            MELON_CHECK(false) << "ERROR: Dynamic Indexing must have tags enabled.";
        }
        if (support_eager_delete && !dynamic_index) {
            MELON_CHECK(false) << "ERROR: Eager deletes must have dynamic indexing enabled.";
        }

        // data stored to _nd * aligned_dim matrix with necessary zero-padding
        _aligned_dim = ROUND_UP(_dim, 8);

        if (dynamic_index) {
            _num_frozen_pts = 1;
        }
        // Sanity check. While logically it is correct, max_points = 0 causes
        // downstream problems.
        if (_max_points == 0) {
            _max_points = 1;
        }
        const size_t total_internal_points = _max_points + _num_frozen_pts;
        alloc_aligned(((void **) &_data),
                      total_internal_points * _aligned_dim * sizeof(T),
                      8 * sizeof(T));
        std::memset(_data, 0, total_internal_points * _aligned_dim * sizeof(T));

        _start = (unsigned) _max_points;

        _final_graph.resize(total_internal_points);

        if (_support_eager_delete) {
            _in_graph.reserve(total_internal_points);
            _in_graph.resize(total_internal_points);
        }

        if (m == lambda::Metric::COSINE && std::is_floating_point<T>::value) {
            // This is safe because T is float inside the if block.
            this->_distance = (vector_distance<T> *) new AVXNormalizedCosineDistanceFloat();
            this->_normalize_vecs = true;
            MELON_LOG(INFO) << "Normalizing vectors and using L2 for cosine "
                               "AVXNormalizedCosineDistanceFloat().";
        } else {
            this->_distance = get_distance_function<T>(m);
        }

        _locks = std::vector<non_recursive_mutex>(total_internal_points);

        if (_support_eager_delete)
            _locks_in = std::vector<non_recursive_mutex>(total_internal_points);

        if (enable_tags) {
            _location_to_tag.reserve(total_internal_points);
            _tag_to_location.reserve(total_internal_points);
        }
    }

    template<typename T, typename TagT>
    Index<T, TagT>::~Index() {
        // Ensure that no other activity is happening before dtor()
        std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
        std::unique_lock<std::shared_timed_mutex> cl(_consolidate_lock);
        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
        std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);

        for (auto &lock : _locks) {
            LockGuard lg(lock);
        }
        for (auto &lock : _locks_in) {
            LockGuard lg(lock);
        }

        if (this->_distance != nullptr) {
            delete this->_distance;
            this->_distance = nullptr;
        }
        if (this->_data != nullptr) {
            aligned_free(this->_data);
            this->_data = nullptr;
        }

        while (!_query_scratch.empty()) {
            auto val = _query_scratch.pop();
            while (val.indices == nullptr) {
                _query_scratch.wait_for_push_notify();
                val = _query_scratch.pop();
            }
            val.destroy();
        }
    }

    template<typename T, typename TagT>
    melon::result_status Index<T, TagT>::initialize_query_scratch(uint32_t num_threads,
                                                  uint32_t search_l,
                                                  uint32_t indexing_l, uint32_t r,
                                                  size_t dim) {
        for (uint32_t i = 0; i < num_threads; i++) {
            in_mem_query_scratch<T> scratch;
            auto rs = scratch.setup(search_l, indexing_l, r, dim);
            if(!rs.is_ok()) {
                return rs;
            }
            _query_scratch.push(scratch);
        }
        return melon::result_status::success();
    }

    template<typename T, typename TagT>
    melon::result_status Index<T, TagT>::save_tags(std::string tags_file, size_t *tag_bytes_written) {
        if (!_enable_tags) {
            MELON_LOG(INFO) << "Not saving tags as they are not enabled.";
            return melon::result_status::success();
        }
        TagT *tag_data = new TagT[_nd + _num_frozen_pts];
        for (uint32_t i = 0; i < _nd; i++) {
            TagT tag;
            if (_location_to_tag.try_get(i, tag)) {
                tag_data[i] = tag;
            } else {
                // catering to future when tagT can be any type.
                std::memset((char *) &tag_data[i], 0, sizeof(TagT));
            }
        }
        if (_num_frozen_pts > 0) {
            std::memset((char *) &tag_data[_start], 0, sizeof(TagT));
        }
        auto rs = lambda::binary_file::save_bin<TagT>(tags_file, tag_data, _nd + _num_frozen_pts, 1, 0, tag_bytes_written);
        delete[] tag_data;
        return rs;
    }

    template<typename T, typename TagT>
    uint64_t Index<T, TagT>::save_data(std::string data_file) {
        return save_data_in_base_dimensions(data_file, _data, _nd + _num_frozen_pts,
                                            _dim, _aligned_dim);
    }

    // save the graph index on a file as an adjacency list. For each point,
    // first store the number of neighbors, and then the neighbor list (each as
    // 4 byte unsigned)
    template<typename T, typename TagT>
    uint64_t Index<T, TagT>::save_graph(std::string graph_file) {
        melon::sequential_write_file out;
        out.open(graph_file, false);
        uint64_t file_offset = 0;  // we will use this if we want
        out.reset(file_offset);
        uint64_t index_size = 24;
        uint32_t max_degree = 0;
        out.write((char *) &index_size, sizeof(uint64_t));
        out.write((char *) &_max_observed_degree, sizeof(unsigned));
        unsigned ep_u32 = _start;
        out.write((char *) &ep_u32, sizeof(unsigned));
        out.write((char *) &_num_frozen_pts, sizeof(uint64_t));
        for (unsigned i = 0; i < _nd + _num_frozen_pts; i++) {
            unsigned GK = (unsigned) _final_graph[i].size();
            out.write((char *) &GK, sizeof(unsigned));
            out.write((char *) _final_graph[i].data(), GK * sizeof(unsigned));
            max_degree = _final_graph[i].size() > max_degree
                         ? (uint32_t) _final_graph[i].size()
                         : max_degree;
            index_size += (uint64_t) (sizeof(unsigned) * (GK + 1));
        }
        out.reset(file_offset);
        out.write((char *) &index_size, sizeof(uint64_t));
        out.write((char *) &max_degree, sizeof(uint32_t));
        out.close();
        return index_size;  // number of bytes written
    }

    template<typename T, typename TagT>
    melon::result_status Index<T, TagT>::save_delete_list(const std::string &filename, size_t *saved_size) {
        if (_delete_set.size() == 0) {
            return melon::result_status::success();
        }
        std::unique_ptr<uint32_t[]> delete_list =
                std::make_unique<uint32_t[]>(_delete_set.size());
        uint32_t i = 0;
        for (auto &del : _delete_set) {
            delete_list[i++] = del;
        }
        return lambda::binary_file::save_bin<uint32_t>(filename, delete_list.get(), _delete_set.size(), 1, 0, saved_size);
    }

    template<typename T, typename TagT>
    melon::result_status Index<T, TagT>::save(const char *filename, bool compact_before_save) {
        melon::stop_watcher timer;

        if (compact_before_save) {
            compact_data();
            compact_frozen_point();
        } else {
            if (not _data_compacted) {
                return melon::result_status(-1,
                                            "Index save for non-compacted index is not yet implemented");
            }
        }

        std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
        std::unique_lock<std::shared_timed_mutex> cl(_consolidate_lock);
        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
        std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);

        std::error_code ec;
        size_t tmp_size;
        if (!_save_as_one_file) {
            std::string graph_file = std::string(filename);
            std::string tags_file = std::string(filename) + ".tags";
            std::string data_file = std::string(filename) + ".data";
            std::string delete_list_file = std::string(filename) + ".del";

            // Because the save_* functions use append mode, ensure that
            // the files are deleted before save. Ideally, we should check
            // the error code for melon::remove, but will ignore now because
            // delete should succeed if save will succeed.
            melon::remove(graph_file, ec);
            save_graph(graph_file);
            melon::remove(data_file, ec);
            save_data(data_file);
            melon::remove(tags_file, ec);
            save_tags(tags_file, &tmp_size);
            melon::remove(delete_list_file, ec);
            save_delete_list(delete_list_file, &tmp_size);
        } else {
            MELON_LOG(INFO) << "Save index in a single file currently not supported. "
                               "Not saving the index.";
        }

        reposition_frozen_point_to_end();

        MELON_LOG(INFO) << "Time taken for save: " << timer.elapsed() / 1000000.0
                        << "s.";
        return melon::result_status::success();
    }


    template<typename T, typename TagT>
    melon::result_status Index<T, TagT>::load_tags(const std::string tag_filename, size_t &ret) {
        if (_enable_tags && !file_exists(tag_filename)) {
            MELON_LOG(ERROR) << "Tag file provided does not exist!";
            return melon::result_status(-1, "Tag file provided does not exist!");
        }
        if (!_enable_tags) {
            MELON_LOG(INFO) << "Tags not loaded as tags not enabled.";
            ret =  0;
            return melon::result_status::success();
        }

        size_t file_dim, file_num_points;
        TagT *tag_data;
        auto rs = lambda::binary_file::load_bin<TagT>(std::string(tag_filename), tag_data, file_num_points,
                                            file_dim);
        if(!rs.is_ok()) {
            return rs;
        }
        if (file_dim != 1) {
            std::stringstream stream;
            stream << "ERROR: Found " << file_dim << " dimensions for tags,"
                   << "but tag file must have 1 dimension.\n";
            MELON_LOG(ERROR) << stream.str();
            delete[] tag_data;
            return melon::result_status(-1, stream.str());
        }

        size_t num_data_points =
                _num_frozen_pts > 0 ? file_num_points - 1 : file_num_points;
        _location_to_tag.reserve(num_data_points);
        _tag_to_location.reserve(num_data_points);
        for (uint32_t i = 0; i < (uint32_t) num_data_points; i++) {
            TagT tag = *(tag_data + i);
            if (_delete_set.find(i) == _delete_set.end()) {
                _location_to_tag.set(i, tag);
                _tag_to_location[tag] = i;
            }
        }
        MELON_LOG(INFO) << "Tags loaded.";
        delete[] tag_data;
        ret =  file_num_points;
        return melon::result_status::success();
    }

    template<typename T, typename TagT>
    size_t Index<T, TagT>::load_data(std::string filename) {
        size_t file_dim, file_num_points;
        if (!file_exists(filename)) {
            std::stringstream stream;
            stream << "ERROR: data file " << filename << " does not exist.\n";
            MELON_LOG(ERROR) << stream.str();
            aligned_free(_data);
            MELON_CHECK(false) << stream.str();
        }
        lambda::get_bin_metadata(filename, file_num_points, file_dim);

        // since we are loading a new dataset, _empty_slots must be cleared
        _empty_slots.clear();

        if (file_dim != _dim) {
            std::stringstream stream;
            stream << "ERROR: Driver requests loading " << _dim << " dimension,"
                   << "but file has " << file_dim << " dimension.\n";
            MELON_LOG(ERROR) << stream.str();
            aligned_free(_data);
            MELON_CHECK(false) << stream.str();
        }

        if (file_num_points > _max_points) {
            // update and tag lock acquired in load() before calling load_data
            resize(file_num_points);
        }
        copy_aligned_data_from_file<T>(filename.c_str(), _data, file_num_points,
                                       file_dim, _aligned_dim);
        return file_num_points;
    }


    template<typename T, typename TagT>
    melon::result_status Index<T, TagT>::load_delete_set(const std::string &filename, size_t &ret) {
        std::unique_ptr<uint32_t[]> delete_list;
        size_t npts, ndim;
        auto rs = lambda::binary_file::load_bin<uint32_t>(filename, delete_list, npts, ndim);
        if(!rs.is_ok()) {
            return rs;
        }
        assert(ndim == 1);
        for (uint32_t i = 0; i < npts; i++) {
            _delete_set.insert(delete_list[i]);
        }
        ret =  npts;
        return melon::result_status::success();
    }

    // load the index from file and update the max_degree, cur (navigating
    // node loc), and _final_graph (adjacency list)
    template<typename T, typename TagT>
    melon::result_status Index<T, TagT>::load(const char *filename, uint32_t num_threads,
                                              uint32_t search_l) {
        std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);

        _has_built = true;

        size_t tags_file_num_pts = 0, graph_num_pts = 0, data_file_num_pts = 0;
        if (!_save_as_one_file) {
            // For DLVS Store, we will not support saving the index in multiple files.
            std::string data_file = std::string(filename) + ".data";
            std::string tags_file = std::string(filename) + ".tags";
            std::string delete_set_file = std::string(filename) + ".del";
            std::string graph_file = std::string(filename);
            data_file_num_pts = load_data(data_file);
            if (file_exists(delete_set_file)) {
                size_t tmp;
                auto rs = load_delete_set(delete_set_file, tmp);
                if(!rs.is_ok()) {
                    return rs;
                }
            }
            if (_enable_tags) {
                auto rs = load_tags(tags_file, tags_file_num_pts);
                if(!rs.is_ok()) {
                    return rs;
                }
            }
            graph_num_pts = load_graph(graph_file, data_file_num_pts);

        } else {
            MELON_LOG(INFO) << "Single index file saving/loading support not yet "
                               "enabled. Not loading the index.";
            return melon::result_status::success();
        }

        if (data_file_num_pts != graph_num_pts ||
            (data_file_num_pts != tags_file_num_pts && _enable_tags)) {
            std::stringstream stream;
            stream << "ERROR: When loading index, loaded " << data_file_num_pts
                   << " points from datafile, " << graph_num_pts
                   << " from graph, and " << tags_file_num_pts
                   << " tags, with num_frozen_pts being set to " << _num_frozen_pts
                   << " in constructor.\n";
            MELON_LOG(ERROR) << stream.str();
            aligned_free(_data);
            return melon::result_status(-1, stream.str());
        }

        _nd = data_file_num_pts - _num_frozen_pts;
        _empty_slots.clear();
        _empty_slots.reserve(_max_points);
        for (auto i = _nd; i < _max_points; i++) {
            _empty_slots.insert((uint32_t) i);
        }

        _lazy_done = _delete_set.size() != 0;

        reposition_frozen_point_to_end();
        MELON_LOG(INFO) << "Num frozen points:" << _num_frozen_pts << " _nd: " << _nd
                        << " _start: " << _start
                        << " size(_location_to_tag): " << _location_to_tag.size()
                        << " size(_tag_to_location):" << _tag_to_location.size()
                        << " Max points: " << _max_points;

        _search_queue_size = search_l;
        // For incremental index, _query_scratch is initialized in the constructor.
        // For the bulk index, the params required to initialize _query_scratch
        // are known only at load time, hence this check and the call to
        // initialize_q_s().
        if (_query_scratch.size() == 0) {
            _status = initialize_query_scratch(num_threads, search_l, search_l,
                                     (uint32_t) _max_range_of_loaded_graph, _dim);
        }
        return melon::result_status::success();
    }


    template<typename T, typename TagT>
    size_t Index<T, TagT>::load_graph(std::string filename,
                                      size_t expected_num_points) {
        size_t expected_file_size;
        uint64_t file_frozen_pts;

        uint64_t file_offset = 0;  // will need this for single file format support
        std::ifstream in;
        in.exceptions(std::ios::badbit | std::ios::failbit);
        in.open(filename, std::ios::binary);
        in.seekg(file_offset, in.beg);
        in.read((char *) &expected_file_size, sizeof(uint64_t));
        in.read((char *) &_max_observed_degree, sizeof(unsigned));
        in.read((char *) &_start, sizeof(unsigned));
        in.read((char *) &file_frozen_pts, sizeof(uint64_t));
        uint64_t vamana_metadata_size =
                sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint64_t);

        MELON_LOG(INFO) << "From graph header, expected_file_size: "
                        << expected_file_size
                        << ", _max_observed_degree: " << _max_observed_degree
                        << ", _start: " << _start
                        << ", file_frozen_pts: " << file_frozen_pts;

        if (file_frozen_pts != _num_frozen_pts) {
            std::stringstream stream;
            if (file_frozen_pts == 1) {
                stream << "ERROR: When loading index, detected dynamic index, but "
                          "constructor asks for static index. Exitting.\n";
            } else {
                stream << "ERROR: When loading index, detected static index, but "
                          "constructor asks for dynamic index. Exitting.\n";
            }
            MELON_LOG(ERROR) << stream.str();
            aligned_free(_data);
            MELON_CHECK(false) << stream.str();
        }

        MELON_LOG(INFO) << "Loading vamana graph " << filename << "...";

        // If user provides more points than max_points
        // resize the _final_graph to the larger size.
        if (_max_points < expected_num_points) {
            MELON_LOG(INFO) << "Number of points in data: " << expected_num_points
                            << " is greater than max_points: " << _max_points
                            << " Setting max points to: " << expected_num_points;
            _final_graph.resize(expected_num_points + _num_frozen_pts);
            _max_points = expected_num_points;
        }

        size_t bytes_read = vamana_metadata_size;
        size_t cc = 0;
        unsigned nodes_read = 0;
        while (bytes_read != expected_file_size) {
            unsigned k;
            in.read((char *) &k, sizeof(unsigned));
            if (k == 0) {
                MELON_LOG(ERROR) << "ERROR: Point found with no out-neighbors, point#"
                                 << nodes_read;
            }

            cc += k;
            ++nodes_read;
            std::vector<unsigned> tmp(k);
            tmp.reserve(k);
            in.read((char *) tmp.data(), k * sizeof(unsigned));
            _final_graph[nodes_read - 1].swap(tmp);
            bytes_read += sizeof(uint32_t) * ((uint64_t) k + 1);
            if (nodes_read % 10000000 == 0)
                MELON_LOG(INFO) << "." << std::flush;
            if (k > _max_range_of_loaded_graph) {
                _max_range_of_loaded_graph = k;
            }
        }

        MELON_LOG(INFO) << "done. Index has " << nodes_read << " nodes and " << cc
                        << " out-edges, _start is set to " << _start;
        return nodes_read;
    }

    template<typename T, typename TagT>
    int Index<T, TagT>::get_vector_by_tag(TagT &tag, T *vec) {
        std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
        if (_tag_to_location.find(tag) == _tag_to_location.end()) {
            MELON_LOG(INFO) << "Tag " << tag << " does not exist";
            return -1;
        }

        size_t location = _tag_to_location[tag];
        memcpy((void *) vec, (void *) (_data + location * _aligned_dim),
               (size_t) _dim * sizeof(T));
        return 0;
    }

    /**************************************************************
     *      Support for Static Index Building and Searching
     **************************************************************/

    /* This function finds out the navigating node, which is the medoid node
     * in the graph.
     */
    template<typename T, typename TagT>
    unsigned Index<T, TagT>::calculate_entry_point() {
        // allocate and init centroid
        float *center = new float[_aligned_dim]();
        for (size_t j = 0; j < _aligned_dim; j++)
            center[j] = 0;

        for (size_t i = 0; i < _nd; i++)
            for (size_t j = 0; j < _aligned_dim; j++)
                center[j] += (float) _data[i * _aligned_dim + j];

        for (size_t j = 0; j < _aligned_dim; j++)
            center[j] /= (float) _nd;

        // compute all to one distance
        float *distances = new float[_nd]();
#pragma omp parallel for schedule(static, 65536)
        for (int64_t i = 0; i < (int64_t) _nd; i++) {
            // extract point and distance reference
            float &dist = distances[i];
            const T *cur_vec = _data + (i * (size_t) _aligned_dim);
            dist = 0;
            float diff = 0;
            for (size_t j = 0; j < _aligned_dim; j++) {
                diff =
                        (center[j] - (float) cur_vec[j]) * (center[j] - (float) cur_vec[j]);
                dist += diff;
            }
        }
        // find imin
        unsigned min_idx = 0;
        float min_dist = distances[0];
        for (unsigned i = 1; i < _nd; i++) {
            if (distances[i] < min_dist) {
                min_idx = i;
                min_dist = distances[i];
            }
        }

        delete[] distances;
        delete[] center;
        return min_idx;
    }

    template<typename T, typename TagT>
    std::pair<uint32_t, uint32_t> Index<T, TagT>::iterate_to_fixed_point(
            const T *node_coords, const unsigned Lsize,
            const std::vector<unsigned> &init_ids,
            std::vector<neighbor> &expanded_nodes_info,
            melon::robin_set<unsigned> &expanded_nodes_ids,
            std::vector<neighbor> &best_L_nodes, std::vector<unsigned> &des,
            melon::robin_set<unsigned> &inserted_into_pool_rs,
            melon::dynamic_bitset<> &inserted_into_pool_bs, bool ret_frozen,
            bool search_invocation) {
        for (unsigned i = 0; i < Lsize + 1; i++) {
            best_L_nodes[i].distance = std::numeric_limits<float>::max();
        }
        if (!search_invocation) {
            expanded_nodes_ids.clear();
            expanded_nodes_info.clear();
            des.clear();
        }

        unsigned l = 0;
        neighbor nn;

        bool fast_iterate =
                (_max_points + _num_frozen_pts) <= MAX_POINTS_FOR_USING_BITSET;

        if (fast_iterate) {
            auto total_num_points = _max_points + _num_frozen_pts;
            if (inserted_into_pool_bs.size() < total_num_points) {
                // hopefully using 2X will reduce the number of allocations.
                auto resize_size = 2 * total_num_points > MAX_POINTS_FOR_USING_BITSET
                                   ? MAX_POINTS_FOR_USING_BITSET
                                   : 2 * total_num_points;
                inserted_into_pool_bs.resize(resize_size);
            }
        }

        for (auto id : init_ids) {
            if (id >= _max_points + _num_frozen_pts) {
                MELON_LOG(ERROR) << "Out of range loc found as an edge : " << id;
                MELON_CHECK(false) <<
                                   std::string("Wrong loc") + std::to_string(id);
            }
            nn = neighbor(id,
                          _distance->compare(_data + _aligned_dim * (size_t) id,
                                             node_coords, (unsigned) _aligned_dim),
                          true);
            if (fast_iterate) {
                if (inserted_into_pool_bs[id] == 0) {
                    inserted_into_pool_bs[id] = 1;
                    best_L_nodes[l++] = nn;
                }
            } else {
                if (inserted_into_pool_rs.find(id) == inserted_into_pool_rs.end()) {
                    inserted_into_pool_rs.insert(id);
                    best_L_nodes[l++] = nn;
                }
            }
            if (l == Lsize)
                break;
        }

        // sort best_L_nodes based on distance of each point to node_coords
        std::sort(best_L_nodes.begin(), best_L_nodes.begin() + l);
        unsigned k = 0;
        uint32_t hops = 0;
        uint32_t cmps = 0;

        while (k < l) {
            unsigned nk = l;

            if (best_L_nodes[k].flag) {
                best_L_nodes[k].flag = false;
                auto n = best_L_nodes[k].id;
                if (!(best_L_nodes[k].id == _start && _num_frozen_pts > 0 &&
                      !ret_frozen)) {
                    if (!search_invocation) {
                        expanded_nodes_info.emplace_back(best_L_nodes[k]);
                        expanded_nodes_ids.insert(n);
                    }
                }
                des.clear();
                if (_dynamic_index) {
                    LockGuard guard(_locks[n]);
                    for (unsigned m = 0; m < _final_graph[n].size(); m++) {
                        if (_final_graph[n][m] >= _max_points + _num_frozen_pts) {
                            std::stringstream msg;
                            msg << "Out of range edge " << _final_graph[n][m]
                                << " found at vertex " << n;
                            MELON_CHECK(false) << msg.str();
                        }
                        des.emplace_back(_final_graph[n][m]);
                    }
                } else {
                    for (unsigned m = 0; m < _final_graph[n].size(); m++) {
                        if (_final_graph[n][m] >= _max_points + _num_frozen_pts) {
                            std::stringstream msg;
                            msg << "Out of range edge " << _final_graph[n][m]
                                << " found at vertex " << n;
                            MELON_CHECK(false) << msg.str();
                        }
                        des.emplace_back(_final_graph[n][m]);
                    }
                }

                for (unsigned m = 0; m < des.size(); ++m) {
                    unsigned id = des[m];
                    bool id_is_missing = fast_iterate ? inserted_into_pool_bs[id] == 0
                                                      : inserted_into_pool_rs.find(id) ==
                                                        inserted_into_pool_rs.end();
                    if (id_is_missing) {
                        if (fast_iterate) {
                            inserted_into_pool_bs[id] = 1;
                        } else {
                            inserted_into_pool_rs.insert(id);
                        }
                        if ((m + 1) < des.size()) {
                            auto nextn = des[m + 1];
                            lambda::prefetch_vector(
                                    (const char *) _data + _aligned_dim * (size_t) nextn,
                                    sizeof(T) * _aligned_dim);
                        }

                        cmps++;
                        float dist = _distance->compare(node_coords,
                                                        _data + _aligned_dim * (size_t) id,
                                                        (unsigned) _aligned_dim);

                        if (dist >= best_L_nodes[l - 1].distance && (l == Lsize))
                            continue;

                        neighbor nn(id, dist, true);
                        unsigned r = insert_into_pool(best_L_nodes.data(), l, nn);
                        if (l < Lsize)
                            ++l;
                        if (r < nk)
                            nk = r;
                    }
                }

                if (nk <= k)
                    k = nk;
                else
                    ++k;
            } else
                k++;
        }
        return std::make_pair(hops, cmps);
    }

    template<typename T, typename TagT>
    void Index<T, TagT>::get_expanded_nodes(
            const size_t node_id, const unsigned Lindex,
            std::vector<unsigned> init_ids,
            std::vector<neighbor> &expanded_nodes_info,
            melon::robin_set<unsigned> &expanded_nodes_ids, std::vector<unsigned> &des,
            std::vector<neighbor> &best_L_nodes,
            melon::robin_set<unsigned> &inserted_into_pool_rs,
            melon::dynamic_bitset<> &inserted_into_pool_bs) {
        const T *node_coords = _data + _aligned_dim * node_id;

        if (init_ids.size() == 0)
            init_ids.emplace_back(_start);

        iterate_to_fixed_point(node_coords, Lindex, init_ids, expanded_nodes_info,
                               expanded_nodes_ids, best_L_nodes, des,
                               inserted_into_pool_rs, inserted_into_pool_bs);
    }

    template<typename T, typename TagT>
    void Index<T, TagT>::get_expanded_nodes(
            const size_t node_id, const unsigned Lindex,
            std::vector<unsigned> init_ids,
            std::vector<neighbor> &expanded_nodes_info,
            melon::robin_set<unsigned> &expanded_nodes_ids) {
        const T *node_coords = _data + _aligned_dim * node_id;

        if (init_ids.size() == 0)
            init_ids.emplace_back(_start);

        std::vector<unsigned> des;
        std::vector<neighbor> best_L_nodes;
        best_L_nodes.resize(Lindex + 1);
        melon::robin_set<unsigned> inserted_into_pool_rs;
        melon::dynamic_bitset<> inserted_into_pool_bs;

        iterate_to_fixed_point(node_coords, Lindex, init_ids, expanded_nodes_info,
                               expanded_nodes_ids, best_L_nodes, des,
                               inserted_into_pool_rs, inserted_into_pool_bs);
    }

    template<typename T, typename TagT>
    void Index<T, TagT>::search_for_point_and_add_links(
            int location, uint32_t Lindex, std::vector<neighbor> &pool,
            melon::robin_set<unsigned> &visited, std::vector<unsigned> &des,
            std::vector<neighbor> &best_l_nodes,
            melon::robin_set<unsigned> &inserted_into_pool_rs,
            melon::dynamic_bitset<> &inserted_into_pool_bs) {
        std::vector<unsigned> init_ids;
        get_expanded_nodes(location, Lindex, init_ids, pool, visited, des,
                           best_l_nodes, inserted_into_pool_rs,
                           inserted_into_pool_bs);

        for (unsigned i = 0; i < pool.size(); i++) {
            if (pool[i].id == (unsigned) location) {
                pool.erase(pool.begin() + i);
                visited.erase((unsigned) location);
                i--;
            } else if (_delete_set.find(pool[i].id) != _delete_set.end()) {
                pool.erase(pool.begin() + i);
                visited.erase((unsigned) pool[i].id);
                i--;
            }
        }

        std::vector<unsigned> pruned_list;
        prune_neighbors(location, pool, pruned_list);

        assert(!pruned_list.empty());
        assert(_final_graph.size() == _max_points + _num_frozen_pts);

        if (_support_eager_delete) {
            for (unsigned i = 0; i < _final_graph[location].size(); i++) {
                {
                    LockGuard guard(_locks_in[_final_graph[location][i]]);
                    _in_graph[_final_graph[location][i]].erase(
                            std::remove(_in_graph[_final_graph[location][i]].begin(),
                                        _in_graph[_final_graph[location][i]].end(), location),
                            _in_graph[_final_graph[location][i]].end());
                }
            }
        }

        {
            std::shared_lock<std::shared_timed_mutex> tlock(_tag_lock,
                                                            std::defer_lock);
            if (_conc_consolidate)
                tlock.lock();

            LockGuard guard(_locks[location]);
            _final_graph[location].clear();
            _final_graph[location].shrink_to_fit();
            _final_graph[location].reserve(
                    (uint64_t) (_indexingRange * GRAPH_SLACK_FACTOR * 1.05));

            for (auto link : pruned_list) {
                if (_conc_consolidate)
                    if (!_location_to_tag.contains(link))
                        continue;
                _final_graph[location].emplace_back(link);
                if (_support_eager_delete) {
                    LockGuard guard(_locks_in[link]);
                    _in_graph[link].emplace_back(location);
                }
            }

            if (_conc_consolidate)
                tlock.unlock();
        }

        assert(_final_graph[location].size() <= _indexingRange);
        inter_insert(location, pruned_list, _support_eager_delete);
    }

    template<typename T, typename TagT>
    void Index<T, TagT>::occlude_list(std::vector<neighbor> &pool,
                                      const float alpha, const unsigned degree,
                                      const unsigned maxc,
                                      std::vector<neighbor> &result) {
        if (pool.size() == 0)
            return;

        assert(std::is_sorted(pool.begin(), pool.end()));
        if (pool.size() > maxc)
            pool.resize(maxc);
        std::vector<float> occlude_factor(pool.size(), 0);

        float cur_alpha = 1;
        while (cur_alpha <= alpha && result.size() < degree) {
            // used for MIPS, where we store a value of eps in cur_alpha to
            // denote pruned out entries which we can skip in later rounds.
            float eps = cur_alpha + 0.01f;

            for (auto iter = pool.begin();
                 result.size() < degree && iter != pool.end(); ++iter) {
                if (occlude_factor[iter - pool.begin()] > cur_alpha) {
                    continue;
                }
                occlude_factor[iter - pool.begin()] = std::numeric_limits<float>::max();
                result.push_back(*iter);
                for (auto iter2 = iter + 1; iter2 != pool.end(); iter2++) {
                    auto t = iter2 - pool.begin();
                    if (occlude_factor[t] > alpha)
                        continue;
                    float djk =
                            _distance->compare(_data + _aligned_dim * (size_t) iter2->id,
                                               _data + _aligned_dim * (size_t) iter->id,
                                               (unsigned) _aligned_dim);
                    if (_dist_metric == lambda::Metric::L2 ||
                        _dist_metric == lambda::Metric::COSINE) {
                        if (djk == 0.0)
                            occlude_factor[t] = std::numeric_limits<float>::max();
                        else
                            occlude_factor[t] =
                                    std::max(occlude_factor[t], iter2->distance / djk);
                    } else if (_dist_metric == lambda::Metric::INNER_PRODUCT) {
                        // Improvization for flipping max and min dist for MIPS
                        float x = -iter2->distance;
                        float y = -djk;
                        if (y > cur_alpha * x) {
                            occlude_factor[t] = std::max(occlude_factor[t], eps);
                        }
                    }
                }
            }
            cur_alpha *= 1.2;
        }
    }

    template<typename T, typename TagT>
    void Index<T, TagT>::prune_neighbors(const unsigned location,
                                         std::vector<neighbor> &pool,
                                         std::vector<unsigned> &pruned_list) {
        prune_neighbors(location, pool, _indexingRange, _indexingMaxC,
                        _indexingAlpha, pruned_list);
    }

    template<typename T, typename TagT>
    void Index<T, TagT>::prune_neighbors(const unsigned location,
                                         std::vector<neighbor> &pool,
                                         const uint32_t range,
                                         const uint32_t max_candidate_size,
                                         const float alpha,
                                         std::vector<unsigned> &pruned_list) {
        if (pool.size() == 0) {
            std::stringstream ss;
            ss << "Thread loc:" << std::this_thread::get_id()
               << " Pool address: " << &pool;
            MELON_LOG(INFO) << ss.str();
            MELON_CHECK(false) << "Pool passed to prune_neighbors is empty";
        }

        _max_observed_degree = (std::max)(_max_observed_degree, range);

        // sort the pool based on distance to query
        std::sort(pool.begin(), pool.end());

        std::vector<neighbor> result;
        result.reserve(range);

        occlude_list(pool, alpha, range, max_candidate_size, result);

        pruned_list.clear();
        assert(result.size() <= range);
        for (auto iter : result) {
            if (iter.id != location)
                pruned_list.emplace_back(iter.id);
        }

        if (_saturate_graph && alpha > 1) {
            for (uint32_t i = 0; i < pool.size() && pruned_list.size() < range; i++) {
                if ((std::find(pruned_list.begin(), pruned_list.end(), pool[i].id) ==
                     pruned_list.end()) &&
                    pool[i].id != location)
                    pruned_list.emplace_back(pool[i].id);
            }
        }
    }

    template<typename T, typename TagT>
    void Index<T, TagT>::batch_inter_insert(
            unsigned n, const std::vector<unsigned> &pruned_list, const uint32_t range,
            std::vector<unsigned> &need_to_sync) {
        // assert(!src_pool.empty());

        for (auto des : pruned_list) {
            if (des == n)
                continue;
            // des.loc is the loc of the neighbors of n
            assert(des >= 0 && des < _max_points + _num_frozen_pts);
            if (des > _max_points)
                MELON_LOG(INFO) << "error. " << des << " exceeds max_pts";
            // des_pool contains the neighbors of the neighbors of n

            {
                LockGuard guard(_locks[des]);
                if (std::find(_final_graph[des].begin(), _final_graph[des].end(), n) ==
                    _final_graph[des].end()) {
                    _final_graph[des].push_back(n);
                    if (_final_graph[des].size() >
                        (unsigned) (range * GRAPH_SLACK_FACTOR))
                        need_to_sync[des] = 1;
                }
            }  // des lock is released by this point
        }
    }

    template<typename T, typename TagT>
    void Index<T, TagT>::batch_inter_insert(
            unsigned n, const std::vector<unsigned> &pruned_list,
            std::vector<unsigned> &need_to_sync) {
        batch_inter_insert(n, pruned_list, _indexingRange, need_to_sync);
    }

    template<typename T, typename TagT>
    void Index<T, TagT>::inter_insert(unsigned n,
                                      std::vector<unsigned> &pruned_list,
                                      const uint32_t range, bool update_in_graph) {
        const auto &src_pool = pruned_list;

        assert(!src_pool.empty());

        for (auto des : src_pool) {
            // des.loc is the loc of the neighbors of n
            assert(des >= 0 && des < _max_points + _num_frozen_pts);
            // des_pool contains the neighbors of the neighbors of n
            std::vector<unsigned> copy_of_neighbors;
            bool prune_needed = false;
            {
                LockGuard guard(_locks[des]);
                auto &des_pool = _final_graph[des];
                if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end()) {
                    if (des_pool.size() < (uint64_t) (GRAPH_SLACK_FACTOR * range)) {
                        des_pool.emplace_back(n);
                        if (update_in_graph) {
                            LockGuard guard(_locks_in[n]);
                            _in_graph[n].emplace_back(des);
                        }
                        prune_needed = false;
                    } else {
                        copy_of_neighbors = des_pool;
                        prune_needed = true;
                    }
                }
            }  // des lock is released by this point

            if (prune_needed) {
                copy_of_neighbors.push_back(n);
                melon::robin_set<unsigned> dummy_visited(0);
                std::vector<neighbor> dummy_pool(0);

                size_t reserveSize =
                        (size_t) (std::ceil(1.05 * GRAPH_SLACK_FACTOR * range));
                dummy_visited.reserve(reserveSize);
                dummy_pool.reserve(reserveSize);

                for (auto cur_nbr : copy_of_neighbors) {
                    if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
                        cur_nbr != des) {
                        float dist =
                                _distance->compare(_data + _aligned_dim * (size_t) des,
                                                   _data + _aligned_dim * (size_t) cur_nbr,
                                                   (unsigned) _aligned_dim);
                        dummy_pool.emplace_back(neighbor(cur_nbr, dist, true));
                        dummy_visited.insert(cur_nbr);
                    }
                }
                std::vector<unsigned> new_out_neighbors;
                prune_neighbors(des, dummy_pool, new_out_neighbors);
                {
                    LockGuard guard(_locks[des]);
                    // updating in_graph of out-neighbors of des
                    if (update_in_graph) {
                        for (auto out_nbr : _final_graph[des]) {
                            {
                                LockGuard guard(_locks_in[out_nbr]);
                                for (unsigned i = 0; i < _in_graph[out_nbr].size(); i++) {
                                    if (_in_graph[out_nbr][i] == des) {
                                        _in_graph[out_nbr].erase(_in_graph[out_nbr].begin() + i);
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    _final_graph[des].clear();
                    for (auto new_nbr : new_out_neighbors) {
                        _final_graph[des].emplace_back(new_nbr);
                        if (update_in_graph) {
                            LockGuard guard(_locks_in[new_nbr]);
                            _in_graph[new_nbr].emplace_back(des);
                        }
                    }
                }
            }
        }
    }

    template<typename T, typename TagT>
    void Index<T, TagT>::inter_insert(unsigned n,
                                      std::vector<unsigned> &pruned_list,
                                      bool update_in_graph) {
        inter_insert(n, pruned_list, _indexingRange, update_in_graph);
    }

    template<typename T, typename TagT>
    void Index<T, TagT>::link(Parameters &parameters) {
        unsigned num_threads = parameters.Get<unsigned>("num_threads");
        if (num_threads != 0)
            omp_set_num_threads(num_threads);

        _saturate_graph = parameters.Get<bool>("saturate_graph");

        if (num_threads != 0)
            omp_set_num_threads(num_threads);

        _indexingQueueSize = parameters.Get<unsigned>("L");  // Search list size
        _indexingRange = parameters.Get<unsigned>("R");
        _indexingMaxC = parameters.Get<unsigned>("C");
        _indexingAlpha = parameters.Get<float>("alpha");

        /* visit_order is a vector that is initialized to the entire graph */
        std::vector<unsigned> visit_order;
        std::vector<lambda::neighbor> pool, tmp;
        melon::robin_set<unsigned> visited;
        visit_order.reserve(_nd + _num_frozen_pts);
        for (unsigned i = 0; i < (unsigned) _nd; i++) {
            visit_order.emplace_back(i);
        }

        if (_num_frozen_pts > 0)
            visit_order.emplace_back((unsigned) _max_points);

        // if there are frozen points, the first such one is set to be the _start
        if (_num_frozen_pts > 0)
            _start = (unsigned) _max_points;
        else
            _start = calculate_entry_point();

        if (_support_eager_delete) {
            _in_graph.reserve(_max_points + _num_frozen_pts);
            _in_graph.resize(_max_points + _num_frozen_pts);
        }

        for (uint64_t p = 0; p < _nd; p++) {
            _final_graph[p].reserve(
                    (size_t) (std::ceil(_indexingRange * GRAPH_SLACK_FACTOR * 1.05)));
        }

        std::vector<unsigned> init_ids;
        init_ids.emplace_back(_start);

        melon::stop_watcher link_timer;

#pragma omp parallel for schedule(dynamic, 2048)
        for (int64_t node_ctr = 0; node_ctr < (int64_t) (visit_order.size()); node_ctr++) {
            auto node = visit_order[node_ctr];
            std::vector<neighbor> pool;
            melon::robin_set<unsigned> visited;
            pool.reserve(_indexingQueueSize * 2);
            visited.reserve(_indexingQueueSize * 2);
            std::vector<unsigned> des;
            des.reserve(_indexingRange * GRAPH_SLACK_FACTOR);
            std::vector<neighbor> best_L_nodes;
            best_L_nodes.resize(_indexingQueueSize + 1);
            melon::robin_set<unsigned> inserted_into_pool_rs;
            melon::dynamic_bitset<> inserted_into_pool_bs;

            search_for_point_and_add_links(node, _indexingQueueSize, pool, visited,
                                           des, best_L_nodes, inserted_into_pool_rs,
                                           inserted_into_pool_bs);

            if (node_ctr % 100000 == 0) {
                MELON_LOG(INFO) << "\r" << (100.0 * node_ctr) / (visit_order.size())
                                << "\% of index build completed." << std::flush;
            }
        }

        if (_nd > 0) {
            MELON_LOG(INFO) << "Starting final cleanup.." << std::flush;
        }
#pragma omp parallel for schedule(dynamic, 2048)
        for (int64_t node_ctr = 0; node_ctr < (int64_t) (visit_order.size()); node_ctr++) {
            auto node = visit_order[node_ctr];
            if (_final_graph[node].size() > _indexingRange) {
                melon::robin_set<unsigned> dummy_visited(0);
                std::vector<neighbor> dummy_pool(0);
                std::vector<unsigned> new_out_neighbors;

                for (auto cur_nbr : _final_graph[node]) {
                    if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
                        cur_nbr != node) {
                        float dist =
                                _distance->compare(_data + _aligned_dim * (size_t) node,
                                                   _data + _aligned_dim * (size_t) cur_nbr,
                                                   (unsigned) _aligned_dim);
                        dummy_pool.emplace_back(neighbor(cur_nbr, dist, true));
                        dummy_visited.insert(cur_nbr);
                    }
                }
                prune_neighbors(node, dummy_pool, new_out_neighbors);

                _final_graph[node].clear();
                for (auto id : new_out_neighbors)
                    _final_graph[node].emplace_back(id);
            }
        }
        if (_nd > 0) {
            MELON_LOG(INFO) << "done. Link time: " << (double) link_timer.s_elapsed()<<"s";
        }
    }

    template<typename T, typename TagT>
    void Index<T, TagT>::prune_all_nbrs(const Parameters &parameters) {
        const unsigned range = parameters.Get<unsigned>("R");

        melon::stop_watcher timer;
#pragma omp parallel for
        for (int64_t node = 0; node < (int64_t) (_max_points + _num_frozen_pts); node++) {
            if ((size_t) node < _nd || (size_t) node == _max_points) {
                if (_final_graph[node].size() > range) {
                    melon::robin_set<unsigned> dummy_visited(0);
                    std::vector<neighbor> dummy_pool(0);
                    std::vector<unsigned> new_out_neighbors;

                    for (auto cur_nbr : _final_graph[node]) {
                        if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
                            cur_nbr != node) {
                            float dist =
                                    _distance->compare(_data + _aligned_dim * (size_t) node,
                                                       _data + _aligned_dim * (size_t) cur_nbr,
                                                       (unsigned) _aligned_dim);
                            dummy_pool.emplace_back(neighbor(cur_nbr, dist, true));
                            dummy_visited.insert(cur_nbr);
                        }
                    }
                    prune_neighbors((uint32_t) node, dummy_pool, new_out_neighbors);

                    _final_graph[node].clear();
                    for (auto id : new_out_neighbors)
                        _final_graph[node].emplace_back(id);
                }
            }
        }

        MELON_LOG(INFO) << "Prune time : " << timer.elapsed() / 1000 << "ms";
        size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
        for (size_t i = 0; i < (_nd + _num_frozen_pts); i++) {
            auto &pool = _final_graph[i];
            max = (std::max)(max, pool.size());
            min = (std::min)(min, pool.size());
            total += pool.size();
            if (pool.size() < 2)
                cnt++;
        }
        if (min > max)
            min = max;
        if (_nd > 0) {
            MELON_LOG(INFO) << "Index built with degree: max:" << max << "  avg:"
                            << (float) total / (float) (_nd + _num_frozen_pts)
                            << "  min:" << min << "  count(deg<2):" << cnt;
        }
    }

    template<typename T, typename TagT>
    melon::result_status Index<T, TagT>::set_start_point(T *data) {
        std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
        if (_nd > 0)
            return melon::result_status(-1, "Can not set starting point for a non-empty index");

        memcpy(_data + _aligned_dim * _max_points, data, _aligned_dim * sizeof(T));
        _has_built = true;
        MELON_LOG(INFO) << "Index start point set";
        return melon::result_status::success();
    }

    template<typename T, typename TagT>
    void Index<T, TagT>::set_start_point_at_random(T radius) {
        std::vector<double> real_vec;
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::normal_distribution<> d{0.0, 1.0};
        double norm_sq = 0.0;
        for (size_t i = 0; i < _aligned_dim; ++i) {
            auto r = d(gen);
            real_vec.push_back(r);
            norm_sq += r * r;
        }

        double norm = std::sqrt(norm_sq);
        std::vector<T> start_vec;
        for (auto iter : real_vec)
            start_vec.push_back(static_cast<T>(iter * radius / norm));

        set_start_point(start_vec.data());
    }

    template<typename T, typename TagT>
    melon::result_status Index<T, TagT>::build_with_data_populated(
            Parameters &parameters, const std::vector<TagT> &tags) {
        MELON_LOG(INFO) << "Starting index build with " << _nd << " points... ";

        if (_nd < 1)
            return melon::result_status(-1, "Error: Trying to build an index with 0 points");

        if (_enable_tags && tags.size() != _nd) {
            std::stringstream stream;
            stream << "ERROR: Driver requests loading " << _nd << " points from file,"
                   << "but tags vector is of size " << tags.size() << ".";
            MELON_LOG(ERROR) << stream.str();
            aligned_free(_data);
            return melon::result_status(-1, stream.str());
        }
        if (_enable_tags) {
            for (size_t i = 0; i < tags.size(); ++i) {
                _tag_to_location[tags[i]] = (unsigned) i;
                _location_to_tag.set(static_cast<unsigned>(i), tags[i]);
            }
        }

        generate_frozen_point();
        link(parameters);

        if (_support_eager_delete) {
            update_in_graph();  // copying values to in_graph
        }

        size_t max = 0, min = SIZE_MAX, total = 0, cnt = 0;
        for (size_t i = 0; i < _nd; i++) {
            auto &pool = _final_graph[i];
            max = (std::max)(max, pool.size());
            min = (std::min)(min, pool.size());
            total += pool.size();
            if (pool.size() < 2)
                cnt++;
        }
        MELON_LOG(INFO) << "Index built with degree: max:" << max
                        << "  avg:" << (float) total / (float) (_nd + _num_frozen_pts)
                        << "  min:" << min << "  count(deg<2):" << cnt;

        _max_observed_degree = (std::max)((unsigned) max, _max_observed_degree);
        _has_built = true;
        return melon::result_status::success();
    }

    template<typename T, typename TagT>
    melon::result_status Index<T, TagT>::build(const T *data, const size_t num_points_to_load,
                                               Parameters &parameters,
                                               const std::vector<TagT> &tags) {
        if (num_points_to_load == 0)
            return melon::result_status(-1, "Do not call build with 0 points");

        _nd = num_points_to_load;

        memcpy((char *) _data, (char *) data, _aligned_dim * _nd * sizeof(T));

        if (_normalize_vecs) {
            for (uint64_t i = 0; i < num_points_to_load; i++) {
                normalize(_data + _aligned_dim * i, _aligned_dim);
            }
        }

        return build_with_data_populated(parameters, tags);
    }

    template<typename T, typename TagT>
    melon::result_status Index<T, TagT>::build(const char *filename,
                                               const size_t num_points_to_load,
                                               Parameters &parameters,
                                               const std::vector<TagT> &tags) {
        if (num_points_to_load == 0)
            return melon::result_status(-1, "Do not call build with 0 points");

        if (!file_exists(filename)) {
            MELON_LOG(ERROR) << "Data file " << filename
                             << " does not exist!!! Exiting....";
            std::stringstream stream;
            stream << "Data file " << filename << " does not exist.";
            MELON_LOG(ERROR) << stream.str();
            return melon::result_status(-1, stream.str());
        }

        size_t file_num_points, file_dim;
        if (filename == nullptr) {
            return melon::result_status(-1, "Can not build with an empty file");
        }

        lambda::get_bin_metadata(filename, file_num_points, file_dim);
        if (file_num_points > _max_points) {
            std::stringstream stream;
            stream << "ERROR: Driver requests loading " << num_points_to_load
                   << " points and file has " << file_num_points << " points, but "
                   << "index can support only " << _max_points
                   << " points as specified in constructor.";
            aligned_free(_data);
            return melon::result_status(-1, stream.str());
        }

        if (num_points_to_load > file_num_points) {
            std::stringstream stream;
            stream << "ERROR: Driver requests loading " << num_points_to_load
                   << " points and file has only " << file_num_points << " points.";
            aligned_free(_data);
            return melon::result_status(-1, stream.str());
        }

        if (file_dim != _dim) {
            std::stringstream stream;
            stream << "ERROR: Driver requests loading " << _dim << " dimension,"
                   << "but file has " << file_dim << " dimension.";
            MELON_LOG(ERROR) << stream.str();
            aligned_free(_data);
            return melon::result_status(-1, stream.str());
        }

        copy_aligned_data_from_file<T>(filename, _data, file_num_points, file_dim,
                                       _aligned_dim);
        if (_normalize_vecs) {
            for (uint64_t i = 0; i < file_num_points; i++) {
                normalize(_data + _aligned_dim * i, _aligned_dim);
            }
        }

        MELON_LOG(INFO) << "Using only first " << num_points_to_load
                        << " from file.. ";
        _nd = num_points_to_load;
        return build_with_data_populated(parameters, tags);
    }

    template<typename T, typename TagT>
    melon::result_status Index<T, TagT>::build(const char *filename,
                                               const size_t num_points_to_load,
                                               Parameters &parameters, const char *tag_filename) {
        std::vector<TagT> tags;
        melon::result_status rs;
        if (_enable_tags) {
            if (tag_filename == nullptr) {
                return melon::result_status(-1, "Tag filename is null, while _enable_tags is set");
            } else {
                if (file_exists(tag_filename)) {
                    MELON_LOG(INFO) << "Loading tags from " << tag_filename
                                    << " for vamana index build";
                    TagT *tag_data = nullptr;
                    size_t npts, ndim;
                    rs = lambda::binary_file::load_bin(tag_filename, tag_data, npts, ndim);
                    if(!rs.is_ok()) {
                        return rs;
                    }
                    if (npts < num_points_to_load) {
                        std::stringstream sstream;
                        sstream << "Loaded " << npts
                                << " tags, insufficient to populate tags for "
                                << num_points_to_load << "  points to load";
                        return melon::result_status(-1, sstream.str());
                    }
                    for (size_t i = 0; i < num_points_to_load; i++) {
                        tags.push_back(tag_data[i]);
                    }
                    delete[] tag_data;
                } else {
                    return melon::result_status(-1, std::string("Tag file") + tag_filename + " does not exist");
                }
            }
        }
        return build(filename, num_points_to_load, parameters, tags);
    }

    template<typename T, typename TagT>
    template<typename IdType>
    std::pair<uint32_t, uint32_t> Index<T, TagT>::search(const T *query,
                                                         const size_t K,
                                                         const unsigned L,
                                                         IdType *indices,
                                                         float *distances) {
        ScratchStoreManager<T> manager(_query_scratch);
        auto scratch = manager.scratch_space();

        return search_impl(query, K, L, indices, distances, scratch);
    }

    template<typename T, typename TagT>
    template<typename IdType>
    std::pair<uint32_t, uint32_t> Index<T, TagT>::search_impl(
            const T *query, const size_t K, const unsigned L, IdType *indices,
            float *distances, in_mem_query_scratch<T> &scratch) {
        std::vector<neighbor> &expanded_nodes_info = scratch.pool();
        melon::robin_set<unsigned> &expanded_nodes_ids = scratch.visited();
        std::vector<unsigned> &des = scratch.des();
        std::vector<neighbor> best_L_nodes = scratch.best_l_nodes();
        melon::robin_set<unsigned> &inserted_into_pool_rs =
                scratch.inserted_into_pool_rs();
        melon::dynamic_bitset<> &inserted_into_pool_bs =
                scratch.inserted_into_pool_bs();

        std::vector<unsigned> init_ids;

        std::shared_lock<std::shared_timed_mutex> lock(_update_lock);

        if (init_ids.size() == 0) {
            init_ids.emplace_back(_start);
        }
        T *aligned_query = scratch.aligned_query;
        memcpy(aligned_query, query, _dim * sizeof(T));

        if (_normalize_vecs) {
            normalize((float *) aligned_query, _dim);
        }

        auto retval = iterate_to_fixed_point(
                aligned_query, L, init_ids, expanded_nodes_info, expanded_nodes_ids,
                best_L_nodes, des, inserted_into_pool_rs, inserted_into_pool_bs, true,
                true);

        size_t pos = 0;
        for (auto it : best_L_nodes) {
            if (it.id < _max_points) {
                indices[pos] =
                        (IdType) it.id;  // safe because our indices are always uint32_t and
                // IDType will be uint32_t or uint64_t
                if (distances != nullptr) {
                    distances[pos] = _dist_metric == lambda::Metric::INNER_PRODUCT
                                     ? -1 * it.distance
                                     : it.distance;
                }
                pos++;
            }
            if (pos == K)
                break;
        }
        return retval;
    }

    template<typename T, typename TagT>
    size_t Index<T, TagT>::search_with_tags(const T *query, const uint64_t K,
                                            const unsigned L, TagT *tags,
                                            float *distances,
                                            std::vector<T *> &res_vectors) {
        ScratchStoreManager<T> manager(_query_scratch);
        auto scratch = manager.scratch_space();

        if (L > scratch.search_l) {
            scratch.resize_for_query(L);
            MELON_LOG(INFO) << "Expanding query scratch_space. Was created with Lsize: "
                            << scratch.search_l << " but search L is: " << L;
        }
        uint32_t *indices = scratch.indices;
        float *dist_interim = scratch.interim_dists;
        search_impl(query, L, L, indices, dist_interim, scratch);

        std::shared_lock<std::shared_timed_mutex> ul(_update_lock);
        std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
        size_t pos = 0;

        for (int i = 0; i < (int) L; ++i) {
            TagT tag;

            if (_location_to_tag.try_get(indices[i], tag)) {
                tags[pos] = tag;

                if (res_vectors.size() > 0) {
                    memcpy(res_vectors[pos], _data + ((size_t) indices[i]) * _aligned_dim,
                           _dim * sizeof(T));
                }

                if (distances != nullptr) {
                    distances[pos] = _dist_metric == INNER_PRODUCT ? -1 * dist_interim[i]
                                                                   : dist_interim[i];
                }
                pos++;
                // If res_vectors.size() < k, clip at the value.
                if (pos == K || pos == res_vectors.size())
                    break;
            }
        }

        return pos;
    }

    template<typename T, typename TagT>
    size_t Index<T, TagT>::get_num_points() {
        return _nd;
    }

    template<typename T, typename TagT>
    size_t Index<T, TagT>::get_max_points() {
        return _max_points;
    }

    /*************************************************
     *      Support for Incremental Update
     *************************************************/

    template<typename T, typename TagT>
    int Index<T, TagT>::generate_frozen_point() {
        if (_num_frozen_pts == 0)
            return 0;

        if (_nd == 0) {
            memset(_data + (_max_points) * _aligned_dim, 0, _aligned_dim * sizeof(T));
            return 1;
        }
        size_t res = calculate_entry_point();
        memcpy(_data + _max_points * _aligned_dim, _data + res * _aligned_dim,
               _aligned_dim * sizeof(T));
        return 0;
    }

    template<typename T, typename TagT>
    int Index<T, TagT>::enable_delete() {
        assert(_enable_tags);

        if (!_enable_tags) {
            MELON_LOG(ERROR) << "Tags must be instantiated for deletions";
            return -2;
        }

        std::unique_lock<std::shared_timed_mutex> update_lock(_update_lock);
        if (_data_compacted) {
            for (unsigned slot = (unsigned) _nd; slot < _max_points; ++slot) {
                _empty_slots.insert(slot);
            }
        }

        _lazy_done = false;
        _eager_done = false;

        if (_support_eager_delete) {
            _in_graph.resize(_max_points + _num_frozen_pts);
            _in_graph.reserve(_max_points + _num_frozen_pts);
            update_in_graph();
        }
        return 0;
    }

    template<typename T, typename TagT>
    int Index<T, TagT>::eager_delete(const TagT tag, const Parameters &parameters,
                                     int delete_mode) {
        if (_lazy_done && (!_data_compacted)) {
            MELON_LOG(INFO) << "Lazy delete requests issued but data not consolidated, "
                               "cannot proceed with eager deletes.";
            return -1;
        }

        unsigned loc;  // since we will return if tag is not found, ok to leave it
        // uninitialized.
        {
            std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
            if (_tag_to_location.find(tag) == _tag_to_location.end()) {
                MELON_LOG(ERROR) << "Delete tag " << tag << " not found";
                return -1;
            }
            loc = _tag_to_location[tag];
        }

        {
            std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
            std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);
            _location_to_tag.erase(_tag_to_location[tag]);
            _tag_to_location.erase(tag);

            _delete_set.insert(loc);
            _empty_slots.insert(loc);
        }

        const unsigned range = parameters.Get<unsigned>("R");
        const unsigned maxc = parameters.Get<unsigned>("C");
        const float alpha = parameters.Get<float>("alpha");

        // delete point from out-neighbors' in-neighbor list
        {
            LockGuard guard(_locks[loc]);
            for (size_t i = 0; i < _final_graph[loc].size(); i++) {
                unsigned j = _final_graph[loc][i];
                {
                    LockGuard guard(_locks_in[j]);
                    for (unsigned k = 0; k < _in_graph[j].size(); k++) {
                        if (_in_graph[j][k] == loc) {
                            _in_graph[j].erase(_in_graph[j].begin() + k);
                            break;
                        }
                    }
                }
            }
        }

        melon::robin_set<unsigned> in_nbr;
        {
            LockGuard guard(_locks_in[loc]);
            for (unsigned i = 0; i < _in_graph[loc].size(); i++)
                in_nbr.insert(_in_graph[loc][i]);
        }
        assert(_in_graph[loc].size() == in_nbr.size());

        std::vector<neighbor> pool, tmp;
        melon::robin_set<unsigned> visited;
        std::vector<unsigned> intersection;
        unsigned Lindex = parameters.Get<unsigned>("L");
        std::vector<unsigned> init_ids;

        if (delete_mode == 2) {
            // constructing list of in-neighbors to be processed
            get_expanded_nodes(loc, Lindex, init_ids, pool, visited);

            for (auto node : visited) {
                if (in_nbr.find(node) != in_nbr.end()) {
                    intersection.push_back(node);
                }
            }
        }

        // deleting deleted point from all in-neighbors' out-neighbor list
        for (auto it : in_nbr) {
            LockGuard guard(_locks[it]);
            _final_graph[it].erase(
                    std::remove(_final_graph[it].begin(), _final_graph[it].end(), loc),
                    _final_graph[it].end());
        }

        melon::robin_set<unsigned> candidate_set;
        std::vector<neighbor> expanded_nghrs;
        std::vector<neighbor> result;

        for (size_t i = 0; i < intersection.size(); i++) {
            auto ngh = intersection[i];

            candidate_set.clear();
            expanded_nghrs.clear();
            result.clear();

            {
                std::shared_lock<std::shared_timed_mutex> lock(_delete_lock);
                if (_delete_set.find(ngh) != _delete_set.end())
                    continue;
            }

            {
                LockGuard guard(_locks[ngh]);

                // constructing candidate set from out-neighbors of ngh and loc
                {  // should a shared reader lock on delete_lock be held here at the
                    // beginning of the two for loops or should it be held and release
                    // for ech iteration of the for loops? Which is faster?

                    std::shared_lock<std::shared_timed_mutex> lock(_delete_lock);
                    for (auto j : _final_graph[loc]) {
                        if ((j != loc) && (j != ngh) &&
                            (_delete_set.find(j) == _delete_set.end()))
                            candidate_set.insert(j);
                    }

                    for (auto j : _final_graph[ngh]) {
                        if ((j != loc) && (j != ngh) &&
                            (_delete_set.find(j) == _delete_set.end()))
                            candidate_set.insert(j);
                    }
                }

                for (auto j : candidate_set)
                    expanded_nghrs.push_back(
                            neighbor(j,
                                     _distance->compare(_data + _aligned_dim * (size_t) ngh,
                                                        _data + _aligned_dim * (size_t) j,
                                                        (unsigned) _aligned_dim),
                                     true));
                std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
                occlude_list(expanded_nghrs, alpha, range, maxc, result);

                // deleting ngh from its old out-neighbors' in-neighbor list
                for (auto iter : _final_graph[ngh]) {
                    {
                        LockGuard guard(_locks_in[iter]);
                        for (unsigned k = 0; k < _in_graph[iter].size(); k++) {
                            if (_in_graph[iter][k] == ngh) {
                                _in_graph[iter].erase(_in_graph[iter].begin() + k);
                                break;
                            }
                        }
                    }
                }

                _final_graph[ngh].clear();

                // updating out-neighbors and in-neighbors of ngh
                {
                    std::shared_lock<std::shared_timed_mutex> lock(_delete_lock);
                    for (size_t i = 0; i < result.size(); i++) {
                        auto j = result[i];
                        if (_delete_set.find(j.id) == _delete_set.end()) {
                            _final_graph[ngh].push_back(j.id);
                            {
                                LockGuard guard(_locks_in[j.id]);
                                if (std::find(_in_graph[j.id].begin(), _in_graph[j.id].end(),
                                              ngh) == _in_graph[j.id].end()) {
                                    _in_graph[j.id].emplace_back(ngh);
                                }
                            }
                        }
                    }
                }
            }
        }

        _final_graph[loc].clear();
        _in_graph[loc].clear();

        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
        release_location(loc);

        _eager_done = true;
        _data_compacted = false;
        return 0;
    }

    template<typename T, typename TagT>
    void Index<T, TagT>::update_in_graph() {
        for (unsigned i = 0; i < _in_graph.size(); i++)
            _in_graph[i].clear();

        for (size_t i = 0; i < _final_graph.size();
             i++)  // copying to in-neighbor graph
            for (size_t j = 0; j < _final_graph[i].size(); j++)
                _in_graph[_final_graph[i][j]].emplace_back((uint32_t) i);
    }

    template<typename T, typename TagT>
    inline void Index<T, TagT>::process_delete(
            const melon::robin_set<unsigned> &old_delete_set, size_t i,
            const unsigned &range, const unsigned &maxc, const float &alpha) {
        melon::robin_set<unsigned> candidate_set;
        std::vector<neighbor> expanded_nghrs;
        std::vector<neighbor> result;

        bool modify = false;

        for (auto ngh : _final_graph[(uint32_t) i]) {
            if (old_delete_set.find(ngh) != old_delete_set.end()) {
                modify = true;

                // Add outgoing links from
                if (_conc_consolidate)
                    _locks[ngh].lock();
                for (auto j : _final_graph[ngh])
                    if (old_delete_set.find(j) == old_delete_set.end())
                        candidate_set.insert(j);
                if (_conc_consolidate)
                    _locks[ngh].unlock();
            } else {
                candidate_set.insert(ngh);
            }
        }
        if (modify) {
            for (auto j : candidate_set) {
                expanded_nghrs.push_back(
                        neighbor(j,
                                 _distance->compare(_data + _aligned_dim * i,
                                                    _data + _aligned_dim * (size_t) j,
                                                    (unsigned) _aligned_dim),
                                 true));
            }

            std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
            occlude_list(expanded_nghrs, alpha, range, maxc, result);

            {
                _final_graph[i].clear();

                for (auto j : result) {
                    if (j.id != (uint32_t) i &&
                        (old_delete_set.find(j.id) == old_delete_set.end()))
                        _final_graph[(uint32_t) i].push_back(j.id);
                }
            }
        }
    }

    // Returns number of live points left after consolidation
    template<typename T, typename TagT>
    consolidation_report Index<T, TagT>::consolidate_deletes(
            const Parameters &params) {
        MELON_CHECK(_enable_tags) << "Point tag array not instantiated";
        MELON_CHECK(!_eager_done) << "Can not consolidates eager deletes.";

        {
            std::shared_lock<std::shared_timed_mutex> ul(_update_lock);
            std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
            std::shared_lock<std::shared_timed_mutex> dl(_delete_lock);
            if (_empty_slots.size() + _nd != _max_points) {
                std::string err = "#empty slots + nd != max points";
                MELON_LOG(ERROR) << err;
                MELON_CHECK(false) << err;
            }

            if (_location_to_tag.size() + _delete_set.size() != _nd) {
                MELON_LOG(ERROR) << "Error: _location_to_tag.size ("
                                 << _location_to_tag.size() << ")  + _delete_set.size ("
                                 << _delete_set.size() << ") != _nd(" << _nd << ") ";
                return consolidation_report(lambda::consolidation_report::status_code::
                                            INCONSISTENT_COUNT_ERROR,
                                            0, 0, 0, 0, 0, 0);
            }

            if (_location_to_tag.size() != _tag_to_location.size()) {
                MELON_CHECK(false) <<
                                   "_location_to_tag and _tag_to_location not of same size";
            }
        }

        std::unique_lock<std::shared_timed_mutex> update_lock(_update_lock,
                                                              std::defer_lock);
        if (!_conc_consolidate)
            update_lock.lock();

        std::unique_lock<std::shared_timed_mutex> cl(_consolidate_lock,
                                                     std::defer_lock);
        if (not cl.try_lock()) {
            MELON_LOG(ERROR)
                    << "Consildate delete function failed to acquire consolidate lock";
            return consolidation_report(
                    lambda::consolidation_report::status_code::LOCK_FAIL, 0, 0, 0, 0, 0,
                    0);
        }

        MELON_LOG(INFO) << "Starting consolidate_deletes... ";

        melon::robin_set<unsigned> old_delete_set;
        {
            std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);
            _delete_set.swap(old_delete_set);
        }

        const unsigned range = params.Get<unsigned>("R");
        const unsigned maxc = params.Get<unsigned>("C");
        const float alpha = params.Get<float>("alpha");
        const unsigned num_threads = params.Get<unsigned>("num_threads") == 0
                                     ? omp_get_num_threads()
                                     : params.Get<unsigned>("num_threads");

        melon::stop_watcher timer;
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 8192)
        for (int64_t loc = 0; loc < (int64_t) _max_points; loc++) {
            if (old_delete_set.find((uint32_t) loc) == old_delete_set.end() &&
                not _empty_slots.is_in_set((uint32_t) loc)) {
                if (_conc_consolidate) {
                    LockGuard adj_list_lock(_locks[loc]);
                    process_delete(old_delete_set, loc, range, maxc, alpha);
                } else {
                    process_delete(old_delete_set, loc, range, maxc, alpha);
                }
            }
        }
        for (int64_t loc = _max_points; loc < (int64_t) (_max_points + _num_frozen_pts);
             loc++) {
            LockGuard adj_list_lock(_locks[loc]);
            process_delete(old_delete_set, loc, range, maxc, alpha);
        }
        if (_support_eager_delete)
            update_in_graph();

        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
        size_t ret_nd = release_locations(old_delete_set);

        if (!_conc_consolidate) {
            update_lock.unlock();
        }

        if (_delete_set.size() == 0)
            _lazy_done = false;

        double duration = timer.s_elapsed();
        MELON_LOG(INFO) << " done in " << duration << " seconds.";
        return consolidation_report(
                lambda::consolidation_report::status_code::SUCCESS, ret_nd,
                this->_max_points, _empty_slots.size(), old_delete_set.size(),
                _delete_set.size(), duration);
    }

    template<typename T, typename TagT>
    melon::result_status Index<T, TagT>::compact_frozen_point() {
        if (_nd < _max_points) {
            if (_num_frozen_pts == 1) {
                // set new _start to be frozen point
                _start = (uint32_t) _nd;
                if (!_final_graph[_max_points].empty()) {
                    for (unsigned i = 0; i < _nd; i++)
                        for (unsigned j = 0; j < _final_graph[i].size(); j++)
                            if (_final_graph[i][j] == _max_points)
                                _final_graph[i][j] = (uint32_t) _nd;

                    _final_graph[_nd].clear();
                    _final_graph[_nd].swap(_final_graph[_max_points]);

                    if (_support_eager_delete)
                        update_in_graph();

                    memcpy((void *) (_data + _aligned_dim * _nd),
                           _data + (size_t) _aligned_dim * _max_points, sizeof(T) * _dim);
                    memset((_data + (size_t) _aligned_dim * _max_points), 0,
                           sizeof(T) * _aligned_dim);
                }
            } else if (_num_frozen_pts > 1) {
                return melon::result_status(-1, "Case not implemented.");
            }
        }
        return melon::result_status::success();
    }

    // Should be called after acquiring _update_lock
    template<typename T, typename TagT>
    melon::result_status Index<T, TagT>::compact_data() {
        if (!_dynamic_index)
            return melon::result_status(-1, "Can not compact a non-dynamic index");

        if (_data_compacted) {
            MELON_LOG(ERROR)
                    << "Warning! Calling compact_data() when _data_compacted is true!";
            return melon::result_status::success();
        }

        std::unique_lock<std::shared_timed_mutex> cl(_consolidate_lock);

        if (_delete_set.size() > 0) {
            return melon::result_status(-1,
                                        "Can not compact data when index has non-trivial _delete_set of "
                                        "size: " +
                                        std::to_string(_delete_set.size()));
        }

        melon::stop_watcher timer;

        std::vector<unsigned> new_location =
                std::vector<unsigned>(_max_points + _num_frozen_pts, (uint32_t) UINT32_MAX);

        uint32_t new_counter = 0;
        std::set<uint32_t> empty_locations;
        for (uint32_t old_location = 0; old_location < _max_points; old_location++) {
            if (_location_to_tag.contains(old_location)) {
                new_location[old_location] = new_counter;
                new_counter++;
            } else {
                empty_locations.insert(old_location);
            }
        }
        for (uint32_t old_location = _max_points;
             old_location < _max_points + _num_frozen_pts; old_location++) {
            new_location[old_location] = old_location;
        }

        // If cur node is removed, replace it.
        if (_delete_set.find(_start) != _delete_set.end()) {
            MELON_LOG(ERROR) << "Replacing cur node which has been deleted... "
                             << std::flush;
            auto old_ep = _start;
            // First active neighbor of old cur node is new cur node
            for (auto iter : _final_graph[_start])
                if (_delete_set.find(iter) != _delete_set.end()) {
                    _start = iter;
                    break;
                }
            if (_start == old_ep) {
                return melon::result_status(-1,
                                            "ERROR: Did not find a replacement for cur node.");
            } else {
                assert(_delete_set.find(_start) == _delete_set.end());
            }
        }

        size_t num_dangling = 0;
        for (unsigned old = 0; old < _max_points + _num_frozen_pts; ++old) {
            if ((new_location[old] < _max_points)  // If point continues to exist
                || (old >= _max_points && old < _max_points + _num_frozen_pts)) {
                for (size_t i = 0; i < _final_graph[old].size(); ++i) {
                    if (empty_locations.find(_final_graph[old][i]) !=
                        empty_locations.end()) {
                        ++num_dangling;
                        MELON_LOG(ERROR) << "Error in compact_data(). _final_graph[" << old
                                         << "][" << i << "] = " << _final_graph[old][i]
                                         << " which is a location not associated with any tag.";
                        _final_graph[old].erase(_final_graph[old].begin() + i);
                        i--;
                    } else {
                        _final_graph[old][i] = new_location[_final_graph[old][i]];
                    }
                }

                if (_support_eager_delete)
                    for (size_t i = 0; i < _in_graph[old].size(); ++i) {
                        if (new_location[_in_graph[old][i]] <= _in_graph[old][i])
                            _in_graph[old][i] = new_location[_in_graph[old][i]];
                    }

                // Move the data and adj list to the correct position
                if (new_location[old] != old) {
                    assert(new_location[old] < old);
                    _final_graph[new_location[old]].swap(_final_graph[old]);
                    if (_support_eager_delete)
                        _in_graph[new_location[old]].swap(_in_graph[old]);
                    memcpy((void *) (_data + _aligned_dim * (size_t) new_location[old]),
                           (void *) (_data + _aligned_dim * (size_t) old),
                           _aligned_dim * sizeof(T));
                }
            } else {
                _final_graph[old].clear();
            }
        }
        MELON_LOG(ERROR) << "#dangling references after data compaction: "
                         << num_dangling;
        _tag_to_location.clear();
        for (auto pos = _location_to_tag.find_first(); pos.is_valid();
             pos = _location_to_tag.find_next(pos)) {
            const auto tag = _location_to_tag.get(pos);
            _tag_to_location[tag] = new_location[pos._key];
        }
        _location_to_tag.clear();
        for (auto iter : _tag_to_location) {
            _location_to_tag.set(iter.second, iter.first);
        }

        for (uint64_t old = _nd; old < _max_points; ++old) {
            _final_graph[old].clear();
        }
        _delete_set.clear();
        _empty_slots.clear();
        for (auto i = _nd; i < _max_points; i++) {
            _empty_slots.insert((uint32_t) i);
        }

        _eager_done = false;
        _data_compacted = true;
        MELON_LOG(INFO) << "Time taken for compact_data: "
                        << timer.elapsed() / 1000000. << "s.";
        return melon::result_status::success();
    }

    template<typename T, typename TagT>
    melon::result_status Index<T, TagT>::reserve_location(int &ret) {
        if (_nd >= _max_points) {
            return melon::result_status(-1, "overflow");
        }
        unsigned location;
        if (_data_compacted && _empty_slots.is_empty()) {
            // This code path is encountered when enable_delete hasn't been
            // called yet, so no points have been deleted and _empty_slots
            // hasn't been filled in. In that case, just keep assigning
            // consecutive locations.
            location = (unsigned) _nd;
        } else {
            // no need of delete_lock here, _tag_lock will ensure lazy delete does
            // not update empty slots
            assert(_empty_slots.size() != 0);
            assert(_empty_slots.size() + _nd == _max_points);

            auto r = _empty_slots.pop_any(location);
            if(!r) {
                return melon::result_status(-1, "no slots");
            }
            _delete_set.erase(location);
        }

        ++_nd;
        ret = location;
        return melon::result_status::success();
    }

    template<typename T, typename TagT>
    size_t Index<T, TagT>::release_location(int location) {
        if (_empty_slots.is_in_set(location))
            MELON_CHECK(false) <<
                               "Trying to release location, but location already in empty slots";
        _empty_slots.insert(location);

        _nd--;
        return _nd;
    }

    template<typename T, typename TagT>
    size_t Index<T, TagT>::release_locations(
            melon::robin_set<unsigned> &locations) {
        for (auto location : locations) {
            if (_empty_slots.is_in_set(location))
                MELON_CHECK(false) <<
                                   "Trying to release location, but location already in empty slots";
            _empty_slots.insert(location);

            _nd--;
        }

        if (_empty_slots.size() + _nd != _max_points)
            MELON_CHECK(false) << "#empty slots + nd != max points";

        return _nd;
    }

    template<typename T, typename TagT>
    void Index<T, TagT>::reposition_point(unsigned old_location,
                                          unsigned new_location) {
        for (unsigned i = 0; i < _nd; i++)
            for (unsigned j = 0; j < _final_graph[i].size(); j++)
                if (_final_graph[i][j] == old_location)
                    _final_graph[i][j] = (unsigned) new_location;

        _final_graph[new_location].clear();
        for (unsigned k = 0; k < _final_graph[_nd].size(); k++)
            _final_graph[new_location].emplace_back(_final_graph[old_location][k]);

        _final_graph[old_location].clear();

        if (_support_eager_delete) {
            update_in_graph();
        }
        memcpy((void *) (_data + (size_t) _aligned_dim * new_location),
               _data + (size_t) _aligned_dim * old_location,
               sizeof(T) * _aligned_dim);
        memset((_data + (size_t) _aligned_dim * old_location), 0,
               sizeof(T) * _aligned_dim);
    }

    template<typename T, typename TagT>
    void Index<T, TagT>::reposition_frozen_point_to_end() {
        if (_num_frozen_pts == 0)
            return;

        if (_nd == _max_points) {
            MELON_LOG(INFO) << "Not repositioning frozen point as it is already at the end.";
            return;
        }
        reposition_point((uint32_t) _nd, (uint32_t) _max_points);
        _start = (uint32_t) _max_points;
    }

    template<typename T, typename TagT>
    void Index<T, TagT>::resize(size_t new_max_points) {
        auto start = std::chrono::high_resolution_clock::now();
        assert(_empty_slots.size() ==
               0);  // should not resize if there are empty slots.
        T *new_data;
        alloc_aligned((void **) &new_data,
                      (new_max_points + 1) * _aligned_dim * sizeof(T),
                      8 * sizeof(T));
        memcpy(new_data, _data, (_max_points + 1) * _aligned_dim * sizeof(T));
        aligned_free(_data);
        _data = new_data;
        _final_graph.resize(new_max_points + 1);
        _locks = std::vector<non_recursive_mutex>(new_max_points + 1);
        if (_support_eager_delete) {
            _in_graph.resize(new_max_points + 1);
            _locks_in = std::vector<non_recursive_mutex>(new_max_points + 1);
        }

        reposition_point((uint32_t) _max_points, (uint32_t) new_max_points);
        _max_points = new_max_points;
        _start = (uint32_t) new_max_points;

        _empty_slots.reserve(_max_points);
        for (auto i = _nd; i < _max_points; i++) {
            _empty_slots.insert((uint32_t) i);
        }

        auto stop = std::chrono::high_resolution_clock::now();
        MELON_LOG(INFO) << "Resizing took: "
                        << std::chrono::duration<double>(stop - start).count() << "s";
    }

    template<typename T, typename TagT>
    melon::result_status Index<T, TagT>::insert_point(const T *point, const TagT tag) {
        assert(_has_built);

        std::shared_lock<std::shared_timed_mutex> shared_ul(_update_lock);
        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);

        // Find a vacant location in the data array to insert the new point
        int location;
        auto rs =  reserve_location(location);
        if (!rs.is_ok()) {
#if EXPAND_IF_FULL
            tl.unlock();
            shared_ul.unlock();

            {
              std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
              tl.lock();

              if (_nd >= _max_points) {
                auto new_max_points = (size_t)(_max_points * INDEX_GROWTH_FACTOR);
                resize(new_max_points);
              }

              tl.unlock();
              ul.unlock();
            }

            shared_ul.lock();
            tl.lock();

            location = reserve_location();
            if (location == -1) {
              throw lambda::ANNException(
                  "Cannot reserve location even after expanding graph. Terminating.",
                  -1, __FUNCSIG__, __FILE__, __LINE__);
            }
#else
            return rs;
#endif
        }

        // Insert tag and mapping to location
        if (_enable_tags) {
            if (_tag_to_location.find(tag) != _tag_to_location.end()) {
                release_location(location);
                return melon::result_status(-1, "already exists");
            }

            _tag_to_location[tag] = location;
            _location_to_tag.set(location, tag);
        }
        tl.unlock();

        // Copy the vector in to the data array
        auto offset_data = _data + (size_t) _aligned_dim * location;
        memset((void *) offset_data, 0, sizeof(T) * _aligned_dim);
        memcpy((void *) offset_data, point, sizeof(T) * _dim);

        if (_normalize_vecs) {
            normalize((float *) offset_data, _dim);
        }

        // Find and add appropriate graph edges
        std::vector<unsigned> pruned_list;

        ScratchStoreManager<T> manager(_query_scratch);
        auto scratch = manager.scratch_space();
        std::vector<neighbor> &pool = scratch.pool();
        melon::robin_set<unsigned> &visited = scratch.visited();
        pool.clear();
        visited.clear();
        search_for_point_and_add_links(location, _indexingQueueSize, pool, visited,
                                       scratch.des(), scratch.best_l_nodes(),
                                       scratch.inserted_into_pool_rs(),
                                       scratch.inserted_into_pool_bs());
        return melon::result_status::success();
    }

    template<typename T, typename TagT>
    int Index<T, TagT>::lazy_delete(const TagT &tag) {
        if ((_eager_done) && (!_data_compacted)) {
            MELON_CHECK(false) <<
                               "Eager delete requests were issued but data was not compacted, "
                               "cannot proceed with lazy_deletes";
        }
        std::shared_lock<std::shared_timed_mutex> ul(_update_lock);
        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
        std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);
        _lazy_done = true;
        _data_compacted = false;

        if (_tag_to_location.find(tag) == _tag_to_location.end()) {
            MELON_LOG(ERROR) << "Delete tag not found " << tag;
            return -1;
        }
        assert(_tag_to_location[tag] < _max_points);

        const auto location = _tag_to_location[tag];
        _delete_set.insert(location);
        _location_to_tag.erase(location);
        _tag_to_location.erase(tag);

        return 0;
    }

    template<typename T, typename TagT>
    melon::result_status Index<T, TagT>::lazy_delete(const std::vector<TagT> &tags,
                                                     std::vector<TagT> &failed_tags) {
        if (failed_tags.size() > 0) {
            return melon::result_status(-1, "failed_tags should be passed as an empty list");
        }
        if ((_eager_done) && (!_data_compacted)) {
            return melon::result_status(-1,
                                        "Eager delete requests were issued but data was not compacted, "
                                        "cannot proceed with lazy_deletes");
        }
        std::shared_lock<std::shared_timed_mutex> ul(_update_lock);
        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
        std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);
        _lazy_done = true;
        _data_compacted = false;

        for (auto tag : tags) {
            if (_tag_to_location.find(tag) == _tag_to_location.end()) {
                failed_tags.push_back(tag);
            } else {
                const auto location = _tag_to_location[tag];
                _delete_set.insert(location);
                _location_to_tag.erase(location);
                _tag_to_location.erase(tag);
            }
        }
        return melon::result_status::success();
    }

    template<typename T, typename TagT>
    bool Index<T, TagT>::is_index_saved() {
        return _is_saved;
    }

    template<typename T, typename TagT>
    void Index<T, TagT>::get_active_tags(melon::robin_set<TagT> &active_tags) {
        active_tags.clear();
        std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
        for (auto iter : _tag_to_location) {
            active_tags.insert(iter.first);
        }
    }

    template<typename T, typename TagT>
    void Index<T, TagT>::print_status() const {
        MELON_LOG(INFO) << "------------------- Index object: " << (uint64_t) this
                        << " -------------------";
        MELON_LOG(INFO) << "Number of points: " << _nd;
        MELON_LOG(INFO) << "Graph size: " << _final_graph.size();
        MELON_LOG(INFO) << "Location to tag size: " << _location_to_tag.size();
        MELON_LOG(INFO) << "Tag to location size: " << _tag_to_location.size();
        MELON_LOG(INFO) << "Number of empty slots: " << _empty_slots.size();
        MELON_LOG(INFO) << std::boolalpha
                        << "Data compacted: " << this->_data_compacted
                        << " Lazy done: " << this->_lazy_done
                        << " Eager done: " << this->_eager_done;
        MELON_LOG(INFO) << "---------------------------------------------------------"
                           "------------";
    }

    template<typename T, typename TagT>
    melon::result_status Index<T, TagT>::optimize_index_layout() {  // use after build or load
        if (_dynamic_index)
            return melon::result_status(-1,
                                        "Optimize_index_layout not implemented for dyanmic indices");

        _data_len = (_aligned_dim + 1) * sizeof(float);
        _neighbor_len = (_max_observed_degree + 1) * sizeof(unsigned);
        _node_size = _data_len + _neighbor_len;
        _opt_graph = (char *) malloc(_node_size * _nd);
        DistanceFastL2<T> *dist_fast = (DistanceFastL2<T> *) _distance;
        for (unsigned i = 0; i < _nd; i++) {
            char *cur_node_offset = _opt_graph + i * _node_size;
            float cur_norm = dist_fast->norm(_data + i * _aligned_dim, _aligned_dim);
            std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
            std::memcpy(cur_node_offset + sizeof(float), _data + i * _aligned_dim,
                        _data_len - sizeof(float));

            cur_node_offset += _data_len;
            unsigned k = _final_graph[i].size();
            std::memcpy(cur_node_offset, &k, sizeof(unsigned));
            std::memcpy(cur_node_offset + sizeof(unsigned), _final_graph[i].data(),
                        k * sizeof(unsigned));
            std::vector<unsigned>().swap(_final_graph[i]);
        }
        _final_graph.clear();
        _final_graph.shrink_to_fit();
        return melon::result_status::success();
    }

    template<typename T, typename TagT>
    void Index<T, TagT>::search_with_optimized_layout(const T *query, size_t K,
                                                      size_t L,
                                                      unsigned *indices) {
        DistanceFastL2<T> *dist_fast = (DistanceFastL2<T> *) _distance;

        std::vector<neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        melon::dynamic_bitset<> flags{_nd, 0};
        unsigned tmp_l = 0;
        unsigned *neighbors =
                (unsigned *) (_opt_graph + _node_size * _start + _data_len);
        unsigned MaxM_ep = *neighbors;
        neighbors++;

        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % _nd;
            if (flags[id])
                continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= _nd)
                continue;
            _mm_prefetch(_opt_graph + _node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= _nd)
                continue;
            T *x = (T *) (_opt_graph + _node_size * id);
            float norm_x = *x;
            x++;
            float dist =
                    dist_fast->compare(x, query, norm_x, (unsigned) _aligned_dim);
            retset[i] = neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(_opt_graph + _node_size * n + _data_len, _MM_HINT_T0);
                unsigned *neighbors =
                        (unsigned *) (_opt_graph + _node_size * n + _data_len);
                unsigned MaxM = *neighbors;
                neighbors++;
                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(_opt_graph + _node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    if (flags[id])
                        continue;
                    flags[id] = 1;
                    T *data = (T *) (_opt_graph + _node_size * id);
                    float norm = *data;
                    data++;
                    float dist =
                            dist_fast->compare(query, data, norm, (unsigned) _aligned_dim);
                    if (dist >= retset[L - 1].distance)
                        continue;
                    neighbor nn(id, dist, true);
                    int r = insert_into_pool(retset.data(), L, nn);

                    // if(L+1 < retset.size()) ++L;
                    if (r < nk)
                        nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }

    /*  Internals of the library */
    template<typename T, typename TagT>
    const float Index<T, TagT>::INDEX_GROWTH_FACTOR = 1.5f;

    // EXPORTS
    template MELON_EXPORT
    class Index<float, int32_t>;

    template MELON_EXPORT
    class Index<int8_t, int32_t>;

    template MELON_EXPORT
    class Index<uint8_t, int32_t>;

    template MELON_EXPORT
    class Index<float, uint32_t>;

    template MELON_EXPORT
    class Index<int8_t, uint32_t>;

    template MELON_EXPORT
    class Index<uint8_t, uint32_t>;

    template MELON_EXPORT
    class Index<float, int64_t>;

    template MELON_EXPORT
    class Index<int8_t, int64_t>;

    template MELON_EXPORT
    class Index<uint8_t, int64_t>;

    template MELON_EXPORT
    class Index<float, uint64_t>;

    template MELON_EXPORT
    class Index<int8_t, uint64_t>;

    template MELON_EXPORT
    class Index<uint8_t, uint64_t>;

    template MELON_EXPORT std::pair<uint32_t, uint32_t>
    Index<float, uint64_t>::search<uint64_t>(const float *query, const size_t K,
                                             const unsigned L, uint64_t *indices,
                                             float *distances);

    template MELON_EXPORT std::pair<uint32_t, uint32_t>
    Index<float, uint64_t>::search<uint32_t>(const float *query, const size_t K,
                                             const unsigned L, uint32_t *indices,
                                             float *distances);

    template MELON_EXPORT std::pair<uint32_t, uint32_t>
    Index<uint8_t, uint64_t>::search<uint64_t>(const uint8_t *query,
                                               const size_t K, const unsigned L,
                                               uint64_t *indices,
                                               float *distances);

    template MELON_EXPORT std::pair<uint32_t, uint32_t>
    Index<uint8_t, uint64_t>::search<uint32_t>(const uint8_t *query,
                                               const size_t K, const unsigned L,
                                               uint32_t *indices,
                                               float *distances);

    template MELON_EXPORT std::pair<uint32_t, uint32_t>
    Index<int8_t, uint64_t>::search<uint64_t>(const int8_t *query, const size_t K,
                                              const unsigned L, uint64_t *indices,
                                              float *distances);

    template MELON_EXPORT std::pair<uint32_t, uint32_t>
    Index<int8_t, uint64_t>::search<uint32_t>(const int8_t *query, const size_t K,
                                              const unsigned L, uint32_t *indices,
                                              float *distances);

    // TagT==uint32_t
    template MELON_EXPORT std::pair<uint32_t, uint32_t>
    Index<float, uint32_t>::search<uint64_t>(const float *query, const size_t K,
                                             const unsigned L, uint64_t *indices,
                                             float *distances);

    template MELON_EXPORT std::pair<uint32_t, uint32_t>
    Index<float, uint32_t>::search<uint32_t>(const float *query, const size_t K,
                                             const unsigned L, uint32_t *indices,
                                             float *distances);

    template MELON_EXPORT std::pair<uint32_t, uint32_t>
    Index<uint8_t, uint32_t>::search<uint64_t>(const uint8_t *query,
                                               const size_t K, const unsigned L,
                                               uint64_t *indices,
                                               float *distances);

    template MELON_EXPORT std::pair<uint32_t, uint32_t>
    Index<uint8_t, uint32_t>::search<uint32_t>(const uint8_t *query,
                                               const size_t K, const unsigned L,
                                               uint32_t *indices,
                                               float *distances);

    template MELON_EXPORT std::pair<uint32_t, uint32_t>
    Index<int8_t, uint32_t>::search<uint64_t>(const int8_t *query, const size_t K,
                                              const unsigned L, uint64_t *indices,
                                              float *distances);

    template MELON_EXPORT std::pair<uint32_t, uint32_t>
    Index<int8_t, uint32_t>::search<uint32_t>(const int8_t *query, const size_t K,
                                              const unsigned L, uint32_t *indices,
                                              float *distances);

}  // namespace lambda
