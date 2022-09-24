/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#pragma once

#include <atomic>
#include <cassert>
#include <map>
#include <shared_mutex>
#include <sstream>
#include <stack>
#include <string>
#include "flare/container/robin_set.h"
#include "flare/container/robin_map.h"
#include "flare/container/sparse_map.h"
#include "flare/container/dynamic_bitset.h"
#include "lambda/common/vector_distance.h"
#include "locking.h"
#include "flare/base/natural_number_map.h"
#include "flare/base/natural_number_set.h"
#include "lambda/common/math_utils.h"
#include "neighbor.h"
#include "parameters.h"
#include "utils.h"
#include "concurrent_queue.h"
#include <flare/base/profile.h>

#define GRAPH_SLACK_FACTOR 1.3
#define OVERHEAD_FACTOR 1.1
#define EXPAND_IF_FULL 0

namespace lambda {
    inline double estimate_ram_usage(uint64_t size, uint32_t dim, uint32_t datasize,
                                     uint32_t degree) {
        double size_of_data = ((double) size) * ROUND_UP(dim, 8) * datasize;
        double size_of_graph =
                ((double) size) * degree * sizeof(unsigned) * GRAPH_SLACK_FACTOR;
        double size_of_locks = ((double) size) * sizeof(non_recursive_mutex);
        double size_of_outer_vector = ((double) size) * sizeof(ptrdiff_t);

        return OVERHEAD_FACTOR * (size_of_data + size_of_graph + size_of_locks +
                                  size_of_outer_vector);
    }

    template<typename T>
    struct in_mem_query_scratch {
        std::vector<Neighbor> *_pool = nullptr;
        flare::robin_set<unsigned> *_visited = nullptr;
        std::vector<unsigned> *_des = nullptr;
        std::vector<Neighbor> *_best_l_nodes = nullptr;
        flare::robin_set<unsigned> *_inserted_into_pool_rs = nullptr;
        flare::dynamic_bitset<> *_inserted_into_pool_bs = nullptr;

        T *aligned_query = nullptr;
        uint32_t *indices = nullptr;
        float *interim_dists = nullptr;

        uint32_t search_l;
        uint32_t indexing_l;
        uint32_t r;

        in_mem_query_scratch();

        flare::result_status setup(uint32_t search_l, uint32_t indexing_l, uint32_t r, size_t dim);

        void clear();

        void resize_for_query(uint32_t new_search_l);

        void destroy();

        std::vector<Neighbor> &pool() {
            return *_pool;
        }

        std::vector<unsigned> &des() {
            return *_des;
        }

        flare::robin_set<unsigned> &visited() {
            return *_visited;
        }

        std::vector<Neighbor> &best_l_nodes() {
            return *_best_l_nodes;
        }

        flare::robin_set<unsigned> &inserted_into_pool_rs() {
            return *_inserted_into_pool_rs;
        }

        flare::dynamic_bitset<> &inserted_into_pool_bs() {
            return *_inserted_into_pool_bs;
        }
    };

    struct consolidation_report {
        enum status_code {
            SUCCESS = 0,
            FAIL = 1,
            LOCK_FAIL = 2,
            INCONSISTENT_COUNT_ERROR = 3
        };
        status_code _status;
        size_t _active_points, _max_points, _empty_slots, _slots_released,
                _delete_set_size;
        double _time;

        consolidation_report(status_code status, size_t active_points,
                             size_t max_points, size_t empty_slots,
                             size_t slots_released, size_t delete_set_size,
                             double time_secs)
                : _status(status), _active_points(active_points),
                  _max_points(max_points), _empty_slots(empty_slots),
                  _slots_released(slots_released), _delete_set_size(delete_set_size),
                  _time(time_secs) {
        }
    };

    template<typename T, typename TagT = uint32_t>
    class Index {
    public:
        // Constructor for Bulk operations and for creating the index object solely
        // for loading a prexisting index.
        FLARE_EXPORT Index(Metric m, const size_t dim,
                           const size_t max_points = 1,
                           const bool dynamic_index = false,
                           const bool enable_tags = false,
                           const bool support_eager_delete = false,
                           const bool concurrent_consolidate = false);

        // Constructor for incremental index
        FLARE_EXPORT Index(Metric m, const size_t dim, const size_t max_points,
                           const bool dynamic_index,
                           const Parameters &indexParameters,
                           const Parameters &searchParameters,
                           const bool enable_tags = false,
                           const bool support_eager_delete = false,
                           const bool concurrent_consolidate = false);

        FLARE_EXPORT ~Index();

        // Saves graph, data, metadata and associated tags.
        FLARE_EXPORT flare::result_status save(const char *filename,
                                               bool compact_before_save = false);

        // Load functions
        FLARE_EXPORT flare::result_status load(const char *index_file, uint32_t num_threads,
                                               uint32_t search_l);

        // get some private variables
        FLARE_EXPORT size_t get_num_points();

        FLARE_EXPORT size_t get_max_points();

        // Batch build from a file. Optionally pass tags vector.
        FLARE_EXPORT flare::result_status build(
                const char *filename, const size_t num_points_to_load,
                Parameters &parameters,
                const std::vector<TagT> &tags = std::vector<TagT>());

        // Batch build from a file. Optionally pass tags file.
        FLARE_EXPORT flare::result_status build(const char *filename,
                                                const size_t num_points_to_load,
                                                Parameters &parameters,
                                                const char *tag_filename);

        // Batch build from a data array, which must pad vectors to aligned_dim
        FLARE_EXPORT flare::result_status build(const T *data, const size_t num_points_to_load,
                                                Parameters &parameters,
                                                const std::vector<TagT> &tags);

        // Set starting point of an index before inserting any points incrementally
        FLARE_EXPORT flare::result_status set_start_point(T *data);
        // Set starting point to a random point on a sphere of certain radius
        FLARE_EXPORT void set_start_point_at_random(T radius);

        // For Bulk Index FastL2 search, we interleave the data with graph
        FLARE_EXPORT flare::result_status optimize_index_layout();

        // For FastL2 search on optimized layout
        FLARE_EXPORT void search_with_optimized_layout(const T *query,
                                                       size_t K, size_t L,
                                                       unsigned *indices);

        // Added search overload that takes L as parameter, so that we
        // can customize L on a per-query basis without tampering with "Parameters"
        template<typename IDType>
        FLARE_EXPORT std::pair<uint32_t, uint32_t> search(
                const T *query, const size_t K, const unsigned L, IDType *indices,
                float *distances = nullptr);

        // Initialize space for res_vectors before calling.
        FLARE_EXPORT size_t search_with_tags(const T *query, const uint64_t K,
                                             const unsigned L, TagT *tags,
                                             float *distances,
                                             std::vector<T *> &res_vectors);

        // Will fail if tag already in the index
        FLARE_EXPORT flare::result_status insert_point(const T *point, const TagT tag);

        // call this before issuing deletions to sets relevant flags
        FLARE_EXPORT int enable_delete();

        // Record deleted point now and restructure graph later. Return -1 if tag
        // not found, 0 if OK. Do not call if _eager_delete was called earlier and
        // data was not consolidated
        FLARE_EXPORT int lazy_delete(const TagT &tag);

        // Record deleted points now and restructure graph later. Add to failed_tags
        // if tag not found. Do not call if _eager_delete was called earlier and
        // data was not consolidated.
        FLARE_EXPORT flare::result_status lazy_delete(const std::vector<TagT> &tags,
                                                      std::vector<TagT> &failed_tags);

        // Call after a series of lazy deletions
        // Returns number of live points left after consolidation
        // If _conc_consolidates is set in the ctor, then this call can be invoked
        // alongside inserts and lazy deletes, else it acquires _update_lock
        FLARE_EXPORT consolidation_report
        consolidate_deletes(const Parameters &parameters);

        // Delete point from graph and restructure it immediately. Do not call if
        // _lazy_delete was called earlier and data was not consolidated
        FLARE_EXPORT int eager_delete(const TagT tag,
                                      const Parameters &parameters,
                                      int delete_mode = 1);

        FLARE_EXPORT void prune_all_nbrs(const Parameters &parameters);

        FLARE_EXPORT bool is_index_saved();

        // repositions frozen points to the end of _data - if they have been moved
        // during deletion
        FLARE_EXPORT void reposition_frozen_point_to_end();

        FLARE_EXPORT void reposition_point(unsigned old_location,
                                           unsigned new_location);

        // FLARE_EXPORT void save_index_as_one_file(bool flag);

        FLARE_EXPORT void get_active_tags(flare::robin_set<TagT> &active_tags);

        // memory should be allocated for vec before calling this function
        FLARE_EXPORT int get_vector_by_tag(TagT &tag, T *vec);

        FLARE_EXPORT void print_status() const;

        // This variable MUST be updated if the number of entries in the metadata
        // change.
        FLARE_EXPORT static const int METADATA_ROWS = 5;

        // ********************************
        //
        // Internals of the library
        //
        // ********************************

    protected:
        // No copy/assign.
        Index(const Index<T, TagT> &) = delete;

        Index<T, TagT> &operator=(const Index<T, TagT> &) = delete;

        // Use after _data and _nd have been populated
        flare::result_status build_with_data_populated(Parameters &parameters,
                                                       const std::vector<TagT> &tags);

        // generates 1 frozen point that will never be deleted from the graph
        // This is not visible to the user
        int generate_frozen_point();

        // determines navigating node of the graph by calculating medoid of data
        unsigned calculate_entry_point();

        // called only when _eager_delete is to be supported
        void update_in_graph();

        template<typename IDType>
        std::pair<uint32_t, uint32_t> search_impl(const T *query, const size_t K,
                                                  const unsigned L, IDType *indices,
                                                  float *distances,
                                                  in_mem_query_scratch<T> &scratch);

        std::pair<uint32_t, uint32_t> iterate_to_fixed_point(
                const T *node_coords, const unsigned Lindex,
                const std::vector<unsigned> &init_ids,
                std::vector<Neighbor> &expanded_nodes_info,
                flare::robin_set<unsigned> &expanded_nodes_ids,
                std::vector<Neighbor> &best_L_nodes, std::vector<unsigned> &des,
                flare::robin_set<unsigned> &inserted_into_pool_rs,
                flare::dynamic_bitset<> &inserted_into_pool_bs, bool ret_frozen = true,
                bool search_invocation = false);

        void get_expanded_nodes(const size_t node, const unsigned Lindex,
                                std::vector<unsigned> init_ids,
                                std::vector<Neighbor> &expanded_nodes_info,
                                flare::robin_set<unsigned> &expanded_nodes_ids,
                                std::vector<unsigned> &des,
                                std::vector<Neighbor> &best_L_nodes,
                                flare::robin_set<unsigned> &inserted_into_pool_rs,
                                flare::dynamic_bitset<> &inserted_into_pool_bs);

        // get_expanded_nodes for insertion. Must investigate to see if perf can
        // be improved here as well using the same technique as above.
        void get_expanded_nodes(const size_t node_id, const unsigned Lindex,
                                std::vector<unsigned> init_ids,
                                std::vector<Neighbor> &expanded_nodes_info,
                                flare::robin_set<unsigned> &expanded_nodes_ids);

        void search_for_point_and_add_links(
                int location, uint32_t Lindex, std::vector<Neighbor> &pool,
                flare::robin_set<unsigned> &visited, std::vector<unsigned> &des,
                std::vector<Neighbor> &best_L_nodes,
                flare::robin_set<unsigned> &inserted_into_pool_rs,
                flare::dynamic_bitset<> &inserted_into_pool_bs);

        void prune_neighbors(const unsigned location, std::vector<Neighbor> &pool,
                             std::vector<unsigned> &pruned_list);

        void prune_neighbors(const unsigned location, std::vector<Neighbor> &pool,
                             const uint32_t range, const uint32_t max_candidate_size,
                             const float alpha, std::vector<unsigned> &pruned_list);

        void occlude_list(std::vector<Neighbor> &pool, const float alpha,
                          const unsigned degree, const unsigned maxc,
                          std::vector<Neighbor> &result);

        // add reverse links from all the visited nodes to node n.
        void batch_inter_insert(unsigned n,
                                const std::vector<unsigned> &pruned_list,
                                const uint32_t range,
                                std::vector<unsigned> &need_to_sync);

        void batch_inter_insert(unsigned n,
                                const std::vector<unsigned> &pruned_list,
                                std::vector<unsigned> &need_to_sync);

        // add reverse links from all the visited nodes to node n.
        void inter_insert(unsigned n, std::vector<unsigned> &pruned_list,
                          const uint32_t range, bool update_in_graph);

        void inter_insert(unsigned n, std::vector<unsigned> &pruned_list,
                          bool update_in_graph);

        void link(Parameters &parameters);

        // Acquire _tag_lock before calling
        [[nodiscard]] flare::result_status reserve_location(int &ret);

        size_t release_location(int location);

        size_t release_locations(flare::robin_set<unsigned> &locations);

        // Resize the index when no slots are left for insertion.
        // MUST acquire _num_points_lock and _update_lock before calling.
        void resize(size_t new_max_points);

        // Take an unique lock on _update_lock and _consolidate_lock
        // before calling these functions.
        // Renumber nodes, update tag and location maps and compact the
        // graph, mode = _consolidated_order in case of lazy deletion and
        // _compacted_order in case of eager deletion
        FLARE_EXPORT flare::result_status compact_data();

        FLARE_EXPORT flare::result_status compact_frozen_point();

        // Remove deleted nodes from adj list of node i and absorb edges from
        // deleted neighbors Acquire _locks[i] prior to calling for thread-safety
        void process_delete(const flare::robin_set<unsigned> &old_delete_set,
                            size_t i, const unsigned &range, const unsigned &maxc,
                            const float &alpha);

        [[nodiscard]] flare::result_status initialize_query_scratch(uint32_t num_threads, uint32_t search_l,
                                                                    uint32_t indexing_l, uint32_t r, size_t dim);

        // Do not call without acquiring appropriate locks
        // call public member functions save and load to invoke these.
        FLARE_EXPORT uint64_t save_graph(std::string filename);

        FLARE_EXPORT uint64_t save_data(std::string filename);

        FLARE_EXPORT flare::result_status save_tags(std::string filename, size_t *saved_size);

        FLARE_EXPORT flare::result_status save_delete_list(const std::string &filename, size_t *saved_size);

        FLARE_EXPORT size_t load_graph(const std::string filename,
                                       size_t expected_num_points);

        FLARE_EXPORT size_t load_data(std::string filename0);

        FLARE_EXPORT [[nodiscard]] flare::result_status load_tags(const std::string tag_file_name, size_t &n);

        FLARE_EXPORT [[nodiscard]] flare::result_status load_delete_set(const std::string &filename, size_t &ret);

    private:
        flare::result_status _status;
        // vector_distance functions
        Metric _dist_metric = lambda::L2;
        vector_distance<T> *_distance = nullptr;

        // Data
        T *_data = nullptr;
        char *_opt_graph;

        // Graph related data structures
        std::vector<std::vector<unsigned>> _final_graph;
        std::vector<std::vector<unsigned>> _in_graph;

        // Dimensions
        size_t _dim = 0;
        size_t _aligned_dim = 0;
        size_t _nd = 0;  // number of active points i.e. existing in the graph
        size_t _max_points = 0;  // total number of points in given data set
        size_t _num_frozen_pts = 0;
        size_t _max_range_of_loaded_graph = 0;
        size_t _node_size;
        size_t _data_len;
        size_t _neighbor_len;

        unsigned _max_observed_degree = 0;
        unsigned _start = 0;

        bool _has_built = false;
        bool _saturate_graph = false;
        bool _save_as_one_file = false;  // plan to support in next version
        bool _dynamic_index = false;
        bool _enable_tags = false;
        bool _normalize_vecs = false;  // Using normalied L2 for cosine.

        // Indexing parameters
        uint32_t _indexingQueueSize;
        uint32_t _indexingRange;
        uint32_t _indexingMaxC;
        float _indexingAlpha;
        uint32_t _search_queue_size;

        // Query scratch data structures
        ConcurrentQueue<in_mem_query_scratch<T>> _query_scratch;

        // data structures, flags and locks for dynamic indexing
        flare::sparse_map<TagT, unsigned> _tag_to_location;
        flare::natural_number_map<unsigned, TagT> _location_to_tag;

        flare::robin_set<unsigned> _delete_set;
        flare::natural_number_set<unsigned> _empty_slots;

        bool _support_eager_delete =
                false;  // Enables in-graph, requires more space

        bool _eager_done = false;     // true if eager deletions have been made
        bool _lazy_done = false;      // true if lazy deletions have been made
        bool _data_compacted = true;  // true if data has been compacted
        bool _is_saved = false;  // Gopal. Checking if the index is already saved.
        bool _conc_consolidate = false;  // use _lock while searching

        std::vector<non_recursive_mutex>
                _locks;  // Per node lock, cardinality=max_points_
        std::vector<non_recursive_mutex> _locks_in;  // Per node lock

        // If acquiring multiple locks below, acquire locks in the order below
        std::shared_timed_mutex  // RW mutex between save/load (exclusive lock) and
        _update_lock;        // search/inserts/deletes/consolidate (shared lock)
        std::shared_timed_mutex
                _consolidate_lock;  // Ensure only one consolidate is ever active
        std::shared_timed_mutex
                _tag_lock;  // RW lock for _tag_to_location and _location_to_tag
        std::shared_timed_mutex
                _delete_lock;  // RW Lock on _delete_set and _empty_slots

        static const float INDEX_GROWTH_FACTOR;
    };
}  // namespace lambda
