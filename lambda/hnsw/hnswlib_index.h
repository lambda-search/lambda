
#pragma once

#include <stdint.h>

#include <queue>
#include <string>
#include <utility>

#include "lambda/ann_index.h"
#include "flare/thread/rw_lock.h"

namespace hnswlib {

    template<typename T>
    class SpaceInterface;

    template<typename T>
    class HierarchicalNSW;

}  // namespace hnswlib

namespace lambda {

#define DEFAULT_M 16  // defines tha maximum number of outgoing connections in the graph
#define DEFAULT_KR_CONSTRUCTION 200  // defines a construction time/accuracy trade-off
#define DEFAULT_KR 50  // the query time accuracy/speed trade-off
#define DEFAULT_HNSW_MAX_ELEMENTS \
  50000  // defines the maximum number of elements that can be stored in the structure
#define DEFAULT_RANDOM_SEED 100  // default random seed

    // max_elements defines the maximum number of elements that can be
    // stored in the structure(can be increased/shrunk).
    // kr_construction defines a construction time/accuracy trade-off.
    // M defines tha maximum number of outgoing connections in the graph.
    struct hnsw_param {
        size_t m = DEFAULT_M;
        size_t kr_construction = DEFAULT_KR_CONSTRUCTION;
        size_t kr = DEFAULT_KR;
        size_t max_elements = DEFAULT_HNSW_MAX_ELEMENTS;
    };

    class HnswLibSpace;

    // HNSW: https://github.com/nmslib/hnswlib
    class hnsw_index : public ann_index {
    public:
        hnsw_index();

        ~hnsw_index();

        // initializes the index from with no elements.
        int init_new_index(const hnsw_param &param, const size_t random_seed);

        // set/get the query time accuracy/speed trade-off, defined by the kr parameter
        // Note that the parameter is currently not saved along with the index,
        // so you need to set it manually after loading
        void set_kr(size_t kr);

        size_t get_kr();

        // kr_construction defines a construction time/accuracy trade-off
        size_t get_kr_construction();

        // M defines tha maximum number of outgoing connections in the graph
        size_t get_m();

        // set the default number of cpu threads used during data insertion/querying
        void set_num_threads(int num_threads);

        // inserts the data(numpy array of vectors, shape:N*dim) into the structure
        // labels is an optional N-size numpy array of integer labels for all elements in data.
        // num_threads sets the number of cpu threads to use (-1 means use default).
        // data_labels specifies the labels for the data. If index already has the elements with
        // the same labels, their features will be updated. Note that update procedure is slower
        // than insertion of a new element, but more memory- and query-efficient.
        // Thread-safe with other add_items calls, but not with knn_query.
        void add_item(size_t id, float *vector);

        // get origin vector by it's id
        std::vector<float> get_vector_by_id(size_t id);

        // returns a list of all elements' ids
        std::vector<unsigned int> get_ids_list();

        // make a batch query for k closests elements for each element of the
        // data (shape:N*dim). Returns a numpy array of (shape:N*k).
        // num_threads sets the number of cpu threads to use (-1 means use default).
        // Thread-safe with other knn_query calls, but not with add_items
        std::priority_queue<std::pair<float, size_t>> knn_query(
                const float *vector, size_t k, const bitmap_t *bitmap);

        // marks the element as deleted, so it will be ommited from search results.
        void mark_deleted(size_t label);

        // changes the maximum capacity of the index. Not thread safe with add_items and knn_query
        void resize_index(size_t new_size);

        int init(const std::string &model) override;

        size_t size() override;

        bool support_update()  override {
            return true;
        }

        bool support_delete()  override {
            return true;
        }

        bool need_model()  override {
            return false;
        }

        void shrink_to_fit() override;

        void add_with_ids(const std::vector<int64_t> &ids, std::vector<float> &vecs) override;

        void search(search_options &option, std::vector<float> &distances, std::vector<int64_t> &labels) override;

        void remove(const std::set<uint64_t> &delete_ids) override;

        void update(const std::vector<int64_t> &ids, std::vector<float> &vecs) override;

        void clear() override;

        int load(const std::string &file, flare::write_lock &index_wlock) override;

        int save(const std::string &file) override;

        void init_params(hnsw_param &param);

    private:
        bool index_inited_ = false;
        bool ep_added_ = false;
        bool normalize_ = false;
        int num_threads_default_ = 0;
        hnswlib::HierarchicalNSW<float> *appr_alg_ = nullptr;
        hnswlib::SpaceInterface<float> *l2space_ = nullptr;
    };

}  // namespace lambda
