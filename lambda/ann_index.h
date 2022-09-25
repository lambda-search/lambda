

#pragma once

#include <stdint.h>

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "lambda/common/roaring.hh"
#include "melon/thread/rw_lock.h"
#include "lambda/proto/types.pb.h"

namespace lambda {

    using bitmap_t = roaring::Roaring;

#define DEFAULT_BATCH_SIZE 50000
#define DEFAULT_MAX_ELEMENTS 1000

    // 1.add_with_ids
    // 2.search
    // 3.save
    // 4.load

    // search params
    struct search_options {
        int64_t n = 0;  // query num
        int64_t k = 0;  // top k
        std::vector<float> vecs;  // query vector
        bitmap_t *bitmap = nullptr;  // filters
        uint32_t nprobe = 0;  // cluster num, for faiss
        float radius = 0;  // for range search
    };

    class ann_index {
    public:
        ann_index();

        virtual ~ann_index();

        virtual int init(const std::string &model) = 0;

        virtual size_t size() = 0;

        virtual bool support_update() = 0;

        virtual bool support_delete() = 0;

        virtual bool need_model() = 0;

        virtual void add_with_ids(const std::vector<int64_t> &ids, std::vector<float> &vecs) = 0;

        virtual void search(
                search_options &option, std::vector<float> &distances, std::vector<int64_t> &labels) = 0;

        virtual void remove(const std::set<uint64_t> &delete_ids) = 0;

        virtual void update(const std::vector<int64_t> &ids, std::vector<float> &vecs) = 0;

        virtual void clear() = 0;

        virtual int load(const std::string &file, melon::write_lock &index_wlock) = 0;

        virtual int save(const std::string &file) = 0;

        void set_max_elements(size_t max_elements) {
            if (max_elements > max_elements_) {
                max_elements_ = max_elements;
            }
        }

        void set_conf(const IndexShardConf &conf) {
            index_conf_ = conf;
        }

        virtual int build_batch_size() {
            return DEFAULT_BATCH_SIZE;
        }

        virtual void range_search(search_options &option, std::vector<std::vector<float>> &distances,
                                  std::vector<std::vector<int64_t>> &labels) {
        }

        virtual void shrink_to_fit() {
        }

    protected:
        IndexShardConf index_conf_;
        size_t max_elements_ = DEFAULT_MAX_ELEMENTS;
    };

    using ann_index_ptr = std::shared_ptr<ann_index>;

    // create index
    ann_index_ptr create_ann_index(const IndexShardConf &index_conf);

    // create object for index
    static ann_index_ptr create_solo_index(const IndexShardConf &index_conf);

    // doubly buffer adaptor
    static ann_index_ptr create_dbd_index(const IndexShardConf &index_conf);

    bool need_model(int engine);

    bool is_order_less(int engine, int metric);

};  // namespace lambda
