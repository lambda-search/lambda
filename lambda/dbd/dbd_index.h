
#pragma once

#include <memory>
#include <string>
#include <set>
#include <vector>

#include "lambda/common/doubly_buffered_data.h"
#include "melon/log/logging.h"

#include "lambda/ann_index.h"

namespace lambda {

    using DBDScopedPtr = doubly_buffered_data<ann_index_ptr>::ScopedPtr;

    class dbd_index : public ann_index {
    public:
        dbd_index() {}

        dbd_index(ann_index_ptr fg_index, ann_index_ptr bg_index);

        dbd_index(const dbd_index &other) = delete;

        dbd_index(dbd_index &&other) = delete;

        ~dbd_index() {}

        dbd_index &operator=(const dbd_index &other) = delete;

        dbd_index &operator=(dbd_index &&other) = delete;

        int init(const std::string &model) override;

        size_t size() override;

        bool support_update() override;

        bool support_delete() override;

        bool need_model() override;

        void add_with_ids(const std::vector<int64_t> &ids, std::vector<float> &vecs) override;

        void search(search_options &option,
                    std::vector<float> &distances,
                    std::vector<int64_t> &labels) override;

        void remove(const std::set<uint64_t> &delete_ids) override;

        void update(const std::vector<int64_t> &ids, std::vector<float> &vecs) override;

        void clear() override;

        int load(const std::string &file, melon::write_lock &index_wlock) override;

        int save(const std::string &file) override;

        int build_batch_size() override;

        void range_search(search_options &option, std::vector<std::vector<float>> &distances,
                          std::vector<std::vector<int64_t>> &labels) override;

        void shrink_to_fit() override;

    private:
        doubly_buffered_data<ann_index_ptr> ann_index_;
    };

}  // namespace lambda
