
#pragma once

#include "faiss/AutoTune.h"
#include "faiss/MetaIndexes.h"
#include "faiss/impl/AuxIndexStructures.h"
#include "faiss/index_io.h"
#include "lambda/ann_index.h"

namespace lambda {

    class FaissBinaryIndex : public ann_index {
    public:
        FaissBinaryIndex();

        ~FaissBinaryIndex();

        int init(const std::string &model) override;

        size_t size() override;

        bool support_update() override;

        bool support_delete() override;

        bool need_model() override {
            return true;
        }

        void add_with_ids(const std::vector<int64_t> &ids, std::vector<float> &vecs) override;

        void range_search(search_options &option, std::vector<std::vector<float>> &distances,
                          std::vector<std::vector<int64_t>> &labels) override;

        void search(
                search_options &option, std::vector<float> &distances, std::vector<int64_t> &labels) override;


        void remove(const std::set<uint64_t> &delete_ids) override;

        void update(const std::vector<int64_t> &ids, std::vector<float> &vecs) override;

        void clear() override;

        int load(const std::string &file, melon::write_lock &index_wlock) override;

        int save(const std::string &file) override;

        bool check_index(faiss::IndexBinary *tmp);

    private:
        faiss::IndexBinary *index_ = nullptr;
        bool support_update_ = false;
    };

}  // namespace lambda
