

#pragma once

#include "faiss/AutoTune.h"
#include "faiss/MetaIndexes.h"
#include "faiss/impl/AuxIndexStructures.h"
#include "faiss/index_io.h"
#include "lambda/ann_index.h"

namespace lambda {

    class FaissIndex : public ann_index {
    public:
        FaissIndex();

        ~FaissIndex();

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

        int load(const std::string &file, flare::write_lock &index_wlock) override;

        int save(const std::string &file) override;

        bool check_index(faiss::Index *tmp);

    private:
        faiss::Index *index_ = nullptr;
        bool support_update_ = false;
    };

    // virtual index to store vectors, for the purpose of brute force searching
    class FaissVirtualIndex : public ann_index {
    public:
        FaissVirtualIndex();

        ~FaissVirtualIndex();

        int init(const std::string &model) override {
            return 0;
        }

        size_t size() override {
            return 0;
        }

        bool support_update() override {
            return false;
        }

        bool support_delete() override {
            return false;
        }

        bool need_model() override {
            return false;
        }

        void add_with_ids(const std::vector<int64_t> &ids, std::vector<float> &vecs) override {
            return;
        }

        void search(
                search_options &option, std::vector<float> &distances, std::vector<int64_t> &labels) override;

        void remove(const std::set<uint64_t> &delete_ids) override {
            return;
        }

        void update(const std::vector<int64_t> &ids, std::vector<float> &vecs) override {
            return;
        }

        void clear() override {
            return;
        }

        int load(const std::string &file, flare::write_lock &index_wlock) override {
            return 0;
        }

        int save(const std::string &file) override {
            return 0;
        }
    };

}  // namespace lambda
