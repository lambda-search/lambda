
#include "lambda/hnsw/hnswlib_index.h"
#include <iostream>
#include <thread>

#include "lambda/hnsw/hnswlib.h"
#include "flare/strings/numbers.h"
#include "flare/strings/str_split.h"

DEFINE_bool(hnswlib_dynamic_kr, true, "Dynamic to set lambda argmuent for hnswlib index");

namespace lambda {

    hnsw_index::hnsw_index() {
    }

    hnsw_index::~hnsw_index() {
        clear();
    }

    // initializes the index from with no elements
    int hnsw_index::init_new_index(const hnsw_param &param, const size_t random_seed) {
        if (appr_alg_) {
            FLARE_LOG(ERROR) << "The index is already initiated.";
            return -1;
        }
        try {
            appr_alg_ = new hnswlib::HierarchicalNSW<float>(
                    l2space_, param.max_elements, param.m, param.kr_construction, random_seed);
        } catch (std::exception &e) {
            FLARE_LOG(ERROR) << "HNSWLIB exception: " << e.what();
            return -1;
        } catch (...) {
            FLARE_LOG(ERROR) << "Unkown exception";
            return -1;
        }
        if (nullptr == appr_alg_) {
            FLARE_LOG(ERROR) << "Fail to create HNSWLIB index.";
            return -1;
        }
        index_inited_ = true;
        ep_added_ = false;
        return 0;
    }

    void hnsw_index::init_params(hnsw_param &param) {
        std::vector<std::string> opts = flare::string_split(index_conf_.conf().description(), ",");
        for (const std::string &opt : opts) {
            std::vector<std::string> kv = flare::string_split(opt, "=");
            if (kv.size() != 2) {
                FLARE_LOG(ERROR) << "Invalid option:" << opt;
                continue;
            }
            int64_t v;
            if (!flare::simple_atoi(kv[1], &v)) {
                FLARE_LOG(ERROR) << "Invalid option:" << opt;
                continue;
            }
            if (v <= 0) {
                FLARE_LOG(ERROR) << "Invalid option:" << opt;
                continue;
            }
            if (!strcasecmp(kv[0].c_str(), "m")) {
                param.m = v;
            } else if (!strcasecmp(kv[0].c_str(), "kr_construction")) {
                param.kr_construction = v;
            } else if (!strcasecmp(kv[0].c_str(), "kr")) {
                param.kr = v;
            } else if (!strcasecmp(kv[0].c_str(), "max_elements")) {
                param.max_elements = v;
            }
        }
    }

    int hnsw_index::init(const std::string &model) {
        normalize_ = false;
        if (index_conf_.conf().metric() == INDEX_METRIC_L2) {
            l2space_ = new hnswlib::L2Space(index_conf_.conf().dimension());
        } else if (index_conf_.conf().metric() == INDEX_METRIC_INNER_PRODUCT) {
            l2space_ = new hnswlib::InnerProductSpace(index_conf_.conf().dimension());
        } else if (index_conf_.conf().metric() == INDEX_METRIC_COSINE) {
            l2space_ = new hnswlib::InnerProductSpace(index_conf_.conf().dimension());
            normalize_ = true;
        }
        if (nullptr == l2space_) {
            return -1;
        }

        hnsw_param param;
        init_params(param);
        appr_alg_ = nullptr;
        ep_added_ = true;
        index_inited_ = false;
        num_threads_default_ = std::thread::hardware_concurrency();
        set_max_elements(param.max_elements);
        int ret = init_new_index(param, DEFAULT_RANDOM_SEED);
        if (0 != ret) {
            return ret;
        }
        set_kr(param.kr);
        return 0;
    }

    size_t hnsw_index::size() {
        if (nullptr == appr_alg_) {
            return 0;
        }
        return appr_alg_->cur_element_count;
    }

    void hnsw_index::add_with_ids(const std::vector<int64_t> &ids, std::vector<float> &vecs) {
        if (nullptr == appr_alg_) {
            return;
        }
        for (size_t i = 0; i < ids.size(); i++) {
            add_item(ids[i], vecs.data() + i * index_conf_.conf().dimension());
        }
    }

    void hnsw_index::search(
            search_options &option, std::vector<float> &distances, std::vector<int64_t> &labels) {
        if (0 == size()) {
            return;
        }

        // lambda - the size of the dynamic list for the nearest neighbors (used during the search).
        // Higher kr leads to more accurate but slower search.
        // kr cannot be set lower than the number of queried nearest neighbors k.
        // The value kr of can be anything between k and the size of the dataset.
        // 实测kr < k也没有问题,为了兼容增加开关
        if (FLAGS_hnswlib_dynamic_kr) {
            size_t kr = option.k * 2;
            if (get_kr() < kr) {
                set_kr(kr);
            }
        }

        const auto &metric = index_conf_.conf().metric();
        for (int64_t row = 0; row < option.n; row++) {
            std::priority_queue<std::pair<float, size_t>> result = knn_query(
                    option.vecs.data() + row * index_conf_.conf().dimension(), option.k, option.bitmap);
            int offset = row * option.k;
            for (int i = result.size() - 1; i >= 0; i--) {
                auto &result_tuple = result.top();
                int position = offset + i;
                if (metric == INDEX_METRIC_INNER_PRODUCT || metric == INDEX_METRIC_COSINE) {
                    distances[position] = 1.0 - result_tuple.first;
                } else {
                    distances[position] = result_tuple.first;
                }
                labels[position] = result_tuple.second;
                result.pop();
                // FLARE_LOG(INFO) << "row:" << row << ",position:" << position << ",labels:" << labels[position]
                //          << ",distances:" << distances[position];
            }
        }
    }


    void hnsw_index::remove(const std::set<uint64_t> &delete_ids) {
        if (nullptr == appr_alg_) {
            return;
        }
        for (auto id : delete_ids) {
            try {
                appr_alg_->markDelete(id);
            } catch (std::exception &e) {
                FLARE_LOG(ERROR) << "Failed to delete id:" << id << " with err:" << e.what();
            } catch (...) {
                FLARE_LOG(ERROR) << "Unkown exception";
            }
        }
    }

    void hnsw_index::clear() {
        if (l2space_) {
            delete l2space_;
            l2space_ = nullptr;
        }
        if (appr_alg_) {
            delete appr_alg_;
            appr_alg_ = nullptr;
        }
    }

    int hnsw_index::load(const std::string &path_to_index, flare::write_lock &index_wlock) {
        try {
            auto appr_alg_tmp = new hnswlib::HierarchicalNSW<float>(l2space_, path_to_index, false, 0);
            if (nullptr == appr_alg_tmp) {
                return -1;
            }
            FLARE_LOG(INFO) << "Calling load_index for an already inited index. Old index is being deallocated.";
            std::lock_guard<flare::write_lock> guard(index_wlock);
            if (appr_alg_) {
                delete appr_alg_;
            }
            appr_alg_ = appr_alg_tmp;
        } catch (std::exception &e) {
            FLARE_LOG(ERROR) << "Failed to load index:" << path_to_index << " with err:" << e.what();
            return -1;
        } catch (...) {
            FLARE_LOG(ERROR) << "Unkown exception";
            return -1;
        }
        return 0;
    }

    int hnsw_index::save(const std::string &file) {
        try {
            if (appr_alg_) {
                appr_alg_->saveIndex(file);
            }
        } catch (std::exception &e) {
            FLARE_LOG(ERROR) << "Failed to save index:" << file << " with err:" << e.what();
            return -1;
        } catch (...) {
            FLARE_LOG(ERROR) << "Unkown exception";
            return -1;
        }
        return 0;
    }

    void hnsw_index::shrink_to_fit() {
        if (nullptr == appr_alg_) {
            return;
        }
        try {
            if (appr_alg_->max_elements_ > size() + 1024) {
                appr_alg_->resizeIndex(size() + 1024);
            }
        } catch (std::exception &e) {
            FLARE_LOG(ERROR) << "Failed to shrink to fit with err:" << e.what();
        } catch (...) {
            FLARE_LOG(ERROR) << "Unkown exception";
        }
    }

    void hnsw_index::set_kr(size_t kr) {
        appr_alg_->_kr = kr;
    }

    size_t hnsw_index::get_kr() {
        return appr_alg_->_kr;
    }

    size_t hnsw_index::get_kr_construction() {
        return appr_alg_->ef_construction_;
    }

    size_t hnsw_index::get_m() {
        return appr_alg_->M_;
    }

    void hnsw_index::set_num_threads(int num_threads) {
        this->num_threads_default_ = num_threads;
    }

    void hnsw_index::add_item(size_t id, float *vector) {
        try {
            if (size() >= appr_alg_->max_elements_) {
                appr_alg_->resizeIndex(appr_alg_->max_elements_ * 2);
            }
            appr_alg_->addPoint(reinterpret_cast<void *>(vector), id);
        } catch (std::exception &e) {
            FLARE_LOG(ERROR) << "Failed to add id:" << id << " with err:" << e.what();
        } catch (...) {
            FLARE_LOG(ERROR) << "Unkown exception";
        }
    }

    std::vector<float> hnsw_index::get_vector_by_id(size_t id) {
        try {
            return appr_alg_->template getDataByLabel<float>(id);
        } catch (std::exception &e) {
            FLARE_LOG(ERROR) << "Failed to get vector by id:" << id << " with err:" << e.what();
        } catch (...) {
            FLARE_LOG(ERROR) << "Unkown exception";
        }
        return std::vector<float>();
    }

    std::vector<unsigned int> hnsw_index::get_ids_list() {
        std::vector<unsigned int> ids;
        for (auto kv : appr_alg_->label_lookup_) {
            ids.push_back(kv.first);
        }
        return ids;
    }

    std::priority_queue<std::pair<float, size_t>> hnsw_index::knn_query(
            const float *vector, size_t k, const bitmap_t *bitmap) {
        try {
            return appr_alg_->searchKnn(static_cast<const void *>(vector), bitmap, k);
        } catch (std::exception &e) {
            FLARE_LOG(ERROR) << "Failed to search index:" << index_conf_.index() << " with err:" << e.what();
        } catch (...) {
            FLARE_LOG(ERROR) << "Unkown exception";
        }
        return std::priority_queue<std::pair<float, size_t>>();
    }

    void hnsw_index::mark_deleted(size_t label) {
        try {
            appr_alg_->markDelete(label);
        } catch (std::exception &e) {
            FLARE_LOG(ERROR) << "Failed to delete label:" << label << " with err:" << e.what();
        } catch (...) {
            FLARE_LOG(ERROR) << "Unkown exception";
        }
    }

    void hnsw_index::resize_index(size_t new_size) {
        try {
            appr_alg_->resizeIndex(new_size);
        } catch (std::exception &e) {
            FLARE_LOG(ERROR) << "Failed to resize index to:" << new_size << " with err:" << e.what();
        } catch (...) {
            FLARE_LOG(ERROR) << "Unkown exception";
        }
    }

    void hnsw_index::update(const std::vector<int64_t> &ids, std::vector<float> &vecs) {
        add_with_ids(ids, vecs);
    }

}  // namespace lambda
