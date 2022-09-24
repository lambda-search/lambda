
#include "lambda/faiss/faiss_index.h"
#include <flare/log/logging.h>
#include <flare/files/filesystem.h>
#include "faiss/AutoTune.h"
#include "faiss/IVFlib.h"
#include "faiss/IndexBinaryFlat.h"
#include "faiss/IndexBinaryIVF.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexIVFPQ.h"

namespace lambda {

    struct bitmap_id_selector : public faiss::IDSelector {
        explicit bitmap_id_selector(bitmap_t *m) : map(m) {}

        bool is_member(idx_t id) const override {
            return map->contains(id);
        }

        bitmap_t *map;
    };

    int FaissIndex::init(const std::string &model) {
        const auto &index_name = index_conf_.index();
        const auto &conf = index_conf_.conf();
        std::string model_path = model;
        std::error_code ec;
        if (!flare::exists(model_path, ec)) {
            FLARE_LOG(ERROR) << "model file not exist: " << model_path;
            return -1;
        }

        faiss::Index *tmp = nullptr;
        try {
            tmp = faiss::read_index(model_path.c_str(), 0);
        } catch (faiss::FaissException &e) {
            FLARE_LOG(ERROR) << "Faiss exception: " << e.what();
            return -1;
        } catch (std::exception &e) {
            FLARE_LOG(ERROR) << "Faiss exception: " << e.what();
            return -1;
        } catch (...) {
            FLARE_LOG(ERROR) << "unkown exception";
            return -1;
        }
        if (nullptr == tmp) {
            FLARE_LOG(ERROR) << "Fail to read index from model " << model_path << " with engine "
                       << conf.engine();
            return -1;
        }

        if (!check_index(tmp)) {
            tmp->reset();
            delete tmp;
            return -1;
        }

        clear();
        index_ = tmp;
        return 0;
    }

    bool FaissIndex::check_index(faiss::Index *tmp) {
        try {
            const auto &conf = index_conf_.conf();
            if (conf.nprobe() > 0) {
                faiss::ParameterSpace().set_index_parameter(tmp, "nprobe", conf.nprobe());
            }
            faiss::IndexIVF *ivf = dynamic_cast<faiss::IndexIVF *>(tmp);
            // PCA nullptr
            if (nullptr != ivf) {
                if (conf.dimension() != ivf->d) {
                    FLARE_LOG(ERROR) << "index " << index_conf_.index() << " check failed, expect dimension "
                               << conf.dimension() << " but got " << ivf->d;
                    return false;
                }
                if (static_cast<faiss::MetricType>(conf.metric()) != ivf->metric_type) {
                    FLARE_LOG(ERROR) << "index " << index_conf_.index() << " check failed, expect metric "
                               << conf.metric() << " but got " << ivf->metric_type;
                    ivf->metric_type = static_cast<faiss::MetricType>(conf.metric());
                }
            } else {
                // PCA
                ivf = faiss::ivflib::extract_index_ivf(tmp);
            }
            if (nullptr == ivf) {
                return true;
            }
            if (conf.direct_map()) {
                ivf->set_direct_map_type(faiss::DirectMap::Hashtable);
                support_update_ = true;
            }
        } catch (faiss::FaissException &e) {
            FLARE_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (std::exception &e) {
            FLARE_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (...) {
            FLARE_LOG(ERROR) << "unkown exception";
        }
        return true;
    }

    size_t FaissIndex::size() {
        if (nullptr == index_) {
            return 0;
        }
        return (size_t) (index_->ntotal);
    }

    bool FaissIndex::support_update() {
        return support_update_;
    }

    bool FaissIndex::support_delete() {
        return support_update_;
    }

    void FaissIndex::add_with_ids(const std::vector<int64_t> &ids, std::vector<float> &vecs) {
        if (nullptr == index_) {
            FLARE_LOG(ERROR) << "index " << index_conf_.index() << ", shard " << index_conf_.shard_idx()
                       << " is not initialized";
            return;
        }
        try {
            index_->add_with_ids(ids.size(), vecs.data(), ids.data());
        } catch (faiss::FaissException &e) {
            FLARE_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (std::exception &e) {
            FLARE_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (...) {
            FLARE_LOG(ERROR) << "unkown exception";
        }
    }

    void FaissIndex::range_search(search_options &option, std::vector<std::vector<float>> &distances,
                                  std::vector<std::vector<int64_t>> &labels) {
        if (nullptr == index_ || index_->ntotal <= 0) {
            return;
        }

        faiss::RangeSearchResult result(option.n);
        try {
            if (option.bitmap == nullptr) {
                index_->range_search(option.n, option.vecs.data(), option.radius, &result);
            } else {
                bitmap_id_selector filter(option.bitmap);
                index_->condition_range_search(
                        option.n, option.vecs.data(), option.radius, &result, filter);
            }
            distances.resize((size_t) option.n);
            labels.resize((size_t) option.n);
            for (int64_t i = 0; i < option.n; i++) {
                for (size_t j = result.lims[i]; j < result.lims[i + 1]; j++) {
                    int pos = j;
                    if (result.labels[pos] > 0) {
                        distances[i].push_back(result.distances[pos]);
                        labels[i].push_back(result.labels[pos]);
                    }
                }
            }
        } catch (faiss::FaissException &e) {
            FLARE_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (std::exception &e) {
            FLARE_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (...) {
            FLARE_LOG(ERROR) << "unkown exception";
        }
    }

    void FaissIndex::search(
            search_options &option, std::vector<float> &distances, std::vector<int64_t> &labels) {
        if (nullptr == index_ || index_->ntotal <= 0) {
            return;
        }

        try {
            if (option.bitmap == nullptr) {
                if (option.nprobe == 0) {
                    index_->search(option.n, option.vecs.data(), option.k, distances.data(), labels.data());
                } else {
                    faiss::IVFSearchParameters params;
                    params.nprobe = option.nprobe;
                    params.max_codes = 0;
                    faiss::ivflib::search_with_parameters(index_, option.n, option.vecs.data(), option.k,
                                                          distances.data(), labels.data(), &params);
                }
            } else {
                bitmap_id_selector filter(option.bitmap);
              index_->condition_search(
                  option.n, option.vecs.data(), option.k, distances.data(), labels.data(), filter);
            }
        } catch (faiss::FaissException &e) {
            FLARE_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (std::exception &e) {
            FLARE_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (...) {
            FLARE_LOG(ERROR) << "unkown exception";
        }
    }

    class RemoveIds : public faiss::IDSelector {
    public:
        const std::set<uint64_t> &ids;

        explicit RemoveIds(const std::set<uint64_t> &v) : ids(v) {
        }

        bool is_member(idx_t id) const {
            return ids.find(id) != ids.end();
        }
    };

    void FaissIndex::remove(const std::set<uint64_t> &delete_ids) {
        if (nullptr == index_) {
            return;
        }

        std::vector<faiss::IDSelector::idx_t> array(delete_ids.begin(), delete_ids.end());
        faiss::IDSelectorArray ids(array.size(), array.data());
        try {
            index_->remove_ids(ids);
        } catch (faiss::FaissException &e) {
            FLARE_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (std::exception &e) {
            FLARE_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (...) {
            FLARE_LOG(ERROR) << "unkown exception";
        }
    }

    void FaissIndex::update(const std::vector<int64_t> &ids, std::vector<float> &vecs) {
        if (nullptr == index_) {
            return;
        }
        try {
            // reinterpret_cast<faiss::IndexIVFFlat*>(index_)->update_vectors(
            //    ids.size(), const_cast<int64_t*>(ids.data()), vecs.data());
            // just remove then add
            faiss::IDSelectorArray sel(ids.size(), ids.data());
            index_->remove_ids(sel);
            add_with_ids(ids, vecs);
        } catch (faiss::FaissException &e) {
            FLARE_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (std::exception &e) {
            FLARE_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (...) {
            FLARE_LOG(ERROR) << "unkown exception";
        }
    }

    void FaissIndex::clear() {
        if (nullptr != index_) {
            index_->reset();
        }
        delete index_;
        index_ = nullptr;
    }

    int FaissIndex::load(const std::string &file, flare::write_lock &index_wlock) {
        try {
            faiss::Index *tmp = faiss::read_index(file.c_str(), 0);
            if (!check_index(tmp)) {
                tmp->reset();
                delete tmp;
                return -1;
            }
            if (nullptr != tmp) {
                std::lock_guard<flare::write_lock> guard(index_wlock);
                clear();
                index_ = tmp;
                return 0;
            }
        } catch (faiss::FaissException &e) {
            FLARE_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (std::exception &e) {
            FLARE_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (...) {
            FLARE_LOG(ERROR) << "unkown exception";
        }
        return -1;
    }

    int FaissIndex::save(const std::string &file) {
        if (nullptr == index_) {
            return -1;
        }
        try {
            faiss::write_index(index_, file.c_str());
        } catch (faiss::FaissException &e) {
            FLARE_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (std::exception &e) {
            FLARE_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (...) {
            FLARE_LOG(ERROR) << "unkown exception";
        }
        return 0;
    }

    FaissIndex::FaissIndex() {
    }

    FaissIndex::~FaissIndex() {
        clear();
    }

    void FaissVirtualIndex::search(
            search_options &option, std::vector<float> &distances, std::vector<int64_t> &labels) {
        return;
    }

    FaissVirtualIndex::FaissVirtualIndex() {
    }

    FaissVirtualIndex::~FaissVirtualIndex() {
        clear();
    }

}  // namespace lambda
