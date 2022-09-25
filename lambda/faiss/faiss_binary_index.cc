
#include "lambda/faiss/faiss_binary_index.h"
#include <melon/log/logging.h>
#include <melon/files/filesystem.h>
#include "faiss/AutoTune.h"
#include "faiss/IVFlib.h"
#include "faiss/IndexBinaryFlat.h"
#include "faiss/IndexBinaryIVF.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexIVFPQ.h"

namespace lambda {

    int FaissBinaryIndex::init(const std::string &model) {
        const auto &index_name = index_conf_.index();
        const auto &conf = index_conf_.conf();
        std::string model_path = model;
        std::error_code ec;
        if (!melon::exists(model_path)) {
            MELON_LOG(ERROR) << "model file not exist: " << model_path;
            return -1;
        }

        faiss::IndexBinary *tmp = nullptr;
        try {
            tmp = faiss::read_index_binary(model_path.c_str(), 0);
        } catch (faiss::FaissException &e) {
            MELON_LOG(ERROR) << "Faiss exception: " << e.what();
            return -1;
        } catch (std::exception &e) {
            MELON_LOG(ERROR) << "Faiss exception: " << e.what();
            return -1;
        } catch (...) {
            MELON_LOG(ERROR) << "unkown exception";
            return -1;
        }
        if (nullptr == tmp) {
            MELON_LOG(ERROR) << "Fail to read index from model " << model_path << " with engine "
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

    bool FaissBinaryIndex::check_index(faiss::IndexBinary *tmp) {
        faiss::IndexBinaryIVF *ivf = dynamic_cast<faiss::IndexBinaryIVF *>(tmp);
        if (nullptr == ivf) {
            return true;
        }
        const auto &conf = index_conf_.conf();
        if (conf.dimension() * 8 != ivf->d) {
            MELON_LOG(ERROR) << "index " << index_conf_.index() << " check failed, expect dimension "
                       << conf.dimension() << " but got " << ivf->d;
            return false;
        }
        if (conf.nprobe() > 0) {
            ivf->nprobe = conf.nprobe();
        }
        if (conf.direct_map()) {
            support_update_ = true;
            ivf->set_direct_map_type(faiss::DirectMap::Hashtable);
        }
        return true;
    }

    size_t FaissBinaryIndex::size() {
        if (nullptr == index_) {
            return 0;
        }
        return (size_t) (index_->ntotal);
    }

    bool FaissBinaryIndex::support_update() {
        return support_update_;
    }

    bool FaissBinaryIndex::support_delete() {
        return support_update_;
    }

    void FaissBinaryIndex::add_with_ids(const std::vector<int64_t> &ids, std::vector<float> &vecs) {
        if (nullptr == index_) {
            MELON_LOG(ERROR) << "index " << index_conf_.index() << ", shard " << index_conf_.shard_idx()
                       << " is not initialized";
            return;
        }

        std::vector<uint8_t> tmp;
        tmp.reserve(vecs.size());
        for (size_t i = 0; i < vecs.size(); ++i) {
            tmp[i] = static_cast<uint8_t>(vecs[i]);
        }
        try {
            index_->add_with_ids(ids.size(), tmp.data(), ids.data());
        } catch (faiss::FaissException &e) {
            MELON_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (std::exception &e) {
            MELON_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (...) {
            MELON_LOG(ERROR) << "unkown exception";
        }
    }

    void FaissBinaryIndex::range_search(search_options &option,
                                        std::vector<std::vector<float>> &distances,
                                        std::vector<std::vector<int64_t>> &labels) {
        if (nullptr == index_ || index_->ntotal <= 0) {
            return;
        }

        std::vector<uint8_t> tmp_vecs;
        tmp_vecs.reserve(option.vecs.size());
        for (size_t i = 0; i < option.vecs.size(); ++i) {
            tmp_vecs[i] = static_cast<uint8_t>(option.vecs[i]);
        }

        faiss::RangeSearchResult result(option.n);
        try {
            index_->range_search(option.n, tmp_vecs.data(), static_cast<int>(option.radius), &result);
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
            MELON_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (std::exception &e) {
            MELON_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (...) {
            MELON_LOG(ERROR) << "unkown exception";
        }
    }

    void FaissBinaryIndex::search(
            search_options &option, std::vector<float> &distances, std::vector<int64_t> &labels) {
        if (nullptr == index_ || index_->ntotal <= 0) {
            return;
        }

        std::vector<uint8_t> tmp_vecs(option.vecs.size());
        for (size_t i = 0; i < option.vecs.size(); ++i) {
            tmp_vecs[i] = static_cast<uint8_t>(option.vecs[i]);
        }

        std::vector<int32_t> tmp_distances(distances.size());
        try {
            index_->search(option.n, tmp_vecs.data(), option.k, tmp_distances.data(), labels.data());
            for (size_t i = 0; i < tmp_distances.size(); ++i) {
                distances[i] = static_cast<float>(tmp_distances[i]);
            }
        } catch (faiss::FaissException &e) {
            MELON_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (std::exception &e) {
            MELON_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (...) {
            MELON_LOG(ERROR) << "unkown exception";
        }
    }

    void FaissBinaryIndex::remove(const std::set<uint64_t> &delete_ids) {
        if (nullptr == index_) {
            return;
        }
        std::vector<faiss::IDSelector::idx_t> array(delete_ids.begin(), delete_ids.end());
        faiss::IDSelectorArray ids(array.size(), array.data());
        try {
            index_->remove_ids(ids);
        } catch (faiss::FaissException &e) {
            MELON_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (std::exception &e) {
            MELON_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (...) {
            MELON_LOG(ERROR) << "unkown exception";
        }
    }

    void FaissBinaryIndex::update(const std::vector<int64_t> &ids, std::vector<float> &vecs) {
        if (nullptr == index_ || ids.empty()) {
            return;
        }
        try {
            faiss::IDSelectorArray sel(ids.size(), ids.data());
            index_->remove_ids(sel);
            add_with_ids(ids, vecs);
        } catch (faiss::FaissException &e) {
            MELON_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (std::exception &e) {
            MELON_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (...) {
            MELON_LOG(ERROR) << "unkown exception";
        }
    }

    void FaissBinaryIndex::clear() {
        if (nullptr != index_) {
            index_->reset();
        }
        delete index_;
        index_ = nullptr;
    }

    int FaissBinaryIndex::load(const std::string &fname, melon::write_lock &index_wlock) {
        try {
            faiss::IndexBinary *tmp = faiss::read_index_binary(fname.c_str(), 0);
            if (!check_index(tmp)) {
                tmp->reset();
                delete tmp;
                return -1;
            }
            if (nullptr != tmp) {
                std::lock_guard<melon::write_lock> guard(index_wlock);
                clear();
                index_ = tmp;
                return 0;
            }
        } catch (faiss::FaissException &e) {
            MELON_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (std::exception &e) {
            MELON_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (...) {
            MELON_LOG(ERROR) << "unkown exception";
        }
        return -1;
    }

    int FaissBinaryIndex::save(const std::string &fname) {
        if (nullptr == index_) {
            return -1;
        }
        try {
            faiss::write_index_binary(index_, fname.c_str());
        } catch (faiss::FaissException &e) {
            MELON_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (std::exception &e) {
            MELON_LOG(ERROR) << "Faiss exception: " << e.what();
        } catch (...) {
            MELON_LOG(ERROR) << "unkown exception";
        }
        return 0;
    }

    FaissBinaryIndex::FaissBinaryIndex() {
    }

    FaissBinaryIndex::~FaissBinaryIndex() {
        clear();
    }

}  // namespace lambda
