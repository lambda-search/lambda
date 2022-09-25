
#include "lambda/ann_index.h"
#include <melon/log/logging.h>
#include "gflags/gflags.h"
#include "lambda/dbd/dbd_index.h"
#include "lambda/faiss/faiss_binary_index.h"
#include "lambda/faiss/faiss_index.h"
#include "lambda/hnsw/hnswlib_index.h"


namespace lambda {

    bool string_to_bool(const std::string &str, bool default_value) {
        if (str.empty()) {
            return default_value;
        }
        if ("true" == str) {
            return true;
        }
        return false;
    }

    bool is_dbd_index(const IndexConf &conf) {
        static const char kBdb[] = "bdb";
        const auto &options = conf.options();
        return options.count(kBdb) > 0 ? string_to_bool(options.at(kBdb), false) : false;
    }

    ann_index::ann_index() : max_elements_(DEFAULT_MAX_ELEMENTS) {
    }

    ann_index::~ann_index() {
    }

    // create index
    ann_index_ptr create_ann_index(const IndexShardConf &index_conf) {
        ann_index_ptr p = nullptr;
        if (!is_dbd_index(index_conf.conf())) {
            p = create_solo_index(index_conf);
        } else {
            p = create_dbd_index(index_conf);
        }
        return p;
    }

    ann_index_ptr create_solo_index(const IndexShardConf &index_conf) {
        ann_index_ptr p = nullptr;
        switch (index_conf.conf().engine()) {
            case ENGINE_FAISS_VECTOR: {
                p.reset(new FaissIndex);
                break;
            }
            case ENGINE_FAISS_BINARY: {
                p.reset(new FaissBinaryIndex);
                break;
            }
            case ENGINE_HNSWLIB: {
                p.reset(new hnsw_index);
                break;
            }
            case ENGINE_FAISS_VIRTUAL: {
                p.reset(new FaissVirtualIndex);
                break;
            }
            default: {
                MELON_LOG(ERROR) << "Unsupported engine type:" << index_conf.conf().engine();
                break;
            }
        }
        if (p != nullptr) {
            p->set_conf(index_conf);
        }
        return p;
    }

    ann_index_ptr create_dbd_index(const IndexShardConf &index_conf) {
        ann_index_ptr p = nullptr;
        auto fgp = create_solo_index(index_conf);
        if (fgp == nullptr) {
            return nullptr;
        }
        auto bgp = create_solo_index(index_conf);
        if (fgp == nullptr) {
            return nullptr;
        }
        p.reset(new dbd_index(fgp, bgp));
        p->set_conf(index_conf);
        return p;
    }

    // need model?
    bool need_model(int engine) {
        MELON_LOG(INFO) << "need model: " << engine;
        return engine == ENGINE_FAISS_VECTOR || engine == ENGINE_FAISS_BINARY;
    }

    // engine sort method
    bool is_order_less(int engine, int metric) {
        if (engine == ENGINE_FAISS_BINARY) {
            // faiss hamming
            return true;
        }
        // l2
        return metric == INDEX_METRIC_L2;
    }

}  // namespace lambda
