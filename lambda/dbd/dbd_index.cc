
#include "lambda/dbd/dbd_index.h"

#include "gflags/gflags.h"

#define GET_SCOPEDPTR(r_ptr, args...)                               \
  DBDScopedPtr r_ptr;                                               \
  do {                                                              \
    if (ann_index_.Read(&r_ptr) != 0) {                             \
      MELON_LOG_EVERY_N(ERROR, 10) << "dbd read failed";  \
      return args;                                                  \
    }                                                               \
  } while (false)

namespace lambda {

    dbd_index::dbd_index(ann_index_ptr fg_index, ann_index_ptr bg_index) {
        const auto assign_fg_fn = [&](ann_index_ptr &index) -> bool {
            index = fg_index;
            return true;
        };
        ann_index_.Modify(assign_fg_fn);

        const auto assign_bg_fn = [&](ann_index_ptr &index) -> bool {
            index = bg_index;
            return false;
        };
        ann_index_.Modify(assign_bg_fn);
    }

    int dbd_index::init(const std::string &model) {
        const auto init_fn = [&](ann_index_ptr &index) -> bool {
            return index->init(model) == 0;
        };
        bool status = ann_index_.Modify(init_fn);
        if (!status) {
            MELON_LOG_EVERY_N(ERROR, 10) << "Fail to init background:";
            return !status;
        }
        MELON_LOG(INFO) << "Success to init dbd_index";
        return 0;
    }

    size_t dbd_index::size() {
        GET_SCOPEDPTR(reader, 0);
        return (*reader)->size();
    }

    bool dbd_index::support_update() {
        GET_SCOPEDPTR(reader, false);
        return (*reader)->support_update();
    }

    bool dbd_index::support_delete() {
        GET_SCOPEDPTR(reader, false);
        return (*reader)->support_delete();
    }

    bool dbd_index::need_model() {
        GET_SCOPEDPTR(reader, false);
        return (*reader)->need_model();
    }

    void dbd_index::add_with_ids(const std::vector<int64_t> &ids, std::vector<float> &vecs) {
        const auto add_with_ids_fn = [&](ann_index_ptr &index) -> bool {
            index->add_with_ids(ids, vecs);
            return true;
        };
        ann_index_.Modify(add_with_ids_fn);
    }

    void dbd_index::search(search_options &option,
                          std::vector<float> &distances, std::vector<int64_t> &labels) {
        GET_SCOPEDPTR(reader);
        (*reader)->search(option, distances, labels);
    }


    void dbd_index::remove(const std::set<uint64_t> &delete_ids) {
        const auto remove_fn = [&](ann_index_ptr &index) -> bool {
            index->remove(delete_ids);
            return true;
        };
        ann_index_.Modify(remove_fn);
    }

    void dbd_index::update(const std::vector<int64_t> &ids, std::vector<float> &vecs) {
        const auto update_fn = [&](ann_index_ptr &index) -> bool {
            index->update(ids, vecs);
            return true;
        };
        ann_index_.Modify(update_fn);
    }


    void dbd_index::clear() {
        const auto clear_fn = [&](ann_index_ptr &index) -> bool {
            index->clear();
            return true;
        };
        ann_index_.Modify(clear_fn);
    }

    int dbd_index::load(const std::string &file, melon::write_lock &index_wlock) {
        const auto load_fn = [&](ann_index_ptr &index) -> bool {
            return index->load(file, index_wlock) == 0;
        };
        return !ann_index_.Modify(load_fn);
    }

    int dbd_index::save(const std::string &file) {
        GET_SCOPEDPTR(reader, -1);
        return (*reader)->save(file);
    }

    int dbd_index::build_batch_size() {
        GET_SCOPEDPTR(reader, -1);
        return (*reader)->build_batch_size();
    }

    void dbd_index::range_search(search_options &option, std::vector<std::vector<float>> &distances,
                                std::vector<std::vector<int64_t>> &labels) {
        GET_SCOPEDPTR(reader);
        (*reader)->range_search(option, distances, labels);
    }

    void dbd_index::shrink_to_fit() {
        const auto shrink_to_fit_fn = [&](ann_index_ptr &index) -> bool {
            index->shrink_to_fit();
            return true;
        };
        ann_index_.Modify(shrink_to_fit_fn);
    }

}  // namespace lambda
