/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#pragma once

#include "aligned_file_reader.h"

namespace lambda {
    class LinuxAlignedFileReader : public AlignedFileReader {
    private:
        uint64_t file_sz;
        int file_desc;
        IOContext bad_ctx;

    public:
        LinuxAlignedFileReader();

        ~LinuxAlignedFileReader();

        IOContext &get_ctx();

        // register thread-id for a context
        void register_thread();

        // de-register thread-id for a context
        void deregister_thread();

        void deregister_all_threads();

        // Open & close ops
        // Blocking calls
        void open(const std::string &fname);

        void close();

        // process batch of aligned requests in parallel
        // NOTE :: blocking call
        void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
                  bool async = false);
    };

}