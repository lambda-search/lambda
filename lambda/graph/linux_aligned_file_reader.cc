/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "linux_aligned_file_reader.h"
#include <cassert>
#include <cstdio>
#include <iostream>
#include "melon/container/robin_map.h"
#include "utils.h"

#define MAX_EVENTS 1024

namespace lambda {
    namespace {
#ifdef MELON_PLATFORM_OSX

        void execute_io(IOContext ctx, int fd, std::vector<AlignedRead> &read_reqs,
                        uint64_t n_retries = 0) {
#ifdef DEBUG
            for (auto &req : read_reqs) {
          assert(IS_ALIGNED(req.len, 512));
          MELON_LOG(DEBUG)<< "request:"<<req.offset<<":"<<req.len;
          assert(IS_ALIGNED(req.offset, 512));
          assert(IS_ALIGNED(req.buf, 512));
          // assert(malloc_usable_size(req.buf) >= req.len);
        }
#endif
            for (auto &req : read_reqs) {
                // Since this function is called from multiple threads
                // pread is better than read as the file pointer
                // remains unchanged
                pread(fd, req.buf, req.len, req.offset);
            }
        }

#else
        typedef struct io_event io_event_t;
        typedef struct iocb iocb_t;

        void execute_io(io_context_t ctx, int fd, std::vector<AlignedRead> &read_reqs,
                        uint64_t n_retries = 0) {
#ifdef DEBUG
            for (auto &req : read_reqs) {
              assert(IS_ALIGNED(req.len, 512));
              MELON_LOG(DEBUG) << "request:"<<req.offset<<":"<<req.len;
              assert(IS_ALIGNED(req.offset, 512));
              assert(IS_ALIGNED(req.buf, 512));
              // assert(malloc_usable_size(req.buf) >= req.len);
            }
#endif

            // break-up requests into chunks of size MAX_EVENTS each
            uint64_t n_iters = ROUND_UP(read_reqs.size(), MAX_EVENTS) / MAX_EVENTS;
            for (uint64_t iter = 0; iter < n_iters; iter++) {
                uint64_t n_ops =
                        std::min((uint64_t) read_reqs.size() - (iter * MAX_EVENTS),
                                 (uint64_t) MAX_EVENTS);
                std::vector<iocb_t *> cbs(n_ops, nullptr);
                std::vector<io_event_t> evts(n_ops);
                std::vector<struct iocb> cb(n_ops);
                for (uint64_t j = 0; j < n_ops; j++) {
                    io_prep_pread(cb.data() + j, fd, read_reqs[j + iter * MAX_EVENTS].buf,
                                  read_reqs[j + iter * MAX_EVENTS].len,
                                  read_reqs[j + iter * MAX_EVENTS].offset);
                }

                // initialize `cbs` using `cb` array
                //

                for (uint64_t i = 0; i < n_ops; i++) {
                    cbs[i] = cb.data() + i;
                }

                uint64_t n_tries = 0;
                while (n_tries <= n_retries) {
                    // issue reads
                    int64_t ret = io_submit(ctx, (int64_t) n_ops, cbs.data());
                    // if requests didn't get accepted
                    if (ret != (int64_t) n_ops) {
                        std::cerr << "io_submit() failed; returned " << ret
                                  << ", expected=" << n_ops << ", ernno=" << errno << "="
                                  << ::strerror(-ret) << ", try #" << n_tries + 1;
                        std::cout << "ctx: " << ctx << "\n";
                        exit(-1);
                    } else {
                        // wait on io_getevents
                        ret = io_getevents(ctx, (int64_t) n_ops, (int64_t) n_ops, evts.data(),
                                           nullptr);
                        // if requests didn't complete
                        if (ret != (int64_t) n_ops) {
                            std::cerr << "io_getevents() failed; returned " << ret
                                      << ", expected=" << n_ops << ", ernno=" << errno << "="
                                      << ::strerror(-ret) << ", try #" << n_tries + 1;
                            exit(-1);
                        } else {
                            break;
                        }
                    }
                }
                // disabled since req.buf could be an offset into another buf
                /*
                for (auto &req : read_reqs) {
                  // corruption check
                  assert(malloc_usable_size(req.buf) >= req.len);
                }
                */
            }
        }
#endif
    }  // namespace

    LinuxAlignedFileReader::LinuxAlignedFileReader() {
        this->file_desc = -1;
    }

    LinuxAlignedFileReader::~LinuxAlignedFileReader() {
        int64_t ret;
        // check to make sure file_desc is closed
        ret = ::fcntl(this->file_desc, F_GETFD);
        if (ret == -1) {
            if (errno != EBADF) {
                MELON_LOG(ERROR) << "close() not called";
                // close file desc
                ret = ::close(this->file_desc);
                // error checks
                if (ret == -1) {
                    MELON_LOG(ERROR) << "close() failed; returned " << ret << ", errno=" << errno
                                     << ":" << ::strerror(errno);
                }
            }
        }
    }

    IOContext &LinuxAlignedFileReader::get_ctx() {
        std::unique_lock<std::mutex> lk(ctx_mut);
        // perform checks only in DEBUG mode
        if (ctx_map.find(std::this_thread::get_id()) == ctx_map.end()) {
            MELON_LOG(ERROR) << "bad thread access; returning -1 as io_context_t";
            return this->bad_ctx;
        } else {
            return ctx_map[std::this_thread::get_id()];
        }
    }

    void LinuxAlignedFileReader::register_thread() {
        auto my_id = std::this_thread::get_id();
        std::unique_lock<std::mutex> lk(ctx_mut);
        if (ctx_map.find(my_id) != ctx_map.end()) {
            MELON_LOG(ERROR) << "multiple calls to register_thread from the same thread";
            return;
        }
        IOContext ctx;
#ifdef MELON_PLATFORM_OSX
        ctx_map[my_id] = ctx;
#else
        int ret = io_setup(MAX_EVENTS, &ctx);
        if (ret != 0) {
            lk.unlock();
            assert(errno != EAGAIN);
            assert(errno != ENOMEM);
            MELON_LOG(ERROR) << "io_setup() failed; returned " << ret << ", errno=" << errno
                      << ":" << ::strerror(errno);
        } else {
            MELON_LOG(INFO) << "allocating ctx: " << ctx << " to thread-id:" << my_id;
            ctx_map[my_id] = ctx;
        }
#endif
        lk.unlock();
    }

    void LinuxAlignedFileReader::deregister_thread() {
        auto my_id = std::this_thread::get_id();
        std::unique_lock<std::mutex> lk(ctx_mut);
        assert(ctx_map.find(my_id) != ctx_map.end());

        lk.unlock();
#ifndef MELON_PLATFORM_OSX
        io_context_t ctx = this->get_ctx();
        io_destroy(ctx);
#endif
        //  assert(ret == 0);
        lk.lock();
        ctx_map.erase(my_id);
        MELON_LOG(ERROR) << "returned ctx from thread-id:" << my_id;
        lk.unlock();
    }

    void LinuxAlignedFileReader::deregister_all_threads() {
        std::unique_lock<std::mutex> lk(ctx_mut);
        for (auto x = ctx_map.begin(); x != ctx_map.end(); x++) {
#ifndef MELON_PLATFORM_OSX
            IOContext ctx = x.value();
            io_destroy(ctx);
#endif
            //  assert(ret == 0);
            //  lk.lock();
            //  ctx_map.erase(my_id);
            //  std::cerr << "returned ctx from thread-id:" << my_id;
        }
        ctx_map.clear();
        //  lk.unlock();
    }

    void LinuxAlignedFileReader::open(const std::string &fname) {
#ifdef MELON_PLATFORM_OSX
        int flags = O_RDONLY;
#else
        int flags = O_DIRECT | O_RDONLY | O_LARGEFILE;
#endif
        this->file_desc = ::open(fname.c_str(), flags);
        // error checks
        assert(this->file_desc != -1);
        MELON_LOG(INFO) << "Opened file : " << fname;
    }

    void LinuxAlignedFileReader::close() {
        //  int64_t ret;

        // check to make sure file_desc is closed
        ::fcntl(this->file_desc, F_GETFD);
        //  assert(ret != -1);

        ::close(this->file_desc);
        //  assert(ret != -1);
    }

    void LinuxAlignedFileReader::read(std::vector<AlignedRead> &read_reqs,
                                      IOContext &ctx, bool async) {
        if (async == true) {
            MELON_LOG(INFO) << "Async currently not supported in linux.";
        }
        assert(this->file_desc != -1);
        //#pragma omp critical
        //	std::cout << "thread: " << std::this_thread::get_id() << ", crtx: " <<
        // ctx
        //<< "\n";
        execute_io(ctx, this->file_desc, read_reqs);
    }
}