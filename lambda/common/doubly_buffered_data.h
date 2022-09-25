
#pragma once

#include <memory>
#include <vector>

#include "melon/fiber/fiber.h"
#include "melon/base/scoped_lock.h"
#include "melon/log/logging.h"
#include "melon/base/errno.h"
#include "melon/thread/rw_lock.h"

namespace lambda {

    // This data structure makes Read() almost lock-free by making Modify()
    // *much* slower. It's very suitable for implementing LoadBalancers which
    // have a lot of concurrent read-only ops from many threads and occasional
    // modifications of data. As a side effect, this data structure can store
    // a thread-local data for user.
    //
    // Read(): begin with a thread-local mutex locked then read the foreground
    // instance which will not be changed before the mutex is unlocked. Since the
    // mutex is only locked by Modify() with an empty critical section, the
    // function is almost lock-free.
    //
    // Modify(): Modify background instance which is not used by any Read(), flip
    // foreground and background, lock thread-local mutexes one by one to make
    // sure all existing Read() finish and later Read() see new foreground,
    // then modify background(foreground before flip) again.

    template<typename T>
    class doubly_buffered_data {
    public:
        class ScopedPtr {
            friend class doubly_buffered_data;

        public:
            ScopedPtr() : data_(NULL) {}

            ~ScopedPtr() {}

            const T *get() const { return data_; }

            const T &operator*() const { return *data_; }

            const T *operator->() const { return data_; }

        private:
            const T *data_;
            std::unique_lock<melon::read_lock> rl_;
            MELON_DISALLOW_COPY_AND_ASSIGN(ScopedPtr);
        };

        doubly_buffered_data();

        ~doubly_buffered_data();

        // Put foreground instance into ptr. The instance will not be changed until
        // ptr is destructed.
        // This function is not blocked by Read() and Modify() in other threads.
        // Returns 0 on success, -1 otherwise.
        int Read(ScopedPtr *ptr);

        // Modify background and foreground instances. fn(T&, ...) will be called
        // twice. Modify() from different threads are exclusive from each other.
        // NOTE: Call same series of fn to different equivalent instances should
        // result in equivalent instances, otherwise foreground and background
        // instance will be inconsistent.
        template<typename Fn>
        size_t Modify(Fn &fn);

        template<typename Fn, typename Arg1>
        size_t Modify(Fn &fn, const Arg1 &);

        template<typename Fn, typename Arg1, typename Arg2>
        size_t Modify(Fn &fn, const Arg1 &, const Arg2 &);

        // fn(T& background, const T& foreground, ...) will be called to background
        // and foreground instances respectively.
        template<typename Fn>
        size_t ModifyWithForeground(Fn &fn);

        template<typename Fn, typename Arg1>
        size_t ModifyWithForeground(Fn &fn, const Arg1 &);

        template<typename Fn, typename Arg1, typename Arg2>
        size_t ModifyWithForeground(Fn &fn, const Arg1 &, const Arg2 &);

    private:
        template<typename Fn>
        struct WithFG0 {
            WithFG0(Fn &f, T *d) : fn(f), data(d) {}

            size_t operator()(T &bg) {
                return fn(bg, (const T &) data[&bg == data]);
            }

        private:
            Fn &fn;
            T *data;
        };

        template<typename Fn, typename Arg1>
        struct WithFG1 {
            WithFG1(Fn &f, T *d, const Arg1 &a1) : fn(f), data(d), arg1(a1) {}

            size_t operator()(T &bg) {
                return fn(bg, (const T &) data[&bg == data], arg1);
            }

        private:
            Fn &fn;
            T *data;
            const Arg1 &arg1;
        };

        template<typename Fn, typename Arg1, typename Arg2>
        struct WithFG2 {
            WithFG2(Fn &f, T *d, const Arg1 &a1, const Arg2 &a2) : fn(f), data(d), arg1(a1), arg2(a2) {}

            size_t operator()(T &bg) {
                return fn(bg, (const T &) data[&bg == data], arg1, arg2);
            }

        private:
            Fn &fn;
            T *data;
            const Arg1 &arg1;
            const Arg2 &arg2;
        };

        template<typename Fn, typename Arg1>
        struct Closure1 {
            Closure1(Fn &f, const Arg1 &a1) : fn(f), arg1(a1) {}

            size_t operator()(T &bg) { return fn(bg, arg1); }

        private:
            Fn &fn;
            const Arg1 &arg1;
        };

        template<typename Fn, typename Arg1, typename Arg2>
        struct Closure2 {
            Closure2(Fn &f, const Arg1 &a1, const Arg2 &a2) : fn(f), arg1(a1), arg2(a2) {}

            size_t operator()(T &bg) { return fn(bg, arg1, arg2); }

        private:
            Fn &fn;
            const Arg1 &arg1;
            const Arg2 &arg2;
        };

        const T *UnsafeRead(int index) const {
            return data_ + index;
        }

        std::unique_lock<melon::read_lock> GetFGReadLock(int index) const {
            return std::unique_lock<melon::read_lock>(*rlock_[index]);
        }

        // Foreground and background void.
        T data_[2];

        // Index of foreground instance.
        std::atomic<int> index_;

        // Foreground and background bthread_rwlock
        melon::rw_lock rwlock_[2];
        std::shared_ptr<melon::read_lock> rlock_[2];
        std::shared_ptr<melon::write_lock> wlock_[2];

        // Sequence modifications.
        fiber_mutex_t modify_mutex_;
    };

    template<typename T>
    doubly_buffered_data<T>::doubly_buffered_data()
            : index_(0) {
        fiber_mutex_init(&modify_mutex_, nullptr);
        rlock_[0] = std::make_shared<melon::read_lock>(rwlock_[0]);
        rlock_[1] = std::make_shared<melon::read_lock>(rwlock_[1]);
        wlock_[0] = std::make_shared<melon::write_lock>(rwlock_[0]);
        wlock_[1] = std::make_shared<melon::write_lock>(rwlock_[1]);

        // Initialize _data for some POD types. This is essential for pointer
        // types because they should be Read() as NULL before any Modify().
        if (std::is_integral<T>::value || std::is_floating_point<T>::value ||
            std::is_pointer<T>::value || std::is_member_function_pointer<T>::value) {
            data_[0] = T();
            data_[1] = T();
        }
    }

    template<typename T>
    doubly_buffered_data<T>::~doubly_buffered_data() {
        fiber_mutex_destroy(&modify_mutex_);
    }

    template<typename T>
    int doubly_buffered_data<T>::Read(
            typename doubly_buffered_data<T>::ScopedPtr *ptr) {
        int fg_index = index_.load(std::memory_order_acquire);
        ptr->rl_ = GetFGReadLock(fg_index);
        ptr->data_ = UnsafeRead(fg_index);
        return 0;
    }

    template<typename T>
    template<typename Fn>
    size_t doubly_buffered_data<T>::Modify(Fn &fn) {
        // _modify_mutex sequences modifications. Using a separate mutex rather
        // than _wrappers_mutex is to avoid blocking threads calling
        // AddWrapper() or RemoveWrapper() too long. Most of the time, modifications
        // are done by one thread, contention should be negligible.
        MELON_SCOPED_LOCK(modify_mutex_);
        int bg_index = !index_.load(std::memory_order_relaxed);
        // background instance is not accessed by other threads, being safe to
        // modify.
        const size_t ret = fn(data_[bg_index]);
        if (!ret) {
            return 0;
        }

        // Publish, flip background and foreground.
        // The release fence matches with the acquire fence in UnsafeRead() to
        // make readers which just begin to read the new foreground instance see
        // all changes made in fn.
        index_.store(bg_index, std::memory_order_release);
        bg_index = !bg_index;

        // Wait until all threads finishes current reading. When they begin next
        // read, they should see updated _index.
        std::unique_lock<melon::write_lock> bg_lock(*wlock_[bg_index]);

        const size_t ret2 = fn(data_[bg_index]);
        MELON_CHECK_EQ(ret2, ret) << "index=" << index_.load(std::memory_order_relaxed);
        return ret2;
    }

    template<typename T>
    template<typename Fn, typename Arg1>
    size_t doubly_buffered_data<T>::Modify(Fn &fn, const Arg1 &arg1) {
        Closure1<Fn, Arg1> c(fn, arg1);
        return Modify(c);
    }

    template<typename T>
    template<typename Fn, typename Arg1, typename Arg2>
    size_t doubly_buffered_data<T>::Modify(
            Fn &fn, const Arg1 &arg1, const Arg2 &arg2) {
        Closure2<Fn, Arg1, Arg2> c(fn, arg1, arg2);
        return Modify(c);
    }

    template<typename T>
    template<typename Fn>
    size_t doubly_buffered_data<T>::ModifyWithForeground(Fn &fn) {
        WithFG0<Fn> c(fn, data_);
        return Modify(c);
    }

    template<typename T>
    template<typename Fn, typename Arg1>
    size_t doubly_buffered_data<T>::ModifyWithForeground(Fn &fn, const Arg1 &arg1) {
        WithFG1<Fn, Arg1> c(fn, data_, arg1);
        return Modify(c);
    }

    template<typename T>
    template<typename Fn, typename Arg1, typename Arg2>
    size_t doubly_buffered_data<T>::ModifyWithForeground(
            Fn &fn, const Arg1 &arg1, const Arg2 &arg2) {
        WithFG2<Fn, Arg1, Arg2> c(fn, data_, arg1, arg2);
        return Modify(c);
    }

}  // namespace karabor
