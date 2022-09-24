/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include <mutex>

namespace lambda {

    using non_recursive_mutex = std::mutex;
    using LockGuard = std::lock_guard<non_recursive_mutex>;
}  // namespace lambda
