//
// Created by liyinbin on 2022/9/24.
//

#ifndef LAMBDA_COMMON_MEMORY_H_
#define LAMBDA_COMMON_MEMORY_H_

#include <cstddef>
#include <immintrin.h>
#include "lambda/common/math_utils.h"

namespace lambda {

    // NOTE :: good efficiency when total_vec_size is integral multiple of 64
    inline void prefetch_vector(const char *vec, size_t vecsize) {
        size_t max_prefetch_size = (vecsize / 64) * 64;
        for (size_t d = 0; d < max_prefetch_size; d += 64)
            _mm_prefetch((const char *) vec + d, _MM_HINT_T0);
    }


    // NOTE :: good efficiency when total_vec_size is integral multiple of 64
    inline void prefetch_vector_l2(const char *vec, size_t vecsize) {
        size_t max_prefetch_size = (vecsize / 64) * 64;
        for (size_t d = 0; d < max_prefetch_size; d += 64)
            _mm_prefetch((const char *) vec + d, _MM_HINT_T1);
    }

    inline void alloc_aligned(void **ptr, size_t size, size_t align) {
        *ptr = nullptr;
        assert(IS_ALIGNED(size, align));
        *ptr = ::aligned_alloc(align, size);
        assert(*ptr != nullptr);
    }



    inline void realloc_aligned(void **ptr, size_t size, size_t align) {
        assert(IS_ALIGNED(size, align));
        MELON_LOG(ERROR) << "No aligned realloc on GCC. Must malloc and mem_align, "
                            "left it out for now.";
        assert(*ptr != nullptr);
    }

    inline void aligned_free(void *ptr) {
        // Gopal. Must have a check here if the pointer was actually allocated by
        // _alloc_aligned
        if (ptr == nullptr) {
            return;
        }
        free(ptr);
    }

}  // namespace lambda

#endif  // LAMBDA_COMMON_MEMORY_H_
