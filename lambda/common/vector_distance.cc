// TODO
// CHECK COSINE ON LINUX

#include "flare/base/profile.h"
#include <immintrin.h>
#include <iostream>
#include "lambda/common/vector_distance.h"
#include "flare/log/logging.h"

namespace lambda {

    // Cosine similarity.
    float DistanceCosineInt8::compare(const int8_t *a, const int8_t *b,
                                      uint32_t length) const {
        int magA = 0, magB = 0, scalarProduct = 0;
        for (uint32_t i = 0; i < length; i++) {
            magA += ((int32_t) a[i]) * ((int32_t) a[i]);
            magB += ((int32_t) b[i]) * ((int32_t) b[i]);
            scalarProduct += ((int32_t) a[i]) * ((int32_t) b[i]);
        }
        // similarity == 1-cosine distance
        return 1.0f - (float) (scalarProduct / (sqrt(magA) * sqrt(magB)));
    }

    float DistanceCosineFloat::compare(const float *a, const float *b,
                                       uint32_t length) const {
        float magA = 0, magB = 0, scalarProduct = 0;
        for (uint32_t i = 0; i < length; i++) {
            magA += (a[i]) * (a[i]);
            magB += (b[i]) * (b[i]);
            scalarProduct += (a[i]) * (b[i]);
        }
        // similarity == 1-cosine distance
        return 1.0f - (scalarProduct / (sqrt(magA) * sqrt(magB)));
    }

    float SlowDistanceCosineUInt8::compare(const uint8_t *a, const uint8_t *b,
                                           uint32_t length) const {
        int magA = 0, magB = 0, scalarProduct = 0;
        for (uint32_t i = 0; i < length; i++) {
            magA += ((uint32_t) a[i]) * ((uint32_t) a[i]);
            magB += ((uint32_t) b[i]) * ((uint32_t) b[i]);
            scalarProduct += ((uint32_t) a[i]) * ((uint32_t) b[i]);
        }
        // similarity == 1-cosine distance
        return 1.0f - (float) (scalarProduct / (sqrt(magA) * sqrt(magB)));
    }

    // L2 distance functions.
    float DistanceL2Int8::compare(const int8_t *a, const int8_t *b,
                                  uint32_t size) const {
        int32_t result = 0;

#pragma omp simd reduction(+ : result) aligned(a, b : 8)
        for (int32_t i = 0; i < (int32_t) size; i++) {
            result += ((int32_t) ((int16_t) a[i] - (int16_t) b[i])) *
                      ((int32_t) ((int16_t) a[i] - (int16_t) b[i]));
        }
        return (float) result;
    }

    float DistanceL2UInt8::compare(const uint8_t *a, const uint8_t *b,
                                   uint32_t size) const {
        uint32_t result = 0;
#pragma omp simd reduction(+ : result) aligned(a, b : 8)
        for (int32_t i = 0; i < (int32_t) size; i++) {
            result += ((int32_t) ((int16_t) a[i] - (int16_t) b[i])) *
                      ((int32_t) ((int16_t) a[i] - (int16_t) b[i]));
        }
        return (float) result;
    }

    float DistanceL2Float::compare(const float *a, const float *b,
                                   uint32_t size) const {
        a = (const float *) __builtin_assume_aligned(a, 32);
        b = (const float *) __builtin_assume_aligned(b, 32);

        float result = 0;
#ifdef USE_AVX2
        // assume size is divisible by 8
        uint16_t niters = (uint16_t) (size / 8);
        __m256 sum = _mm256_setzero_ps();
        for (uint16_t j = 0; j < niters; j++) {
            // scope is a[8j:8j+7], b[8j:8j+7]
            // load a_vec
            if (j < (niters - 1)) {
                _mm_prefetch((char *) (a + 8 * (j + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (b + 8 * (j + 1)), _MM_HINT_T0);
            }
            __m256 a_vec = _mm256_load_ps(a + 8 * j);
            // load b_vec
            __m256 b_vec = _mm256_load_ps(b + 8 * j);
            // a_vec - b_vec
            __m256 tmp_vec = _mm256_sub_ps(a_vec, b_vec);

            sum = _mm256_fmadd_ps(tmp_vec, tmp_vec, sum);
        }

        // horizontal add sum
        result = _mm256_reduce_add_ps(sum);
#else
#pragma omp simd reduction(+ : result) aligned(a, b : 32)
        for (int32_t i = 0; i < (int32_t) size; i++) {
            result += (a[i] - b[i]) * (a[i] - b[i]);
        }
#endif
        return result;
    }

    float SlowDistanceL2Float::compare(const float *a, const float *b,
                                       uint32_t length) const {
        float result = 0.0f;
        for (uint32_t i = 0; i < length; i++) {
            result += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return result;
    }

    float AVXDistanceL2Int8::compare(const int8_t *, const int8_t *,
                                     uint32_t) const {
        return 0;
    }

    float AVXDistanceL2Float::compare(const float *, const float *,
                                      uint32_t) const {
        return 0;
    }


    template<typename T>
    float DistanceInnerProduct<T>::inner_product(const T *a, const T *b,
                                                 unsigned size) const {
        FLARE_CHECK(std::is_floating_point<T>::value) << "ERROR: Inner Product only defined for float currently.";
        float result = 0;

#ifdef __GNUC__
#ifdef USE_AVX2
#define AVX_DOT(addr1, addr2, dest, tmp1, tmp2) \
  tmp1 = _mm256_loadu_ps(addr1);                \
  tmp2 = _mm256_loadu_ps(addr2);                \
  tmp1 = _mm256_mul_ps(tmp1, tmp2);             \
  dest = _mm256_add_ps(dest, tmp1);

        __m256 sum;
        __m256 l0, l1;
        __m256 r0, r1;
        unsigned D = (size + 7) & ~7U;
        unsigned DR = D % 16;
        unsigned DD = D - DR;
        const float *l = (float *) a;
        const float *r = (float *) b;
        const float *e_l = l + DD;
        const float *e_r = r + DD;
        float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};

        sum = _mm256_loadu_ps(unpack);
        if (DR) {
            AVX_DOT(e_l, e_r, sum, l0, r0);
        }

        for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
            AVX_DOT(l, r, sum, l0, r0);
            AVX_DOT(l + 8, r + 8, sum, l1, r1);
        }
        _mm256_storeu_ps(unpack, sum);
        result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] +
                 unpack[5] + unpack[6] + unpack[7];

#else
#ifdef __SSE2__
#if defined(FLARE_PLATFORM_OSX)
#define SSE_DOT(addr1, addr2, dest, tmp1, tmp2) \
  tmp1 = _mm_loadu_ps(addr1);                \
  tmp2 = _mm_loadu_ps(addr2);                \
  tmp1 = _mm_mul_ps(tmp1, tmp2);             \
  dest = _mm_add_ps(dest, tmp1);
#else
#define SSE_DOT(addr1, addr2, dest, tmp1, tmp2) \
  tmp1 = _mm128_loadu_ps(addr1);                \
  tmp2 = _mm128_loadu_ps(addr2);                \
  tmp1 = _mm128_mul_ps(tmp1, tmp2);             \
  dest = _mm128_add_ps(dest, tmp1);
#endif
        __m128 sum;
        __m128 l0, l1, l2, l3;
        __m128 r0, r1, r2, r3;
        unsigned D = (size + 3) & ~3U;
        unsigned DR = D % 16;
        unsigned DD = D - DR;
        const float *l = (const float *) a;
        const float *r = (const float *) b;
        const float *e_l = l + DD;
        const float *e_r = r + DD;
        float unpack[4] __attribute__((aligned(16))) = {0, 0, 0, 0};

        sum = _mm_load_ps(unpack);
        switch (DR) {
            case 12:
            SSE_DOT(e_l + 8, e_r + 8, sum, l2, r2);
            case 8:
            SSE_DOT(e_l + 4, e_r + 4, sum, l1, r1);
            case 4:
            SSE_DOT(e_l, e_r, sum, l0, r0);
            default:
                break;
        }
        for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
            SSE_DOT(l, r, sum, l0, r0);
            SSE_DOT(l + 4, r + 4, sum, l1, r1);
            SSE_DOT(l + 8, r + 8, sum, l2, r2);
            SSE_DOT(l + 12, r + 12, sum, l3, r3);
        }
        _mm_storeu_ps(unpack, sum);
        result += unpack[0] + unpack[1] + unpack[2] + unpack[3];
#else

        float        dot0, dot1, dot2, dot3;
        const float *last = a + size;
        const float *unroll_group = last - 3;

        /* Process 4 items with each loop for efficiency. */
        while (a < unroll_group) {
          dot0 = a[0] * b[0];
          dot1 = a[1] * b[1];
          dot2 = a[2] * b[2];
          dot3 = a[3] * b[3];
          result += dot0 + dot1 + dot2 + dot3;
          a += 4;
          b += 4;
        }
        /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
        while (a < last) {
          result += *a++ * *b++;
        }
#endif
#endif
#endif
        return result;
    }

    template<typename T>
    float DistanceFastL2<T>::compare(const T *a, const T *b, float norm,
                                     unsigned size) const {
        float result = -2 * DistanceInnerProduct<T>::inner_product(a, b, size);
        result += norm;
        return result;
    }

    template<typename T>
    float DistanceFastL2<T>::norm(const T *a, unsigned size) const {
        FLARE_CHECK(std::is_floating_point<T>::value) << "ERROR: FastL2 only defined for float currently.";
        float result = 0;
#ifdef __GNUC__
#ifdef __AVX__
#define AVX_L2NORM(addr, dest, tmp) \
  tmp = _mm256_loadu_ps(addr);      \
  tmp = _mm256_mul_ps(tmp, tmp);    \
  dest = _mm256_add_ps(dest, tmp);

        __m256       sum;
        __m256       l0, l1;
        unsigned     D = (size + 7) & ~7U;
        unsigned     DR = D % 16;
        unsigned     DD = D - DR;
        const float *l = (float *) a;
        const float *e_l = l + DD;
        float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};

        sum = _mm256_loadu_ps(unpack);
        if (DR) {
          AVX_L2NORM(e_l, sum, l0);
        }
        for (unsigned i = 0; i < DD; i += 16, l += 16) {
          AVX_L2NORM(l, sum, l0);
          AVX_L2NORM(l + 8, sum, l1);
        }
        _mm256_storeu_ps(unpack, sum);
        result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] +
                 unpack[5] + unpack[6] + unpack[7];
#else
#ifdef __SSE2__
#if defined(FLARE_PLATFORM_OSX)
#define SSE_L2NORM(addr, dest, tmp) \
  tmp = _mm_loadu_ps(addr);      \
  tmp = _mm_mul_ps(tmp, tmp);    \
  dest = _mm_add_ps(dest, tmp);
#else
#define SSE_L2NORM(addr, dest, tmp) \
  tmp = _mm128_loadu_ps(addr);      \
  tmp = _mm128_mul_ps(tmp, tmp);    \
  dest = _mm128_add_ps(dest, tmp);
#endif
        __m128 sum;
        __m128 l0, l1, l2, l3;
        unsigned D = (size + 3) & ~3U;
        unsigned DR = D % 16;
        unsigned DD = D - DR;
        const float *l = (float *) a;
        const float *e_l = l + DD;
        float unpack[4] __attribute__((aligned(16))) = {0, 0, 0, 0};

        sum = _mm_load_ps(unpack);
        switch (DR) {
            case 12:
            SSE_L2NORM(e_l + 8, sum, l2);
            case 8:
            SSE_L2NORM(e_l + 4, sum, l1);
            case 4:
            SSE_L2NORM(e_l, sum, l0);
            default:
                break;
        }
        for (unsigned i = 0; i < DD; i += 16, l += 16) {
            SSE_L2NORM(l, sum, l0);
            SSE_L2NORM(l + 4, sum, l1);
            SSE_L2NORM(l + 8, sum, l2);
            SSE_L2NORM(l + 12, sum, l3);
        }
        _mm_storeu_ps(unpack, sum);
        result += unpack[0] + unpack[1] + unpack[2] + unpack[3];
#else
        float        dot0, dot1, dot2, dot3;
        const float *last = a + size;
        const float *unroll_group = last - 3;

        /* Process 4 items with each loop for efficiency. */
        while (a < unroll_group) {
          dot0 = a[0] * a[0];
          dot1 = a[1] * a[1];
          dot2 = a[2] * a[2];
          dot3 = a[3] * a[3];
          result += dot0 + dot1 + dot2 + dot3;
          a += 4;
        }
        /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
        while (a < last) {
          result += (*a) * (*a);
          a++;
        }
#endif
#endif
#endif
        return result;
    }

    float AVXDistanceInnerProductFloat::compare(const float *a, const float *b,
                                                uint32_t size) const {
        float result = 0.0f;
#define AVX_DOT(addr1, addr2, dest, tmp1, tmp2) \
  tmp1 = _mm256_loadu_ps(addr1);                \
  tmp2 = _mm256_loadu_ps(addr2);                \
  tmp1 = _mm256_mul_ps(tmp1, tmp2);             \
  dest = _mm256_add_ps(dest, tmp1);

        __m256 sum;
        __m256 l0, l1;
        __m256 r0, r1;
        unsigned D = (size + 7) & ~7U;
        unsigned DR = D % 16;
        unsigned DD = D - DR;
        const float *l = (float *) a;
        const float *r = (float *) b;
        const float *e_l = l + DD;
        const float *e_r = r + DD;
        float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};

        sum = _mm256_loadu_ps(unpack);
        if (DR) {
            AVX_DOT(e_l, e_r, sum, l0, r0);
        }

        for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
            AVX_DOT(l, r, sum, l0, r0);
            AVX_DOT(l + 8, r + 8, sum, l1, r1);
        }
        _mm256_storeu_ps(unpack, sum);
        result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] +
                 unpack[5] + unpack[6] + unpack[7];

        return -result;
    }

    template FLARE_EXPORT
    class DistanceInnerProduct<float>;

    template FLARE_EXPORT
    class DistanceInnerProduct<int8_t>;

    template FLARE_EXPORT
    class DistanceInnerProduct<uint8_t>;

    template FLARE_EXPORT
    class DistanceFastL2<float>;

    template FLARE_EXPORT
    class DistanceFastL2<int8_t>;

    template FLARE_EXPORT
    class DistanceFastL2<uint8_t>;

}  // namespace lambda
