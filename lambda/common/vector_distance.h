#pragma once

#include <melon/base/profile.h>

namespace lambda {

    // This function is valid only for float data type.
    template<typename T>
    inline void normalize(T *arr, size_t dim) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < dim; i++) {
            sum += arr[i] * arr[i];
        }
        sum = sqrt(sum);
        for (uint32_t i = 0; i < dim; i++) {
            arr[i] = (T) (arr[i] / sum);
        }
    }

    template<typename T>
    class vector_distance {
    public:
        virtual float compare(const T *a, const T *b, uint32_t length) const = 0;

        virtual ~vector_distance() {
        }
    };

    class DistanceCosineInt8 : public vector_distance<int8_t> {
    public:
        MELON_EXPORT virtual float compare(const int8_t *a, const int8_t *b,
                                                uint32_t length) const override;
    };

    class DistanceL2Int8 : public vector_distance<int8_t> {
    public:
        MELON_EXPORT virtual float compare(const int8_t *a, const int8_t *b,
                                                uint32_t size) const override;
    };

    // AVX implementations. Borrowed from HNSW code.
    class AVXDistanceL2Int8 : public vector_distance<int8_t> {
    public:
        MELON_EXPORT virtual float compare(const int8_t *a, const int8_t *b,
                                                uint32_t length) const override;
    };

    // Slow implementations of the distance functions to get lambda to
    // work in pre-AVX machines. Performance here is not a concern, so we are
    // using the simplest possible implementation.
    template<typename T>
    class SlowDistanceL2Int : public vector_distance<T> {
    public:
        // Implementing here because this is a template function
        MELON_EXPORT virtual float compare(const T *a, const T *b,
                                                uint32_t length) const override {
            uint32_t result = 0;
            for (uint32_t i = 0; i < length; i++) {
                result += ((int32_t) ((int16_t) a[i] - (int16_t) b[i])) *
                          ((int32_t) ((int16_t) a[i] - (int16_t) b[i]));
            }
            return (float) result;
        }
    };

    class DistanceCosineFloat : public vector_distance<float> {
    public:
        MELON_EXPORT virtual float compare(const float *a, const float *b,
                                                uint32_t length) const override;
    };

    class DistanceL2Float : public vector_distance<float> {
    public:
        MELON_EXPORT virtual float compare(const float *a, const float *b,
                                                uint32_t size) const override
        __attribute__((hot));
    };

    class AVXDistanceL2Float : public vector_distance<float> {
    public:
        MELON_EXPORT virtual float compare(const float *a, const float *b,
                                                uint32_t length) const override;
    };

    class SlowDistanceL2Float : public vector_distance<float> {
    public:
        MELON_EXPORT virtual float compare(const float *a, const float *b,
                                                uint32_t length) const;
    };

    class SlowDistanceCosineUInt8 : public vector_distance<uint8_t> {
    public:
        MELON_EXPORT virtual float compare(const uint8_t *a, const uint8_t *b,
                                                uint32_t length) const override;
    };

    class DistanceL2UInt8 : public vector_distance<uint8_t> {
    public:
        MELON_EXPORT virtual float compare(const uint8_t *a, const uint8_t *b,
                                                uint32_t size) const override;
    };

    template<typename T>
    class DistanceInnerProduct : public vector_distance<T> {
    public:
        float inner_product(const T *a, const T *b, unsigned size) const;

        float compare(const T *a, const T *b, unsigned size) const override {
            // since we use normally minimization objective for distance
            // comparisons, we are returning 1/x.
            float result = inner_product(a, b, size);
            //      if (result < 0)
            //      return std::numeric_limits<float>::max();
            //      else
            return -result;
        }
    };

    template<typename T>
    class DistanceFastL2
            : public DistanceInnerProduct<T> {  // currently defined only for float.
        // templated for future use.
    public:
        float norm(const T *a, unsigned size) const;

        float compare(const T *a, const T *b, float norm, unsigned size) const;
    };

    class AVXDistanceInnerProductFloat : public vector_distance<float> {
    public:
        MELON_EXPORT virtual float compare(const float *a, const float *b,
                                                uint32_t length) const override;
    };

    class AVXNormalizedCosineDistanceFloat : public vector_distance<float> {
    private:
        AVXDistanceInnerProductFloat _innerProduct;

    public:
        MELON_EXPORT virtual float compare(const float *a, const float *b,
                                                uint32_t length) const override {
            // Inner product returns negative values to indicate distance.
            // This will ensure that cosine is between -1 and 1.
            return 1.0f + _innerProduct.compare(a, b, length);
        }
    };

}  // namespace lambda
