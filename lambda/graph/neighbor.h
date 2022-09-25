/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#pragma once

#include <cstddef>
#include <mutex>
#include <vector>

namespace lambda {

    struct neighbor {
        unsigned id;
        float distance;
        bool flag;

        neighbor() = default;

        neighbor(unsigned id, float distance, bool f)
                : id{id}, distance{distance}, flag(f) {
        }

        inline bool operator<(const neighbor &other) const {
            return distance < other.distance;
        }

        inline bool operator==(const neighbor &other) const {
            return (id == other.id);
        }
    };

    struct simple_neighbor {
        unsigned id;
        float distance;

        simple_neighbor() = default;

        simple_neighbor(unsigned id, float distance) : id(id), distance(distance) {
        }

        inline bool operator<(const simple_neighbor &other) const {
            return distance < other.distance;
        }

        inline bool operator==(const simple_neighbor &other) const {
            return id == other.id;
        }
    };

    struct simple_neighbors {
        std::vector<simple_neighbor> pool;
    };

    static inline unsigned insert_into_pool(neighbor *addr, unsigned K,
                                            neighbor nn) {
        // find the location to insert
        unsigned left = 0, right = K - 1;
        if (addr[left].distance > nn.distance) {
            memmove((char *) &addr[left + 1], &addr[left], K * sizeof(neighbor));
            addr[left] = nn;
            return left;
        }
        if (addr[right].distance < nn.distance) {
            addr[K] = nn;
            return K;
        }
        while (right > 1 && left < right - 1) {
            unsigned mid = (left + right) / 2;
            if (addr[mid].distance > nn.distance)
                right = mid;
            else
                left = mid;
        }
        // check equal ID

        while (left > 0) {
            if (addr[left].distance < nn.distance)
                break;
            if (addr[left].id == nn.id)
                return K + 1;
            left--;
        }
        if (addr[left].id == nn.id || addr[right].id == nn.id)
            return K + 1;
        memmove((char *) &addr[right + 1], &addr[right],
                (K - right) * sizeof(neighbor));
        addr[right] = nn;
        return right;
    }
}  // namespace lambda
