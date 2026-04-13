#pragma once

#include "../core/Assert.h"

#include <cstdint>
#include <vector>

namespace enigma::ecs {

struct Entity {
    uint64_t bits = UINT64_MAX;

    constexpr uint32_t index()      const { return static_cast<uint32_t>(bits & 0xFFFFFFFF); }
    constexpr uint32_t generation() const { return static_cast<uint32_t>(bits >> 32); }

    static constexpr Entity make(uint32_t idx, uint32_t gen) {
        return Entity{ static_cast<uint64_t>(gen) << 32 | static_cast<uint64_t>(idx) };
    }

    constexpr bool operator==(const Entity& o) const { return bits == o.bits; }
    constexpr bool operator!=(const Entity& o) const { return bits != o.bits; }
};

inline constexpr Entity kNullEntity{ UINT64_MAX };

class EntityPool {
public:
    Entity allocate() {
        uint32_t idx;
        if (!m_free_list.empty()) {
            idx = m_free_list.back();
            m_free_list.pop_back();
        } else {
            idx = static_cast<uint32_t>(m_generations.size());
            m_generations.push_back(0);
        }
        return Entity::make(idx, m_generations[idx]);
    }

    void free(Entity e) {
        uint32_t idx = e.index();
        ENIGMA_ASSERT(idx < m_generations.size());
        ENIGMA_ASSERT(m_generations[idx] == e.generation());
        ++m_generations[idx];
        m_free_list.push_back(idx);
    }

    bool is_alive(Entity e) const {
        uint32_t idx = e.index();
        if (idx >= m_generations.size()) return false;
        return m_generations[idx] == e.generation();
    }

    void clear() {
        m_generations.clear();
        m_free_list.clear();
    }

private:
    std::vector<uint32_t> m_generations;
    std::vector<uint32_t> m_free_list;
};

} // namespace enigma::ecs
