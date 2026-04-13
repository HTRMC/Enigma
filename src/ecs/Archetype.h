#pragma once

#include "TypeId.h"
#include "Entity.h"
#include "Component.h"

#include "../core/Assert.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace enigma::ecs {

using ArchetypeId = uint64_t;

struct ColumnLayout {
    TypeId  type_id;
    size_t  elem_size;
    size_t  elem_align;
};

class Archetype {
public:
    explicit Archetype(std::vector<ColumnLayout> layout)
        : m_layout(std::move(layout))
    {
        m_columns.resize(m_layout.size());
    }

    size_t size() const { return m_entities.size(); }

    bool has_component(TypeId tid) const {
        for (auto& col : m_layout)
            if (col.type_id == tid) return true;
        return false;
    }

    int column_index(TypeId tid) const {
        for (int i = 0; i < static_cast<int>(m_layout.size()); ++i)
            if (m_layout[i].type_id == tid) return i;
        return -1;
    }

    const std::vector<ColumnLayout>& layout() const { return m_layout; }
    const std::vector<Entity>& entities() const { return m_entities; }

    std::span<uint8_t> get_column(TypeId tid) {
        int idx = column_index(tid);
        ENIGMA_ASSERT(idx >= 0);
        return std::span<uint8_t>(m_columns[idx]);
    }

    std::span<const uint8_t> get_column(TypeId tid) const {
        int idx = column_index(tid);
        ENIGMA_ASSERT(idx >= 0);
        return std::span<const uint8_t>(m_columns[idx]);
    }

    size_t emplace(Entity e, const std::vector<const void*>& component_data) {
        ENIGMA_ASSERT(component_data.size() == m_layout.size());
        size_t row = m_entities.size();
        m_entities.push_back(e);
        for (size_t i = 0; i < m_layout.size(); ++i) {
            size_t old_size = m_columns[i].size();
            m_columns[i].resize(old_size + m_layout[i].elem_size);
            std::memcpy(m_columns[i].data() + old_size, component_data[i], m_layout[i].elem_size);
        }
        return row;
    }

    Entity remove(size_t row) {
        ENIGMA_ASSERT(row < m_entities.size());
        size_t last = m_entities.size() - 1;
        Entity moved = m_entities[last];

        if (row != last) {
            m_entities[row] = m_entities[last];
            for (size_t i = 0; i < m_layout.size(); ++i) {
                size_t es = m_layout[i].elem_size;
                std::memcpy(m_columns[i].data() + row * es,
                            m_columns[i].data() + last * es,
                            es);
            }
        }

        m_entities.pop_back();
        for (size_t i = 0; i < m_layout.size(); ++i) {
            m_columns[i].resize(m_columns[i].size() - m_layout[i].elem_size);
        }
        return moved;
    }

    void* get_component(size_t row, TypeId tid) {
        int idx = column_index(tid);
        ENIGMA_ASSERT(idx >= 0);
        return m_columns[idx].data() + row * m_layout[idx].elem_size;
    }

    const void* get_component(size_t row, TypeId tid) const {
        int idx = column_index(tid);
        ENIGMA_ASSERT(idx >= 0);
        return m_columns[idx].data() + row * m_layout[idx].elem_size;
    }

private:
    std::vector<ColumnLayout>             m_layout;
    std::vector<Entity>                   m_entities;
    std::vector<std::vector<uint8_t>>     m_columns;
};

template<Component C>
std::span<C> typed_column(Archetype& arch) {
    auto col = arch.get_column(type_id<C>());
    return std::span<C>(reinterpret_cast<C*>(col.data()), col.size() / sizeof(C));
}

template<Component C>
std::span<const C> typed_column(const Archetype& arch) {
    auto col = arch.get_column(type_id<C>());
    return std::span<const C>(reinterpret_cast<const C*>(col.data()), col.size() / sizeof(C));
}

inline ArchetypeId compute_archetype_id(std::vector<TypeId> type_ids) {
    std::sort(type_ids.begin(), type_ids.end());
    uint64_t hash = 14695981039346656037ULL;
    for (auto tid : type_ids) {
        hash ^= tid;
        hash *= 1099511628211ULL;
    }
    return hash;
}

} // namespace enigma::ecs
