#pragma once

#include "TypeId.h"
#include "Component.h"
#include "Entity.h"
#include "Archetype.h"
#include "Query.h"
#include "System.h"

#include "../core/Assert.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace enigma::ecs {

class World {
public:
    template<Component... Cs>
    Entity spawn(Cs... components) {
        std::vector<TypeId> type_ids = { type_id<Cs>()... };
        std::vector<ColumnLayout> layouts = {
            ColumnLayout{ type_id<Cs>(), sizeof(Cs), alignof(Cs) }...
        };
        ArchetypeId aid = compute_archetype_id(type_ids);
        Archetype* arch = get_or_create_archetype(aid, layouts);

        Entity e = m_entity_pool.allocate();

        // Build a TypeId -> component-pointer map first, then iterate the
        // archetype's SORTED layout to produce the data array in the correct
        // column order. Without this, when type_id sort order differs from
        // the Cs... pack order, component bytes are written to the wrong columns.
        const TypeId   pack_ids[]  = { type_id<Cs>()... };
        const void*    pack_ptrs[] = { static_cast<const void*>(&components)... };
        std::vector<const void*> data;
        data.reserve(sizeof...(Cs));
        for (const auto& col : arch->layout()) {
            for (size_t k = 0; k < sizeof...(Cs); ++k) {
                if (pack_ids[k] == col.type_id) {
                    data.push_back(pack_ptrs[k]);
                    break;
                }
            }
        }
        size_t row = arch->emplace(e, data);

        m_entity_records[e.index()] = EntityRecord{ aid, row };
        return e;
    }

    void destroy(Entity e) {
        ENIGMA_ASSERT(is_alive(e));
        auto it = m_entity_records.find(e.index());
        ENIGMA_ASSERT(it != m_entity_records.end());

        EntityRecord rec = it->second;
        Archetype* arch = m_archetypes[rec.archetype_id].get();
        Entity moved = arch->remove(rec.row);

        if (moved != e && rec.row < arch->size()) {
            m_entity_records[moved.index()].row = rec.row;
        }

        m_entity_records.erase(it);
        m_entity_pool.free(e);
    }

    bool is_alive(Entity e) const {
        return m_entity_pool.is_alive(e);
    }

    template<Component C>
    void add_component(Entity e, C component) {
        ENIGMA_ASSERT(is_alive(e));
        auto it = m_entity_records.find(e.index());
        ENIGMA_ASSERT(it != m_entity_records.end());

        EntityRecord old_rec = it->second;
        Archetype* old_arch = m_archetypes[old_rec.archetype_id].get();

        std::vector<ColumnLayout> new_layouts = old_arch->layout();
        new_layouts.push_back(ColumnLayout{ type_id<C>(), sizeof(C), alignof(C) });

        std::vector<TypeId> new_type_ids;
        new_type_ids.reserve(new_layouts.size());
        for (auto& l : new_layouts) new_type_ids.push_back(l.type_id);

        ArchetypeId new_aid = compute_archetype_id(new_type_ids);
        Archetype* new_arch = get_or_create_archetype(new_aid, new_layouts);

        std::vector<const void*> data;
        data.reserve(new_layouts.size());
        for (auto& col : new_arch->layout()) {
            if (col.type_id == type_id<C>()) {
                data.push_back(static_cast<const void*>(&component));
            } else {
                data.push_back(old_arch->get_component(old_rec.row, col.type_id));
            }
        }

        size_t new_row = new_arch->emplace(e, data);

        Entity moved = old_arch->remove(old_rec.row);
        if (moved != e && old_rec.row < old_arch->size()) {
            m_entity_records[moved.index()].row = old_rec.row;
        }

        it->second = EntityRecord{ new_aid, new_row };
    }

    template<Component C>
    void remove_component(Entity e) {
        ENIGMA_ASSERT(is_alive(e));
        auto it = m_entity_records.find(e.index());
        ENIGMA_ASSERT(it != m_entity_records.end());

        EntityRecord old_rec = it->second;
        Archetype* old_arch = m_archetypes[old_rec.archetype_id].get();

        std::vector<ColumnLayout> new_layouts;
        for (auto& col : old_arch->layout()) {
            if (col.type_id != type_id<C>()) {
                new_layouts.push_back(col);
            }
        }

        std::vector<TypeId> new_type_ids;
        new_type_ids.reserve(new_layouts.size());
        for (auto& l : new_layouts) new_type_ids.push_back(l.type_id);

        ArchetypeId new_aid = compute_archetype_id(new_type_ids);
        Archetype* new_arch = get_or_create_archetype(new_aid, new_layouts);

        std::vector<const void*> data;
        data.reserve(new_layouts.size());
        for (auto& col : new_arch->layout()) {
            data.push_back(old_arch->get_component(old_rec.row, col.type_id));
        }

        size_t new_row = new_arch->emplace(e, data);

        Entity moved = old_arch->remove(old_rec.row);
        if (moved != e && old_rec.row < old_arch->size()) {
            m_entity_records[moved.index()].row = old_rec.row;
        }

        it->second = EntityRecord{ new_aid, new_row };
    }

    template<Component C>
    C& get(Entity e) {
        ENIGMA_ASSERT(is_alive(e));
        auto it = m_entity_records.find(e.index());
        ENIGMA_ASSERT(it != m_entity_records.end());
        Archetype* arch = m_archetypes[it->second.archetype_id].get();
        return *static_cast<C*>(arch->get_component(it->second.row, type_id<C>()));
    }

    template<Component C>
    const C& get(Entity e) const {
        ENIGMA_ASSERT(is_alive(e));
        auto it = m_entity_records.find(e.index());
        ENIGMA_ASSERT(it != m_entity_records.end());
        const Archetype* arch = m_archetypes.at(it->second.archetype_id).get();
        return *static_cast<const C*>(arch->get_component(it->second.row, type_id<C>()));
    }

    template<Component... Cs>
    Query<Cs...> query() {
        std::vector<Archetype*> matches;
        for (auto& [aid, arch] : m_archetypes) {
            bool has_all = (arch->has_component(type_id<Cs>()) && ...);
            if (has_all) matches.push_back(arch.get());
        }
        return Query<Cs...>(std::move(matches));
    }

    void add_system(SystemSchedule schedule,
                    std::function<void(float)> fn,
                    ExecutionPolicy policy = ExecutionPolicy::Sequential)
    {
        m_systems.push_back(SystemEntry{ std::move(fn), schedule, policy });
    }

    void run_systems(float dt) {
        static constexpr SystemSchedule order[] = {
            SystemSchedule::PrePhysics,
            SystemSchedule::Physics,
            SystemSchedule::PostPhysics,
            SystemSchedule::PreRender,
            SystemSchedule::Render,
            SystemSchedule::PostRender
        };
        for (auto phase : order) {
            for (auto& entry : m_systems) {
                if (entry.schedule == phase) {
                    entry.fn(dt);
                }
            }
        }
    }

    void clear() {
        m_entity_records.clear();
        m_archetypes.clear();
        m_entity_pool.clear();
        m_systems.clear();
    }

private:
    struct EntityRecord {
        ArchetypeId archetype_id;
        size_t      row;
    };

    std::unordered_map<uint32_t, EntityRecord>                  m_entity_records;
    std::unordered_map<ArchetypeId, std::unique_ptr<Archetype>> m_archetypes;
    EntityPool                                                  m_entity_pool;
    std::vector<SystemEntry>                                    m_systems;

    Archetype* get_or_create_archetype(ArchetypeId id,
                                       const std::vector<ColumnLayout>& layouts)
    {
        auto it = m_archetypes.find(id);
        if (it != m_archetypes.end()) return it->second.get();

        auto sorted = layouts;
        std::sort(sorted.begin(), sorted.end(),
                  [](const ColumnLayout& a, const ColumnLayout& b) {
                      return a.type_id < b.type_id;
                  });

        auto arch = std::make_unique<Archetype>(std::move(sorted));
        Archetype* ptr = arch.get();
        m_archetypes[id] = std::move(arch);
        return ptr;
    }
};

} // namespace enigma::ecs
