#pragma once

#include "TypeId.h"
#include "Component.h"
#include "Archetype.h"

#include <vector>

namespace enigma::ecs {

template<Component... Cs>
class Query {
public:
    explicit Query(std::vector<Archetype*> matches)
        : m_matches(std::move(matches)) {}

    template<typename Fn>
    void for_each(Fn&& fn) {
        for (Archetype* arch : m_matches) {
            const size_t count = arch->size();
            if (count == 0) continue;

            auto columns = std::tuple{ reinterpret_cast<Cs*>(arch->get_column(type_id<Cs>()).data())... };

            // If the callable accepts (Cs&...) without Entity, skip the entity
            // array entirely — saves one memory stream on the hot path.
            if constexpr (std::is_invocable_v<Fn, Cs&...>) {
                for (size_t i = 0; i < count; ++i) {
                    fn(std::get<Cs*>(columns)[i]...);
                }
            } else {
                const Entity* entities = arch->entities().data();
                for (size_t i = 0; i < count; ++i) {
                    fn(entities[i], std::get<Cs*>(columns)[i]...);
                }
            }
        }
    }

    template<typename Fn>
    void for_each(Fn&& fn) const {
        for (const Archetype* arch : m_matches) {
            const size_t count = arch->size();
            if (count == 0) continue;

            auto columns = std::tuple{ reinterpret_cast<const Cs*>(arch->get_column(type_id<Cs>()).data())... };

            if constexpr (std::is_invocable_v<Fn, const Cs&...>) {
                for (size_t i = 0; i < count; ++i) {
                    fn(std::get<const Cs*>(columns)[i]...);
                }
            } else {
                const Entity* entities = arch->entities().data();
                for (size_t i = 0; i < count; ++i) {
                    fn(entities[i], std::get<const Cs*>(columns)[i]...);
                }
            }
        }
    }

    size_t count() const {
        size_t total = 0;
        for (const Archetype* arch : m_matches) total += arch->size();
        return total;
    }

private:
    std::vector<Archetype*> m_matches;

    static bool archetype_matches(const Archetype& arch) {
        return (arch.has_component(type_id<Cs>()) && ...);
    }

    template<Component... Qs>
    friend class Query;

    friend class World;
};

} // namespace enigma::ecs
