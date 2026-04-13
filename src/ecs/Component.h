#pragma once

#include "TypeId.h"

#include <string_view>
#include <type_traits>

namespace enigma::ecs {

template<typename T>
concept Component = std::is_trivially_copyable_v<T>
                  && !std::is_empty_v<T>
                  && std::is_default_constructible_v<T>;

template<typename T>
struct ComponentMeta {
    static_assert(sizeof(T) == 0,
        "ComponentMeta<T> not specialized. Use ENIGMA_COMPONENT(T, ...) to register.");
};

} // namespace enigma::ecs

#define ENIGMA_COMPONENT(Type, ...)                                            \
    template<>                                                                 \
    struct enigma::ecs::ComponentMeta<Type> {                                  \
        static constexpr enigma::ecs::TypeId id    = enigma::ecs::type_id<Type>(); \
        static constexpr std::size_t         size  = sizeof(Type);             \
        static constexpr std::size_t         align = alignof(Type);            \
        static constexpr std::string_view    name  = #Type;                    \
    }
