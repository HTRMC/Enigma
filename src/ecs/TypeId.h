#pragma once

#include <cstdint>

namespace enigma::ecs {

using TypeId = uint64_t;

namespace detail {

constexpr uint64_t fnv1a_64(const char* str, uint64_t hash = 14695981039346656037ULL) {
    return (*str == '\0') ? hash : fnv1a_64(str + 1, (hash ^ static_cast<uint64_t>(*str)) * 1099511628211ULL);
}

template<typename T>
constexpr const char* type_signature() {
    return __FUNCSIG__;
}

} // namespace detail

template<typename T>
constexpr TypeId type_id() {
    return detail::fnv1a_64(detail::type_signature<T>());
}

} // namespace enigma::ecs
