#pragma once

#include <concepts>
#include <cstdint>
#include <functional>

namespace enigma::ecs {

template<typename T>
concept System = std::invocable<T, float>;

enum class SystemSchedule : uint8_t {
    PrePhysics,
    Physics,
    PostPhysics,
    PreRender,
    Render,
    PostRender
};

enum class ExecutionPolicy : uint8_t {
    Sequential,
    Parallel
};

struct SystemEntry {
    std::function<void(float)> fn;
    SystemSchedule             schedule;
    ExecutionPolicy            policy;
};

} // namespace enigma::ecs
