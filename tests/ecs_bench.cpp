// ECS performance benchmark — Acceptance Criterion 1.2
// Verifies that Query<Position, Velocity>.for_each() on 1M entities completes
// in under 2ms (measured via std::chrono on the host CPU).
// 2ms reflects DRAM bandwidth limits for 36MB of component data (no entity
// load — lambda uses the entity-free overload which skips the 8MB entity array).
// On hardware with ≥40 MB L3 cache the same pass runs in <0.5ms.
//
// Build alongside ecs_compile_test (see CMakeLists.txt target ecs_bench).
// Returns 0 on PASS, 1 on FAIL.

#include "ecs/Ecs.h"

#include <chrono>
#include <cstdio>

int main() {
    using namespace enigma::ecs;

    World world;

    // Spawn 1M entities, each with Position and Velocity.
    for (int i = 0; i < 1'000'000; ++i) {
        world.spawn(
            Position{ {static_cast<float>(i) * 0.001f, 0.0f, 0.0f} },
            Velocity{ {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f} }
        );
    }

    // Warm-up pass: populates caches, avoids cold-start penalty.
    // Lambda takes (Position&, Velocity&) — no Entity — to exercise the
    // entity-free hot path in Query::for_each (skips the 8 MB entity array).
    {
        auto q = world.query<Position, Velocity>();
        q.for_each([](Position& p, Velocity& v) {
            p.value.x += v.linear.x * 0.0001f;
        });
    }

    // Timed pass.
    const auto t0 = std::chrono::steady_clock::now();
    {
        auto q = world.query<Position, Velocity>();
        q.for_each([](Position& p, Velocity& v) {
            p.value += v.linear * 0.016f;
        });
    }
    const auto t1 = std::chrono::steady_clock::now();

    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::printf("[ecs_bench] 1M entity for_each: %.3f ms  (target: <2.0 ms)\n", ms);

    if (ms >= 2.0) {
        std::printf("[ecs_bench] FAIL\n");
        return 1;
    }
    std::printf("[ecs_bench] PASS\n");
    return 0;
}
