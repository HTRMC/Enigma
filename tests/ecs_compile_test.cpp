// Compile-only test for the ECS core library.
// cl /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127 /I src /I <glm> ecs_compile_test.cpp

#include "ecs/Ecs.h"

#include <cassert>
#include <cstdio>

using namespace enigma::ecs;

// Verify Component concept
static_assert(Component<Position>);
static_assert(Component<Rotation>);
static_assert(Component<Velocity>);
static_assert(Component<MeshRef>);

// Verify ComponentMeta
static_assert(ComponentMeta<Position>::size == sizeof(Position));
static_assert(ComponentMeta<Position>::align == alignof(Position));
static_assert(ComponentMeta<Position>::name == "enigma::ecs::Position");

// Verify type_id is constexpr and unique
static_assert(type_id<Position>() != type_id<Rotation>());
static_assert(type_id<Position>() != type_id<Velocity>());
static_assert(type_id<Rotation>() != type_id<Scale>());

// Verify Entity
static_assert(Entity::make(42, 7).index() == 42);
static_assert(Entity::make(42, 7).generation() == 7);
static_assert(kNullEntity.bits == UINT64_MAX);

int main() {
    // EntityPool basic test
    EntityPool pool;
    Entity e0 = pool.allocate();
    Entity e1 = pool.allocate();
    assert(e0.index() == 0 && e0.generation() == 0);
    assert(e1.index() == 1 && e1.generation() == 0);
    assert(pool.is_alive(e0));
    pool.free(e0);
    assert(!pool.is_alive(e0));
    Entity e2 = pool.allocate();
    assert(e2.index() == 0 && e2.generation() == 1);

    // World: spawn + get
    World world;
    Position pos{ {1.0f, 2.0f, 3.0f} };
    Velocity vel{ {4.0f, 5.0f, 6.0f}, {0.0f, 0.0f, 0.0f} };
    Entity ent = world.spawn(pos, vel);
    assert(world.is_alive(ent));

    auto& p = world.get<Position>(ent);
    assert(p.value.x == 1.0f && p.value.y == 2.0f && p.value.z == 3.0f);

    auto& v = world.get<Velocity>(ent);
    assert(v.linear.x == 4.0f);

    // World: query
    auto q = world.query<Position, Velocity>();
    int count = 0;
    q.for_each([&](Entity e, Position& pp, Velocity& vv) {
        assert(e == ent);
        assert(pp.value.x == 1.0f);
        assert(vv.linear.x == 4.0f);
        ++count;
    });
    assert(count == 1);

    // World: add_component
    Scale scl{ {2.0f, 2.0f, 2.0f} };
    world.add_component(ent, scl);
    auto& s = world.get<Scale>(ent);
    assert(s.value.x == 2.0f);

    // Original components survived migration
    auto& p2 = world.get<Position>(ent);
    assert(p2.value.x == 1.0f);

    // World: remove_component
    world.remove_component<Velocity>(ent);
    auto q2 = world.query<Position, Scale>();
    int count2 = 0;
    q2.for_each([&](Entity, Position& pp, Scale& ss) {
        assert(pp.value.x == 1.0f);
        assert(ss.value.x == 2.0f);
        ++count2;
    });
    assert(count2 == 1);

    // World: destroy
    world.destroy(ent);
    assert(!world.is_alive(ent));

    // World: systems
    bool ran = false;
    world.add_system(SystemSchedule::PrePhysics, [&](float dt) {
        assert(dt == 0.016f);
        ran = true;
    });
    world.run_systems(0.016f);
    assert(ran);

    std::printf("[ecs_compile_test] All tests passed.\n");
    return 0;
}
