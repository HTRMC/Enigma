#pragma once

#include "core/Types.h"
#include "core/Math.h"

#include <memory>

// Forward declarations — avoid pulling all Jolt headers into every translation unit.
namespace JPH {
    class PhysicsSystem;
    class TempAllocatorMalloc;
    class JobSystemThreadPool;
    class BroadPhaseLayerInterfaceTable;
    class ObjectVsBroadPhaseLayerFilterTable;
    class ObjectLayerPairFilterTable;
    class Body;
    class BodyID;
}

namespace enigma {

// Physics object layers.
namespace PhysicsLayer {
    inline constexpr u16 Static  = 0; // immovable terrain, buildings
    inline constexpr u16 Dynamic = 1; // rigid bodies
    inline constexpr u16 Vehicle = 2; // vehicle bodies
    inline constexpr u16 Count   = 3;
}

// Broad-phase layers (coarse-grained grouping for the broadphase tree).
namespace BPLayer {
    inline constexpr u8 Static  = 0;
    inline constexpr u8 Dynamic = 1;
    inline constexpr u8 Count   = 2;
}

class PhysicsWorld {
public:
    PhysicsWorld();
    ~PhysicsWorld();

    PhysicsWorld(const PhysicsWorld&)            = delete;
    PhysicsWorld& operator=(const PhysicsWorld&) = delete;

    // Fixed timestep physics step (120 Hz = dt 1/120).
    void step(f32 dt);

    // Body creation API — returns BodyID index (use to set/get transforms, apply forces).
    u32 addStaticBox(vec3 position, vec3 halfExtents);
    u32 addStaticPlane(vec3 normal, f32 offset);
    u32 addDynamicBox(vec3 position, vec3 halfExtents, f32 mass = 1.0f);
    u32 addDynamicSphere(vec3 position, f32 radius, f32 mass = 1.0f);

    void removeBody(u32 bodyId);

    // Transform accessors (for rendering sync).
    vec3 getPosition(u32 bodyId) const;
    quat getRotation(u32 bodyId) const;
    mat4 getWorldTransform(u32 bodyId) const;

    // Apply forces / impulses.
    void applyImpulse(u32 bodyId, vec3 impulse);
    void applyAngularImpulse(u32 bodyId, vec3 angularImpulse);
    void setLinearVelocity(u32 bodyId, vec3 velocity);
    vec3 getLinearVelocity(u32 bodyId) const;
    vec3 getAngularVelocity(u32 bodyId) const;

    // Interpolation support.
    f32 accumulator() const { return m_accumulator; }
    static constexpr f32 kFixedDt = 1.0f / 120.0f;

    JPH::PhysicsSystem& system() { return *m_physicsSystem; }

private:
    // Jolt requires these to stay alive for the lifetime of PhysicsSystem.
    std::unique_ptr<JPH::TempAllocatorMalloc>              m_tempAllocator;
    std::unique_ptr<JPH::JobSystemThreadPool>              m_jobSystem;
    std::unique_ptr<JPH::BroadPhaseLayerInterfaceTable>    m_bpLayerInterface;
    std::unique_ptr<JPH::ObjectVsBroadPhaseLayerFilterTable> m_objBpFilter;
    std::unique_ptr<JPH::ObjectLayerPairFilterTable>       m_objLayerFilter;
    std::unique_ptr<JPH::PhysicsSystem>                    m_physicsSystem;

    f32 m_accumulator = 0.0f;
};

} // namespace enigma
