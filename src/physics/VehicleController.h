#pragma once

#include "physics/VehicleConfig.h"
#include "core/Types.h"
#include "core/Math.h"

namespace enigma {

class PhysicsWorld;

struct VehicleInput {
    f32  steering  = 0.0f; // -1 (left) to +1 (right)
    f32  throttle  = 0.0f; //  0 to 1
    f32  brake     = 0.0f; //  0 to 1
    bool handbrake = false;
};

class VehicleController {
public:
    VehicleController(PhysicsWorld& world, const VehicleConfig& config, vec3 spawnPosition);
    ~VehicleController();

    void setInput(const VehicleInput& input);
    void update(f32 dt);

    // Rendering sync.
    mat4 bodyTransform() const;
    mat4 wheelTransform(u32 wheelIndex) const;
    u32  wheelCount() const;
    vec3 velocity() const;
    f32  speedKmh() const;

    u32 bodyId() const { return m_bodyId; }

private:
    PhysicsWorld* m_world = nullptr;
    u32           m_bodyId = 0;
    VehicleConfig m_config;
    VehicleInput  m_input;
    // TODO: replace with JPH::VehicleConstraint for proper tire model
};

} // namespace enigma
