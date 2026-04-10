#include "physics/VehicleController.h"

#include "physics/PhysicsWorld.h"

#include <algorithm>
#include <cmath>

namespace enigma {

VehicleController::VehicleController(PhysicsWorld& world, const VehicleConfig& config, vec3 spawnPosition)
    : m_world(&world)
    , m_config(config) {
    // Create car body as a dynamic box approximating a sedan chassis.
    const vec3 halfExtents{1.0f, 0.5f, 2.0f};
    m_bodyId = m_world->addDynamicBox(spawnPosition, halfExtents, config.mass);
}

VehicleController::~VehicleController() {
    if (m_world) {
        m_world->removeBody(m_bodyId);
    }
}

void VehicleController::setInput(const VehicleInput& input) {
    m_input = input;
}

void VehicleController::update(f32 dt) {
    // TODO: replace with JPH::VehicleConstraint for proper tire model
    // Simplified force-based vehicle simulation:
    // - Forward force from throttle
    // - Brake deceleration
    // - Steering via angular velocity adjustment

    if (dt <= 0.0f) return;

    const vec3 vel     = m_world->getLinearVelocity(m_bodyId);
    const mat4 xform   = m_world->getWorldTransform(m_bodyId);
    const vec3 forward = glm::normalize(vec3(xform[2])); // local Z axis
    const f32  speed   = glm::dot(vel, forward);

    // Throttle force: F = throttle * maxTorque / wheelRadius.
    const f32 wheelRadius = m_config.wheels.empty() ? 0.33f : m_config.wheels[0].tire.radius;
    const f32 driveForce  = m_input.throttle * m_config.maxEngineTorque / wheelRadius;

    // Brake force: opposes current velocity.
    const f32 brakeForce = m_input.brake * 20000.0f; // N (strong brakes)

    // Net longitudinal force.
    f32 netForce = driveForce;
    if (std::abs(speed) > 0.1f) {
        netForce -= brakeForce * (speed > 0.0f ? 1.0f : -1.0f);
    }
    if (m_input.handbrake) {
        netForce -= speed * 5000.0f; // friction-like decel
    }

    // Apply forward impulse.
    const vec3 forwardImpulse = forward * netForce * dt;
    m_world->applyImpulse(m_bodyId, forwardImpulse);

    // Steering: apply a lateral velocity correction to simulate turning.
    if (std::abs(m_input.steering) > 0.01f && std::abs(speed) > 0.5f) {
        const vec3 right = glm::normalize(vec3(xform[0])); // local X axis
        const f32  steerForce = m_input.steering * std::min(std::abs(speed), 30.0f) * 800.0f;
        m_world->applyImpulse(m_bodyId, right * steerForce * dt);
    }

    // Lateral friction: dampen sideways velocity to keep car on track.
    {
        const vec3 right      = glm::normalize(vec3(xform[0]));
        const f32  lateralVel = glm::dot(vel, right);
        const vec3 correction = -right * lateralVel * 0.9f;
        m_world->applyImpulse(m_bodyId, correction * m_config.mass * dt);
    }
}

mat4 VehicleController::bodyTransform() const {
    return m_world->getWorldTransform(m_bodyId);
}

mat4 VehicleController::wheelTransform(u32 wheelIndex) const {
    if (wheelIndex >= wheelCount()) return mat4(1.0f);
    // Approximate wheel positions from body transform + config offsets.
    const mat4 body  = bodyTransform();
    const vec3 offset = m_config.wheels[wheelIndex].offset;
    mat4 result = body;
    result[3]  += body * vec4(offset, 0.0f);
    return result;
}

u32 VehicleController::wheelCount() const {
    return static_cast<u32>(m_config.wheels.size());
}

vec3 VehicleController::velocity() const {
    return m_world->getLinearVelocity(m_bodyId);
}

f32 VehicleController::speedKmh() const {
    return glm::length(velocity()) * 3.6f; // m/s → km/h
}

} // namespace enigma
