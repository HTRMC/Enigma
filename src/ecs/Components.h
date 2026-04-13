#pragma once

#include "Component.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstdint>

namespace enigma::ecs {

struct Position    { glm::vec3 value{0.0f, 0.0f, 0.0f}; };
struct Rotation    { glm::quat value{1.0f, 0.0f, 0.0f, 0.0f}; };
struct Scale       { glm::vec3 value{1.0f, 1.0f, 1.0f}; };
struct LocalToWorld{ glm::mat4 matrix{1.0f}; };
struct Velocity    { glm::vec3 linear{0.0f, 0.0f, 0.0f}; glm::vec3 angular{0.0f, 0.0f, 0.0f}; };
struct MeshRef     { uint32_t mesh_index{0}; };
struct MaterialRef { uint32_t mat_index{0}; };
struct PhysicsBody { uint32_t body_id{0}; };
struct CameraTag   { uint8_t is_primary{0}; };
struct VehicleTag      { uint8_t player_controlled{0}; };

// Input controls written by InputSystem each PrePhysics tick.
// Mirrors enigma::VehicleInput but lives in ECS storage.
// uint8_t pads keep the struct trivially copyable at 16 bytes.
struct VehicleControls {
    float   throttle{0.0f};
    float   brake{0.0f};
    float   steering{0.0f};
    uint8_t handbrake{0};
    uint8_t _pad[3]{};
};

} // namespace enigma::ecs

ENIGMA_COMPONENT(enigma::ecs::Position,       value);
ENIGMA_COMPONENT(enigma::ecs::Rotation,       value);
ENIGMA_COMPONENT(enigma::ecs::Scale,          value);
ENIGMA_COMPONENT(enigma::ecs::LocalToWorld,   matrix);
ENIGMA_COMPONENT(enigma::ecs::Velocity,       linear, angular);
ENIGMA_COMPONENT(enigma::ecs::MeshRef,        mesh_index);
ENIGMA_COMPONENT(enigma::ecs::MaterialRef,    mat_index);
ENIGMA_COMPONENT(enigma::ecs::PhysicsBody,    body_id);
ENIGMA_COMPONENT(enigma::ecs::CameraTag,      is_primary);
ENIGMA_COMPONENT(enigma::ecs::VehicleTag,     player_controlled);
ENIGMA_COMPONENT(enigma::ecs::VehicleControls,throttle, brake, steering, handbrake, _pad);
