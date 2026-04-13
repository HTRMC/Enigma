#pragma once

#include "ecs/World.h"
#include "ecs/Components.h"
#include "physics/VehicleController.h"

#include <functional>

namespace enigma::ecs::systems {

// Returns a Physics system that reads VehicleControls from every entity with
// VehicleTag + VehicleControls and forwards the input to VehicleController,
// then calls VehicleController::update(dt).
// Must run before makePhysicsSystem so the impulses are consumed in the
// same fixed-timestep sub-steps.
inline std::function<void(float)> makeVehicleSystem(VehicleController& vehicle, World& world) {
    return [&vehicle, &world](float dt) {
        world.query<VehicleTag, VehicleControls>().for_each(
            [&vehicle, dt](Entity, VehicleTag&, VehicleControls& vc) {
                VehicleInput vi{};
                vi.throttle  = vc.throttle;
                vi.brake     = vc.brake;
                vi.steering  = vc.steering;
                vi.handbrake = vc.handbrake != 0;
                vehicle.setInput(vi);
                vehicle.update(dt);
            });
    };
}

} // namespace enigma::ecs::systems
