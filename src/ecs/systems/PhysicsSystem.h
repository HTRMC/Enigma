#pragma once

#include "physics/PhysicsInterpolation.h"
#include "physics/PhysicsWorld.h"
#include "physics/VehicleController.h"

#include <functional>

namespace enigma::ecs::systems {

// Returns a Physics system that drives the fixed-timestep sub-step loop.
// Per the plan: ExecutionPolicy::Sequential on the main thread.
// Jolt's internal JPH::JobSystemThreadPool(2) handles parallelism inside
// stepFixed(); ECS must NOT run this system in parallel with other systems.
//
// Sequence each tick:
//   1. Accumulate dt.
//   2. For each fixed step: snapshot interpolation state, then step.
//   3. Update the "current" interpolation snapshot after all steps.
inline std::function<void(float)> makePhysicsSystem(PhysicsWorld&         physics,
                                                     PhysicsInterpolation& interp,
                                                     VehicleController&    vehicle) {
    return [&physics, &interp, &vehicle](float dt) {
        physics.addDt(dt);
        while (physics.canStep()) {
            interp.snapshot(vehicle.bodyId(), physics);
            physics.stepFixed();
        }
        interp.updateCurr(vehicle.bodyId(), physics);
    };
}

} // namespace enigma::ecs::systems
