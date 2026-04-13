#pragma once

#include "physics/PhysicsInterpolation.h"
#include "physics/PhysicsWorld.h"
#include "physics/VehicleController.h"
#include "scene/FollowCamera.h"

#include <functional>

namespace enigma::ecs::systems {

// Returns a PreRender system that updates the spring-arm follow camera to
// track the interpolated vehicle transform.
inline std::function<void(float)> makeCameraSystem(FollowCamera&         followCam,
                                                    PhysicsWorld&         physics,
                                                    PhysicsInterpolation& interp,
                                                    VehicleController&    vehicle) {
    return [&followCam, &physics, &interp, &vehicle](float dt) {
        const float alpha        = physics.accumulator() / PhysicsWorld::kFixedDt;
        const mat4  carTransform = interp.interpolatedTransform(vehicle.bodyId(), alpha);
        followCam.update(carTransform, dt);
    };
}

} // namespace enigma::ecs::systems
