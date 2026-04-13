#pragma once

#include "core/Math.h"
#include "core/Types.h"
#include "physics/PhysicsInterpolation.h"
#include "physics/PhysicsWorld.h"
#include "physics/VehicleController.h"
#include "scene/Scene.h"

#include <functional>
#include <vector>

namespace enigma::ecs::systems {

// Returns a PostPhysics system that computes the interpolated vehicle transform
// and writes it to all scene nodes using their recorded rest transforms.
//
// restTransforms[i] = correction * scene->nodes[i].worldTransform captured at
// load time (GLB model-space pose with -90° Y correction applied).
// Each frame: node.worldTransform = interpolatedCarTransform * restTransforms[i].
inline std::function<void(float)> makeTransformSystem(
    PhysicsWorld&              physics,
    PhysicsInterpolation&      interp,
    VehicleController&         vehicle,
    Scene*                     scene,
    const std::vector<mat4>&   restTransforms)
{
    return [&physics, &interp, &vehicle, scene, &restTransforms](float) {
        const float alpha   = physics.accumulator() / PhysicsWorld::kFixedDt;
        const mat4  carTransform =
            interp.interpolatedTransform(vehicle.bodyId(), alpha);

        if (scene && !restTransforms.empty()) {
            for (u32 i = 0; i < static_cast<u32>(scene->nodes.size()); ++i)
                scene->nodes[i].worldTransform = carTransform * restTransforms[i];
        }
    };
}

} // namespace enigma::ecs::systems
