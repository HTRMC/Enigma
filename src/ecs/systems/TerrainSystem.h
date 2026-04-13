#pragma once

#include "core/Math.h"
#include "core/Types.h"
#include "physics/PhysicsInterpolation.h"
#include "physics/PhysicsWorld.h"
#include "physics/VehicleController.h"
#include "scene/Camera.h"
#include "world/Terrain.h"

#include <cmath>
#include <functional>
#include <vector>

namespace enigma::ecs::systems {

// Returns a PreRender system that:
//   1. Updates the GPU-driven clipmap terrain chunk positions for this frame.
//   2. Streams the heightfield: rebuilds the Jolt heightfield body when the
//      vehicle moves more than rebuildDist world-units from the current
//      heightfield centre, snapping the new centre to a grid of snapGrid units.
//
// hfBodyId and hfCenter are owned by the caller (Application) and are
// mutated by the system on each rebuild.
inline std::function<void(float)> makeTerrainSystem(
    Terrain&                                  terrain,
    Camera&                                   camera,
    PhysicsWorld&                             physics,
    PhysicsInterpolation&                     interp,
    VehicleController&                        vehicle,
    std::function<float(float, float)>        heightFn,
    u32&                                      hfBodyId,
    vec2&                                     hfCenter,
    float                                     hfSize,
    u32                                       hfN,
    float                                     rebuildDist,
    float                                     snapGrid)
{
    return [&terrain, &camera, &physics, &interp, &vehicle,
            heightFn  = std::move(heightFn),
            &hfBodyId, &hfCenter,
            hfSize, hfN, rebuildDist, snapGrid](float)
    {
        // Update clipmap chunk positions (cheap pointer update).
        terrain.update(camera.position);

        // Heightfield streaming.
        const float alpha    = physics.accumulator() / PhysicsWorld::kFixedDt;
        const mat4  carT     = interp.interpolatedTransform(vehicle.bodyId(), alpha);
        const vec2  carXZ    = { carT[3].x, carT[3].z };

        if (glm::length(carXZ - hfCenter) > rebuildDist) {
            const vec2 snapped = {
                std::floor(carXZ.x / snapGrid) * snapGrid,
                std::floor(carXZ.y / snapGrid) * snapGrid,
            };

            if (hfBodyId != ~0u) physics.removeBody(hfBodyId);

            const float spacing = hfSize / static_cast<float>(hfN - 1);
            const float oriX    = snapped.x - hfSize * 0.5f;
            const float oriZ    = snapped.y - hfSize * 0.5f;

            std::vector<float> heights(static_cast<size_t>(hfN) * hfN);
            for (u32 row = 0; row < hfN; ++row)
                for (u32 col = 0; col < hfN; ++col)
                    heights[row * hfN + col] =
                        heightFn(oriX + col * spacing, oriZ + row * spacing);

            hfBodyId = physics.addHeightField(vec3(oriX, 0.0f, oriZ), hfSize, hfN, heights);
            hfCenter = snapped;
        }
    };
}

} // namespace enigma::ecs::systems
