#pragma once

#include "ecs/World.h"
#include "ecs/Components.h"
#include "input/Input.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <functional>

namespace enigma::ecs::systems {

// Returns a PrePhysics system that reads keyboard and gamepad state and writes
// VehicleControls to every entity with VehicleTag + VehicleControls.
// Resets controls to zero each tick before reading, so releasing a key
// immediately zeroes the corresponding axis.
inline std::function<void(float)> makeInputSystem(Input& input, World& world) {
    return [&input, &world](float) {
        world.query<VehicleTag, VehicleControls>().for_each(
            [&input](Entity, VehicleTag&, VehicleControls& vc) {
                vc = {};

                // Keyboard
                if (input.isKeyDown(GLFW_KEY_W)) vc.throttle = 1.0f;
                if (input.isKeyDown(GLFW_KEY_S)) vc.brake    = 1.0f;
                if (input.isKeyDown(GLFW_KEY_A)) vc.steering = -0.7f;
                if (input.isKeyDown(GLFW_KEY_D)) vc.steering =  0.7f;
                if (input.isKeyDown(GLFW_KEY_SPACE)) vc.handbrake = 1;

                // Gamepad override (if connected).
                if (input.isGamepadPresent()) {
                    const float gpThrottle =
                        input.getGamepadAxis(0, Input::GAMEPAD_AXIS_RIGHT_TRIGGER);
                    const float gpBrake =
                        input.getGamepadAxis(0, Input::GAMEPAD_AXIS_LEFT_TRIGGER);
                    const float gpSteer =
                        input.getGamepadAxis(0, Input::GAMEPAD_AXIS_LEFT_X);

                    // Triggers are in [-1,1]; remap to [0,1].
                    if (gpThrottle > 0.0f)
                        vc.throttle = std::max(vc.throttle, (gpThrottle + 1.0f) * 0.5f);
                    if (gpBrake > 0.0f)
                        vc.brake    = std::max(vc.brake,    (gpBrake    + 1.0f) * 0.5f);
                    if (std::abs(gpSteer) > 0.1f)
                        vc.steering = gpSteer;
                }
            });
    };
}

} // namespace enigma::ecs::systems
