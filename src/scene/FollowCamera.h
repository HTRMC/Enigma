#pragma once

#include "core/Math.h"
#include "core/Types.h"

namespace enigma {

class Camera;

// Spring-arm third-person follow camera.
//
// Each frame, update() takes the target (car) world transform and
// computes an ideal camera position behind + above the car along its
// local forward axis, then exponentially smooths toward that pose.
// The camera's quaternion orientation is set directly so it always
// looks at the car's slightly-raised pivot.
class FollowCamera {
public:
    FollowCamera(Camera& camera, f32 armLength = 8.0f, f32 heightOffset = 2.5f);

    // Call each frame with the interpolated world transform of the car body.
    void update(const mat4& targetTransform, f32 dt);

private:
    Camera* m_camera;
    f32     m_armLength;
    f32     m_heightOffset;
    vec3    m_smoothPos{0.0f, 3.0f, 8.0f}; // starts behind spawn position
};

} // namespace enigma
