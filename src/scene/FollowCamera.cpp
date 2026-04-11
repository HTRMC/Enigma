#include "scene/FollowCamera.h"

#include "scene/Camera.h"

#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

namespace enigma {

FollowCamera::FollowCamera(Camera& camera, f32 armLength, f32 heightOffset)
    : m_camera(&camera)
    , m_armLength(armLength)
    , m_heightOffset(heightOffset) {}

void FollowCamera::update(const mat4& targetTransform, f32 dt) {
    // Car world-space origin from the translation column.
    const vec3 carPos = vec3(targetTransform[3]);

    // Car local +Z rotated into world space — the forward direction.
    vec3 carForward = vec3(targetTransform[2]);
    const f32 fwdLen = glm::length(carForward);
    if (fwdLen > 1e-6f) {
        carForward /= fwdLen;
    } else {
        carForward = vec3(0.0f, 0.0f, 1.0f);
    }

    // Ideal camera position: behind and above the car, offset along -forward.
    const vec3 ideal = carPos
                     - carForward * m_armLength
                     + vec3(0.0f, m_heightOffset, 0.0f);

    // Exponential smoothing (frame-rate independent).
    const f32 alpha = 1.0f - std::exp(-8.0f * dt);
    m_smoothPos = glm::mix(m_smoothPos, ideal, alpha);

    // Look target: slightly above the car's pivot so the pitch is natural.
    const vec3 lookTarget = carPos + vec3(0.0f, 0.5f, 0.0f);

    // Compute view-space look direction and derive a quaternion. Camera's
    // canonical forward is -Z, so the rotation maps (0,0,-1) → lookDir.
    vec3 lookDir = lookTarget - m_smoothPos;
    const f32 lookLen = glm::length(lookDir);
    if (lookLen > 1e-6f) {
        lookDir /= lookLen;
    } else {
        lookDir = vec3(0.0f, 0.0f, -1.0f);
    }

    // Build a view matrix using glm::lookAt, then extract its rotation.
    // lookAt() returns the view matrix (world → view), so the camera
    // orientation is the inverse of its upper-3x3 rotation.
    const mat4 view = glm::lookAt(m_smoothPos, lookTarget, vec3(0.0f, 1.0f, 0.0f));
    const mat3 rot  = glm::transpose(mat3(view)); // view→world rotation
    m_camera->orientation = glm::normalize(glm::quat_cast(rot));
    m_camera->position    = m_smoothPos;
}

} // namespace enigma
