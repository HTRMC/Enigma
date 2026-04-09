#pragma once

#include "core/Types.h"

namespace enigma {

class Camera;
class Input;

// First-person fly controller (UE5 editor-style).
//
//   RMB hold  → capture cursor, enable look
//   WASD      → move relative to camera orientation
//   Q / E     → world-space down / up
//   Shift     → 3× speed boost
//   Mouse X/Y → yaw / pitch (pitch clamped ±89°)
//
// All movement is delta-time multiplied for frame-rate independence.
class CameraController {
public:
    CameraController(Camera& camera, Input& input);

    void update(f32 deltaTime);

private:
    Camera* m_camera = nullptr;
    Input*  m_input  = nullptr;
    f32     m_pitch  = 0.0f; // accumulated pitch in degrees
    f32     m_yaw    = 0.0f; // accumulated yaw in degrees
};

} // namespace enigma
