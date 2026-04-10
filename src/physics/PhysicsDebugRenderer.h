#pragma once

#include "core/Types.h"

#include <volk.h>

namespace enigma {

// Stub debug renderer that will eventually collect wireframe line segments
// from Jolt's debug draw interface and render them as a debug overlay.
class PhysicsDebugRenderer {
public:
    PhysicsDebugRenderer()  = default;
    ~PhysicsDebugRenderer() = default;

    // Draw collected debug lines into the given command buffer.
    void drawFrame(VkCommandBuffer cmd);
};

} // namespace enigma
