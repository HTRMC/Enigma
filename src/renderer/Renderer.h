#pragma once

#include "core/Types.h"
#include "gfx/Instance.h"

#include <memory>

namespace enigma {

class Window;

// The top-level graphics facade. At step 19 it owns the Vulkan instance
// wrapper. Subsequent steps add device, swapchain, frames, pipeline, and
// the triangle pass.
class Renderer {
public:
    explicit Renderer(Window& window);
    ~Renderer();

    Renderer(const Renderer&)            = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&)                 = delete;
    Renderer& operator=(Renderer&&)      = delete;

    void drawFrame();

private:
    Window& m_window;

    std::unique_ptr<gfx::Instance> m_instance;
};

} // namespace enigma
