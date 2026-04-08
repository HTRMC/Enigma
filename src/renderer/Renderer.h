#pragma once

#include "core/Types.h"
#include "gfx/Device.h"
#include "gfx/Instance.h"

#include <memory>

namespace enigma {

class Window;

// The top-level graphics facade. Owns the long-lived Vulkan objects and
// drives one frame per `drawFrame()`. Construction order matches the
// Renderer's member-initializer list; destruction is reverse order.
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
    std::unique_ptr<gfx::Device>   m_device;
};

} // namespace enigma
