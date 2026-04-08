#pragma once

#include "core/Types.h"

namespace enigma {

class Window;

// The top-level graphics facade. At step 17 this is a stub — construction
// takes a Window reference and `drawFrame()` is a no-op. Real Vulkan
// subsystem composition arrives at step 38.
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
};

} // namespace enigma
