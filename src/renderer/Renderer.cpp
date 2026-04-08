#include "renderer/Renderer.h"

#include "platform/Window.h"

namespace enigma {

Renderer::Renderer(Window& window)
    : m_window(window) {
    (void)m_window;
}

Renderer::~Renderer() = default;

void Renderer::drawFrame() {
    // Stub at step 17. Real Vulkan frame is wired in step 38+.
}

} // namespace enigma
