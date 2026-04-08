#include "renderer/Renderer.h"

#include "core/Log.h"
#include "gfx/Instance.h"
#include "platform/Window.h"

namespace enigma {

Renderer::Renderer(Window& window)
    : m_window(window)
    , m_instance(std::make_unique<gfx::Instance>()) {
    (void)m_window;
    ENIGMA_LOG_INFO("[renderer] constructed");
}

Renderer::~Renderer() {
    ENIGMA_LOG_INFO("[renderer] shutdown");
}

void Renderer::drawFrame() {
    // Real Vulkan frame is wired at step 38+.
}

} // namespace enigma
