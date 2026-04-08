#pragma once

#include "core/Types.h"
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"
#include "gfx/FrameContext.h"
#include "gfx/Instance.h"
#include "gfx/ShaderHotReload.h"
#include "gfx/ShaderManager.h"
#include "gfx/Swapchain.h"
#include "renderer/TrianglePass.h"

#include <memory>

namespace enigma {

class Window;

// The top-level graphics facade. Owns the long-lived Vulkan objects and
// drives one frame per `drawFrame()`. Construction order matches the
// Renderer's member-initializer list; destruction is reverse order so
// the bottom of the stack (Instance) outlives everything that depends
// on it.
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

    std::unique_ptr<gfx::Instance>             m_instance;
    std::unique_ptr<gfx::Device>               m_device;
    std::unique_ptr<gfx::Allocator>            m_allocator;
    std::unique_ptr<gfx::Swapchain>            m_swapchain;
    std::unique_ptr<gfx::FrameContextSet>      m_frames;
    std::unique_ptr<gfx::DescriptorAllocator>  m_descriptorAllocator;
    std::unique_ptr<gfx::ShaderManager>        m_shaderManager;
    std::unique_ptr<gfx::ShaderHotReload>      m_shaderHotReload;
    std::unique_ptr<TrianglePass>              m_trianglePass;

    u32 m_frameIndex = 0;
};

} // namespace enigma
