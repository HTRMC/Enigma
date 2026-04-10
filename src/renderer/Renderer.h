#pragma once

#include "core/Math.h"
#include "core/Types.h"
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"
#include "gfx/FrameContext.h"
#include "gfx/Instance.h"
#include "gfx/GpuProfiler.h"
#include "gfx/RenderGraph.h"
#include "gfx/ShaderHotReload.h"
#include "gfx/ShaderManager.h"
#include "gfx/Swapchain.h"
#include "renderer/GBufferPass.h"
#include "renderer/LightingPass.h"
#include "renderer/MeshPass.h"
#include "renderer/RTReflectionPass.h"
#include "renderer/RTGIPass.h"
#include "renderer/RTShadowPass.h"
#include "renderer/WetRoadPass.h"
#include "renderer/Denoiser.h"
#include "renderer/TrianglePass.h"
#include "renderer/Upscaler.h"
#include "renderer/UpscalerSettings.h"

#include <volk.h>

#include <memory>

struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

namespace enigma {

class Camera;
class Window;
struct Scene;

// Configurable directional sun light. Direction need not be normalized —
// the shader normalizes it. Intensity scales lightColor in the PBR evaluation.
struct SunLight {
    vec3  direction{0.5f, 1.0f, 0.3f};
    float intensity{3.0f};
    vec3  color{1.0f, 1.0f, 1.0f};
};

class Renderer {
public:
    explicit Renderer(Window& window);
    ~Renderer();

    Renderer(const Renderer&)            = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&)                 = delete;
    Renderer& operator=(Renderer&&)      = delete;

    void drawFrame();

    // Set the active camera for rendering. Null reverts to identity.
    void setCamera(Camera* camera) { m_camera = camera; }

    // Set the scene to render. Null reverts to TrianglePass fallback.
    // On RT hardware, builds acceleration structures for the scene.
    void setScene(Scene* scene);


    // Set the directional sun light. Takes effect on the next drawFrame().
    void setLight(const SunLight& light) { m_light = light; }

    // Set wet road factor (0.0 = dry, 1.0 = standing water).
    void setWetness(f32 w) { m_wetnessFactor = w; }

    // Exposes settings to caller (Engine can wire this to a settings menu).
    void setUpscalerSettings(const UpscalerSettings& s);
    UpscalerSettings& upscalerSettings() { return m_upscalerSettings; }

    gfx::Device& device() { return *m_device; }
    gfx::Allocator& allocator() { return *m_allocator; }
    gfx::DescriptorAllocator& descriptorAllocator() { return *m_descriptorAllocator; }

private:
    void uploadCameraData();
    // Re-allocate G-buffer images and update bindless descriptors after resize.
    void resizeGBuffer(VkExtent2D extent);
    // Build BLASes for all scene primitives and the TLAS.
    void buildAccelerationStructures();

    Window& m_window;

    std::unique_ptr<gfx::Instance>             m_instance;
    std::unique_ptr<gfx::Device>               m_device;
    std::unique_ptr<gfx::Allocator>            m_allocator;
    std::unique_ptr<gfx::Swapchain>            m_swapchain;
    std::unique_ptr<gfx::FrameContextSet>      m_frames;
    std::unique_ptr<gfx::DescriptorAllocator>  m_descriptorAllocator;
    std::unique_ptr<gfx::GpuProfiler>           m_gpuProfiler;
    std::unique_ptr<gfx::RenderGraph>           m_renderGraph;
    std::unique_ptr<gfx::ShaderManager>        m_shaderManager;
    std::unique_ptr<gfx::ShaderHotReload>      m_shaderHotReload;
    std::unique_ptr<TrianglePass>              m_trianglePass;
    std::unique_ptr<MeshPass>                  m_meshPass;
    std::unique_ptr<GBufferPass>              m_gbufferPass;
    std::unique_ptr<LightingPass>             m_lightingPass;
    std::unique_ptr<RTReflectionPass>         m_rtReflectionPass;
    std::unique_ptr<RTGIPass>                m_giPass;
    std::unique_ptr<RTShadowPass>            m_shadowPass;
    std::unique_ptr<WetRoadPass>             m_wetRoadPass;
    std::unique_ptr<Denoiser>                m_giDenoiser;
    std::unique_ptr<Denoiser>                m_shadowDenoiser;
    std::unique_ptr<Denoiser>                m_reflectionDenoiser;

    std::unique_ptr<IUpscaler>               m_upscaler;
    UpscalerSettings                         m_upscalerSettings{};

    // Halton jitter state for TAA/upscaling.
    u32 m_jitterIndex = 0;

    u32 m_frameIndex = 0;

    // Camera state.
    Camera* m_camera = nullptr;
    Scene*  m_scene  = nullptr;

    SunLight m_light{};

    // Previous frame's viewProj — uploaded to the camera SSBO for motion vectors.
    mat4 m_prevViewProj{1.0f};

    // Nearest-neighbour sampler for G-buffer reads in the lighting pass.
    VkSampler m_gbufferSampler = VK_NULL_HANDLE;

    // Bindless slots for the five G-buffer textures.
    u32 m_gbufAlbedoSlot     = 0;
    u32 m_gbufNormalSlot     = 0;
    u32 m_gbufMetalRoughSlot = 0;
    u32 m_gbufMotionVecSlot  = 0;
    u32 m_gbufDepthSlot      = 0;
    u32 m_gbufferSamplerSlot = 0;

    // RT pass state.
    u32 m_tlasSlot           = 0;
    u32 m_reflectionSlot     = 0;
    u32 m_giSlot             = 0;
    u32 m_shadowSlot         = 0;
    u32 m_wetRoadSlot        = 0;
    u32 m_giDenoiseSlot      = 0;
    u32 m_shadowDenoiseSlot  = 0;
    u32 m_reflDenoiseSlot    = 0;

    f32 m_wetnessFactor      = 0.0f;

    // Per-frame camera SSBOs (one per frame-in-flight, double-buffered).
    struct CameraBuffer {
        VkBuffer      buffer     = VK_NULL_HANDLE;
        VmaAllocation allocation = nullptr;
        void*         mapped     = nullptr;
        u32           bindlessSlot = 0;
    };
    CameraBuffer m_cameraBuffers[gfx::MAX_FRAMES_IN_FLIGHT]{};
};

} // namespace enigma
