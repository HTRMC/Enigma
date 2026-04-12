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
#include "gfx/ImGuiLayer.h"
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
#include "renderer/AtmospherePass.h"
#include "renderer/AtmosphereSettings.h"
#include "renderer/SkyBackgroundPass.h"
#include "renderer/PostProcessPass.h"
#include "renderer/Upscaler.h"
#include "renderer/UpscalerSettings.h"
#include "physics/DeformationSystem.h"
#include "physics/PhysicsDebugRenderer.h"
#include "world/Terrain.h"

#include <volk.h>

#include <chrono>
#include <memory>
#include <vector>

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

    // Attach a GPU-driven clipmap terrain. The terrain is drawn inside
    // the same render pass as GBufferPass so it writes straight into
    // the deferred G-buffer and participates in lighting / RT effects.
    // Pass nullptr to detach.
    void setTerrain(Terrain* terrain) { m_terrain = terrain; }


    // Set the directional sun light. Takes effect on the next drawFrame().
    void setLight(const SunLight& light) { m_light = light; }

    // Set wet road factor (0.0 = dry, 1.0 = standing water).
    void setWetness(f32 w) { m_wetnessFactor = w; }

    // Apply a deformation impact to scene primitives near the event.
    void applyImpact(const ImpactEvent& event);

    // Exposes settings to caller (Engine can wire this to a settings menu).
    void setUpscalerSettings(const UpscalerSettings& s);
    UpscalerSettings& upscalerSettings() { return m_upscalerSettings; }

    gfx::Device& device() { return *m_device; }
    gfx::Allocator& allocator() { return *m_allocator; }
    gfx::DescriptorAllocator& descriptorAllocator() { return *m_descriptorAllocator; }
    gfx::ShaderManager& shaderManager() { return *m_shaderManager; }
    gfx::ShaderHotReload& shaderHotReload() { return *m_shaderHotReload; }

    // Expose the last frame's GPU timing results for ImGui display.
    // Call after drawFrame() to get the previous frame's timings.
    const std::vector<gfx::GpuProfiler::ZoneResult>& gpuTimings() const { return m_lastGpuTimings; }

    PhysicsDebugRenderer& physicsDebugRenderer() { return m_physicsDebugRenderer; }

private:
    void uploadCameraData();
    // Re-allocate G-buffer images and update bindless descriptors after resize.
    void resizeGBuffer(VkExtent2D extent);
    // Build BLASes for all scene primitives and the TLAS.
    void buildAccelerationStructures();
    // Allocate/reallocate the HDR intermediate (R16G16B16A16_SFLOAT).
    // If already allocated, idles the device and recreates at the new size.
    void createHdrColor(VkExtent2D extent);
    // Destroy HDR intermediate image + view. Does not idle the device.
    void destroyHdrColor();

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
    std::unique_ptr<gfx::ImGuiLayer>           m_imguiLayer;
    std::unique_ptr<TrianglePass>              m_trianglePass;
    std::unique_ptr<MeshPass>                  m_meshPass;
    std::unique_ptr<GBufferPass>              m_gbufferPass;
    std::unique_ptr<LightingPass>             m_lightingPass;
    std::unique_ptr<AtmospherePass>           m_atmospherePass;
    std::unique_ptr<SkyBackgroundPass>        m_skyPass;
    std::unique_ptr<PostProcessPass>          m_postProcessPass;
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

    std::vector<gfx::GpuProfiler::ZoneResult>  m_lastGpuTimings;
    bool                                        m_showImGui = true;

    // CPU frame timing.
    std::chrono::steady_clock::time_point m_lastFrameTime{};
    f32                                   m_cpuFrameTimeMs = 0.f;

    // Cached memory properties — queried once at construction, never changes.
    VkPhysicalDeviceMemoryProperties m_memoryProperties{};

    // Camera state.
    Camera*  m_camera  = nullptr;
    Scene*   m_scene   = nullptr;
    Terrain* m_terrain = nullptr;

    SunLight m_light{};

    // Atmosphere and post-process settings driven from the UI.
    AtmosphereSettings m_atmosphereSettings{};

    // Canonical sun world direction (FROM surface TO sun, unit length).
    // Computed once per frame from m_atmosphereSettings.sunAzimuth/Elevation
    // and fanned to LightingPass, RT passes, and AtmospherePass push constants.
    // No shader ever recomputes this from az/el — grep "fromAzimuthElevation"
    // should only match Renderer.cpp.
    vec3 m_sunWorldDir{0.574f, 0.574f, 0.574f}; // ~45° elevation, 135° azimuth
    bool m_sunDirty = true; // triggers LUT rebake in AtmospherePass

    // Previous frame's viewProj — uploaded to the camera SSBO for motion vectors.
    mat4 m_prevViewProj{1.0f};

    // Current frame's inverse view-projection — used by post-process depth reconstruction.
    mat4 m_invViewProj{1.0f};

    // Camera basis vectors + FOV tangents — cached for AtmospherePass AP bake.
    // Replaces passing invViewProj into the compute shader (avoids column/row-major ambiguity).
    vec3  m_cameraRight  {1.0f, 0.0f,  0.0f};
    vec3  m_cameraUp     {0.0f, 1.0f,  0.0f};
    vec3  m_cameraForward{0.0f, 0.0f, -1.0f};
    float m_tanHalfFovX  = 0.5774f; // tan(60°/2), updated each frame from camera
    float m_tanHalfFovY  = 0.5774f;

    // Camera world-space position (world units, NOT km) — cached in uploadCameraData
    // and converted to km when passed to AtmospherePass::updatePerFrame.
    vec3 m_cameraWorldPos{0.f, 0.f, 0.f};

    // Nearest-neighbour sampler for G-buffer reads in the lighting pass.
    VkSampler m_gbufferSampler = VK_NULL_HANDLE;
    // Trilinear clamp sampler for LUT / volume reads (AP volume, etc.).
    VkSampler m_linearSampler  = VK_NULL_HANDLE;

    // Bindless slots for the five G-buffer textures.
    u32 m_gbufAlbedoSlot     = 0;
    u32 m_gbufNormalSlot     = 0;
    u32 m_gbufMetalRoughSlot = 0;
    u32 m_gbufMotionVecSlot  = 0;
    u32 m_gbufDepthSlot      = 0;
    u32 m_gbufferSamplerSlot = 0;
    u32 m_linearSamplerSlot  = 0;

    // Per-effect quality toggles (Settings panel → RT conditions in drawFrame).
    bool m_rtReflectionsEnabled = true;
    bool m_rtGIEnabled          = true;
    bool m_rtShadowsEnabled     = true;
    bool m_wetRoadEnabled       = true;
    bool m_denoiseEnabled       = true;

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

    DeformationSystem    m_deformationSystem;
    bool                 m_deformationPending = false;

    PhysicsDebugRenderer m_physicsDebugRenderer;

    // HDR linear intermediate — R16G16B16A16_SFLOAT, render-extent sized.
    // All deferred passes (lighting, physics debug, sky, post-process) target
    // this buffer. The upscaler reads from it; only its output touches the
    // swapchain image.
    VkImage       m_hdrColor             = VK_NULL_HANDLE;
    VkImageView   m_hdrColorView         = VK_NULL_HANDLE;
    VmaAllocation m_hdrColorAlloc        = nullptr;
    u32           m_hdrColorSampledSlot  = UINT32_MAX;
    u32           m_hdrColorStorageSlot  = UINT32_MAX;

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
