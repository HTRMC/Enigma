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
#include "renderer/GBufferFormats.h"
#include "renderer/LightingPass.h"
#include "renderer/MeshPass.h"
#include "renderer/RTReflectionPass.h"
#include "renderer/RTGIPass.h"
#include "renderer/RTShadowPass.h"
#include "renderer/WetRoadPass.h"
#include "renderer/Denoiser.h"
#include "renderer/TrianglePass.h"
#include "renderer/AtmospherePass.h"
#include "renderer/GpuCullPass.h"
#include "renderer/GpuMeshletBuffer.h"
#include "renderer/GpuSceneBuffer.h"
#include "renderer/HiZPass.h"
#include "renderer/IndirectDrawBuffer.h"
#include "renderer/MaterialEvalPass.h"
#include "renderer/DebugVisualizationPass.h"
#include "renderer/VisibilityBufferPass.h"
#include "renderer/AtmosphereSettings.h"
#include "renderer/ClusteredForwardPass.h"
#include "renderer/SkyBackgroundPass.h"
#include "renderer/PostProcessPass.h"
#include "renderer/SMAAPass.h"
#include "renderer/micropoly/MicropolyBlasManager.h"
#include "renderer/micropoly/MicropolyConfig.h"
#include "renderer/micropoly/MicropolyCullPass.h"
#include "renderer/micropoly/MicropolyPass.h"
#include "renderer/micropoly/MicropolyRasterPass.h"
#include "renderer/micropoly/MicropolySwRasterPass.h"
#include "renderer/micropoly/MicropolyStreaming.h"
#include "ui/MicropolySettingsPanel.h"
#include "asset/MpAssetReader.h"
#include "renderer/AASettings.h"
#include "renderer/TextureFilterSettings.h"
#include "renderer/Upscaler.h"
#include "renderer/UpscalerSettings.h"
#include "physics/DeformationSystem.h"
#include "physics/PhysicsDebugRenderer.h"
#include "world/CdlodTerrain.h"
#include "world/HeightmapLoader.h"

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

// Debug visualization mode — controls the active debug view.
// All non-Lit modes bypass post-processing and write directly to the swapchain.
enum class DebugMode {
    Lit,           // Normal PBR rendering (default)
    Unlit,         // Raw G-buffer albedo
    Wireframe,     // Hardware line rasterization on black background
    LitWireframe,  // Lit scene with wireframe overlay
    DetailLighting,// Full PBR on white material (isolates lighting shape)
    Clusters,      // Meshlet colors (requires visibility buffer pipeline)
    MicropolyRasterClass, // M4.6: per-pixel R/G from rasterClassBits of the
                          // 64-bit Micropoly vis image (HW=red, SW=green,
                          // empty=black). Requires Device::supportsShaderImageInt64()
                          // and an active m_micropolyRasterPass — falls back
                          // to Lit when unavailable.
    MicropolyLodHeatmap,  // M6.1: per-pixel heatmap of the DAG node's lodLevel
                          // (blue = leaf, red = coarse). Same availability gate
                          // as MicropolyRasterClass (shaderImageInt64 + active
                          // raster pass).
    MicropolyResidencyHeatmap, // M6.1: per-pixel residency visualisation —
                          // green = page resident, magenta = page evicted,
                          // yellow = wiring bug (DAG/pageToSlot missing),
                          // black = empty vis pixel. Same availability gate.
    MicropolyBounds,      // M6.2b: per-cluster bounding-sphere wireframe
                          // overlay. Iterates the DAG, projects each
                          // sphere to screen space, and draws the outline
                          // with per-cluster hue. Availability: DAG buffer
                          // wired (does NOT require shaderImageInt64 —
                          // doesn't read the vis image).
    MicropolyBinOverflowHeat, // M6 plan §3.M6: per-pixel SW-raster tile bin
                          // fill-level heat. Black=empty, green gradient=1..
                          // BIN_CAP-1, yellow=saturated at BIN_CAP, red=any
                          // spilled entry (dominates). Availability: the
                          // m_micropolySwRasterPass is constructed (the bin
                          // buffers only exist when the SW raster path is
                          // wired). Does not read the vis image.
};

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
    Renderer(Window& window, MicropolyConfig micropolyConfig);
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

    // Access the CDLOD terrain owned by the renderer. Null until setScene()
    // has been invoked (which triggers heightmap load + terrain initialize).
    // Exposed so Application can query the heightmap for physics heightfield
    // construction.
    CdlodTerrain* cdlodTerrain() { return m_terrain.get(); }

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
    // Upload meshlet data for all scene primitives to the GPU meshlet buffer.
    void uploadMeshlets();
    // Allocate/reallocate the HDR intermediate (R16G16B16A16_SFLOAT).
    // If already allocated, idles the device and recreates at the new size.
    void createHdrColor(VkExtent2D extent);
    // Destroy HDR intermediate image + view. Does not idle the device.
    void destroyHdrColor();
    // Allocate/reallocate the SMAA LDR intermediate (swapchain format).
    void createSmaaLdr(VkExtent2D extent);
    // Destroy SMAA LDR intermediate. Does not idle the device.
    void destroySmaaLdr();

    struct GBufferImage {
        VkImage       image      = VK_NULL_HANDLE;
        VkImageView   view       = VK_NULL_HANDLE;
        VmaAllocation allocation = nullptr;
    };
    void createGBufferImage(VkFormat format, VkImageUsageFlags usage,
                            VkImageAspectFlags aspect, VkExtent2D extent, GBufferImage& out);
    void destroyGBufferImage(GBufferImage& img);
    void createGBufferImages(VkExtent2D extent);
    void destroyGBufferImages();

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
    std::unique_ptr<ClusteredForwardPass>    m_clusteredForwardPass;
    std::unique_ptr<SMAAPass>               m_smaaPass;

    // Micropoly scaffolding (M0a). Config + capability-gated pass shell.
    // record() is a no-op whenever m_micropolyConfig.enabled == false, so
    // the bit-identical-when-disabled invariant (Principle 1) holds as
    // long as nothing downstream reads m_micropolyPass state without
    // guarding on active(). The plan allows this pass to exist
    // unconditionally on all devices — the capability row selects behavior.
    MicropolyConfig                  m_micropolyConfig{};
    std::unique_ptr<MicropolyPass>   m_micropolyPass;

    // M2.4a: streaming subsystem constructed only when MicropolyConfig::enabled
    // is true AND MicropolyCapability::row != Disabled. When absent (either
    // disabled or capability-gated off) the Renderer produces BIT-IDENTICAL
    // output to pre-micropoly behavior (Principle 1). Owns an MpAssetReader
    // for the per-frame page-table lookup used by MicropolyStreaming.
    std::unique_ptr<asset::MpAssetReader>                     m_micropolyReader;
    std::unique_ptr<renderer::micropoly::MicropolyStreaming>  m_micropolyStreaming;

    // M3.2: cluster cull compute pass. Same construction gating as
    // m_micropolyStreaming — null when disabled so the Principle-1
    // bit-identity invariant holds. Owns its cull-stats + indirect-draw
    // buffers; registers both as bindless UAV slots. Hot-reload-aware.
    std::unique_ptr<renderer::micropoly::MicropolyCullPass>   m_micropolyCullPass;

    // M3.3: HW rasterisation pass (task+mesh+fragment). Null when the
    // device lacks VK_EXT_mesh_shader OR VK_EXT_shader_image_atomic_int64,
    // or when MicropolyConfig::enabled=false. Dispatched after cull,
    // reading the indirect-draw buffer the cull pass writes and emitting
    // fragments whose atomic-min updates the 64-bit vis image owned by
    // MicropolyPass.
    std::unique_ptr<renderer::micropoly::MicropolyRasterPass> m_micropolyRasterPass;

    // M4.2: SW rasteriser BINNING pass. Same capability gate as the HW
    // raster pass — null when MicropolyConfig::enabled=false or when the
    // device lacks mesh-shaders / shaderImageInt64. Dispatched after
    // MicropolyRasterPass::record() under the same enableMp branch; the
    // fragment compute that consumes the bin buffers lands in M4.3.
    std::unique_ptr<renderer::micropoly::MicropolySwRasterPass> m_micropolySwRasterPass;

    // M5a: per-asset-proxy BLAS manager for micropoly shadow casting.
    // Constructed ONLY when m_device->supportsRayTracing() == true AND
    // MicropolyConfig::enabled == true (double gate). Null on non-RT
    // devices and when micropoly is disabled. The manager owns one BLAS
    // per loaded asset, extracted from DAG level 3, and surfaces them as
    // TLAS instance entries to buildAccelerationStructures().
    std::unique_ptr<renderer::micropoly::MicropolyBlasManager> m_micropolyBlasManager;

    // AA settings — driven from the Settings ImGui panel.
    AASettings m_aaSettings{};

    // Texture-filter settings — driven from the Settings ImGui panel.  On
    // change, applyTextureFilterSettings() rebuilds the material sampler
    // and updates the bindless slot in place.
    TextureFilterSettings m_textureFilterSettings{};
    VkSampler             m_materialSampler = VK_NULL_HANDLE; // active; owned here after first apply

    // Rebuild the material sampler with the current m_textureFilterSettings
    // and rewrite the bindless slot via DescriptorAllocator::updateSampler.
    // Waits idle internally — called from the Settings panel on user change.
    void applyTextureFilterSettings();

    // SMAA LDR intermediate — swapchain format, render-extent sized.
    // PostProcessPass writes here when SMAA is active; the three SMAA passes
    // then read from it and produce the final anti-aliased swapchain output.
    VkImage       m_smaaLdrImage      = VK_NULL_HANDLE;
    VkImageView   m_smaaLdrView       = VK_NULL_HANDLE;
    VmaAllocation m_smaaLdrAlloc      = nullptr;
    u32           m_smaaLdrSampledSlot = UINT32_MAX;

    // Visibility buffer pipeline.
    std::unique_ptr<GpuSceneBuffer>          m_gpuScene;
    std::unique_ptr<GpuMeshletBuffer>        m_gpuMeshlets;
    std::unique_ptr<IndirectDrawBuffer>      m_indirectBuffer;
    std::unique_ptr<IndirectDrawBuffer>      m_terrainWireIndirectBuffer;
    std::unique_ptr<HiZPass>                 m_hizPass;
    std::unique_ptr<GpuCullPass>             m_gpuCullPass;
    std::unique_ptr<VisibilityBufferPass>    m_visibilityPass;
    std::unique_ptr<MaterialEvalPass>        m_materialEvalPass;

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

    // CDLOD terrain — owned by the renderer. Created in setScene() after the
    // scene's meshlets have been appended but before they are uploaded, so
    // GpuMeshletBuffer::reserveCapacity() can size the device buffer to cover
    // both scene meshlets and the terrain's per-patch activation ceiling.
    std::unique_ptr<HeightmapLoader> m_heightmapLoader;
    std::unique_ptr<CdlodTerrain>    m_terrain;

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

    // Bindless slots for the five G-buffer textures (sampled — read by
    // LightingPass, RT passes, Denoiser, post-process).
    u32 m_gbufAlbedoSlot     = 0;
    u32 m_gbufNormalSlot     = 0;
    u32 m_gbufMetalRoughSlot = 0;
    u32 m_gbufMotionVecSlot  = 0;
    u32 m_gbufDepthSlot      = 0;
    u32 m_gbufferSamplerSlot = 0;
    u32 m_linearSamplerSlot  = 0;

    // Storage-image slots for the four colour G-buffer targets — written by
    // MaterialEvalPass compute shader via imageStore(). Depth cannot be a
    // storage image in Vulkan, so no storage slot for depth.
    u32 m_gbufAlbedoStorageSlot     = 0;
    u32 m_gbufNormalStorageSlot     = 0;
    u32 m_gbufMetalRoughStorageSlot = 0;
    u32 m_gbufMotionVecStorageSlot  = 0;

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

    // Debug visualization.
    std::unique_ptr<DebugVisualizationPass> m_debugVisPass;
    DebugMode m_debugMode     = DebugMode::Lit;
    vec3      m_wireframeColor{1.0f, 1.0f, 1.0f};

    // M6.2a: centralised micropoly runtime knobs. Stateless — the panel
    // reads/writes through `m_micropolyConfig` + `m_debugMode` directly.
    // Visibility is driven by `m_mpSettingsPanelOpen` (ImGui idiom).
    MicropolySettingsPanel m_mpSettingsPanel{};
    bool                   m_mpSettingsPanelOpen = true;

    // HDR linear intermediate — R16G16B16A16_SFLOAT, render-extent sized.
    // All deferred passes (lighting, physics debug, sky, post-process) target
    // this buffer. The upscaler reads from it; only its output touches the
    // swapchain image.
    VkImage       m_hdrColor             = VK_NULL_HANDLE;
    VkImageView   m_hdrColorView         = VK_NULL_HANDLE;
    VmaAllocation m_hdrColorAlloc        = nullptr;
    u32           m_hdrColorSampledSlot  = UINT32_MAX;
    u32           m_hdrColorStorageSlot  = UINT32_MAX;

    GBufferImage m_gbufAlbedo{};
    GBufferImage m_gbufNormal{};
    GBufferImage m_gbufMetalRough{};
    GBufferImage m_gbufMotionVec{};
    GBufferImage m_gbufDepth{};

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
