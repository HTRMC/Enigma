#pragma once

#include "core/Math.h"
#include "core/Types.h"

#include <volk.h>

#include <filesystem>

namespace enigma::gfx {
class Device;
class Pipeline;
class ShaderManager;
class ShaderHotReload;
} // namespace enigma::gfx

namespace enigma {

struct Scene;

// ClusteredForwardPass
// ====================
// Forward-shading pass for transparent and alpha-blended geometry.
// Renders all scene primitives whose material has FLAG_BLEND set, blending
// them onto the existing HDR colour buffer with standard src-alpha blending.
//
// Uses the depth buffer from the preceding visibility buffer / GBuffer pass
// in read-only mode (depth test on, depth write off) so transparents sort
// correctly against opaque geometry without corrupting the depth buffer.
//
// Lighting model: single directional sun light (no light grid — the name
// "Clustered" is aspirational for future multi-light support).
class ClusteredForwardPass {
public:
    explicit ClusteredForwardPass(gfx::Device& device);
    ~ClusteredForwardPass();

    ClusteredForwardPass(const ClusteredForwardPass&)            = delete;
    ClusteredForwardPass& operator=(const ClusteredForwardPass&) = delete;

    // Build the alpha-blending pipeline. hdrColorFormat must match the HDR
    // intermediate image format used by the renderer (R16G16B16A16_SFLOAT).
    void buildPipeline(gfx::ShaderManager& shaderManager,
                       VkDescriptorSetLayout globalSetLayout,
                       VkFormat hdrColorFormat);

    void registerHotReload(gfx::ShaderHotReload& reloader);

    // Record transparent draws onto the HDR colour buffer.
    // hdrColorView:       view of the HDR intermediate (R16G16B16A16_SFLOAT).
    // depthView:          G-buffer depth view (D32_SFLOAT, read-only).
    // cameraSlot:         bindless slot for the CameraData SSBO.
    // materialBufferSlot: bindless slot for the scene's material array.
    // sunDir:             world-space direction FROM surface TO sun (unit).
    // sunColor:           linear RGB sun colour.
    // sunIntensity:       scale applied to sunColor in PBR evaluation.
    // Called from inside the RenderGraph execute lambda (between BeginRendering/EndRendering).
    // The graph handles layout transitions and dynamic rendering setup.
    void record(VkCommandBuffer cmd,
                VkDescriptorSet globalSet,
                VkExtent2D      extent,
                const Scene&    scene,
                u32             cameraSlot,
                u32             materialBufferSlot,
                vec3            sunDir,
                vec3            sunColor,
                float           sunIntensity);

private:
    void rebuildPipeline();

    gfx::Device*    m_device    = nullptr;
    gfx::Pipeline*  m_pipeline  = nullptr;

    gfx::ShaderManager*   m_shaderManager  = nullptr;
    VkDescriptorSetLayout m_globalSetLayout = VK_NULL_HANDLE;
    VkFormat              m_hdrColorFormat  = VK_FORMAT_R16G16B16A16_SFLOAT;

    std::filesystem::path m_shaderPath;
};

} // namespace enigma
