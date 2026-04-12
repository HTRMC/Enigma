#pragma once

#include "core/Math.h"
#include "core/Types.h"

#include <volk.h>

#include <filesystem>

namespace enigma::gfx {
class Device;
class Pipeline;
class ShaderHotReload;
class ShaderManager;
} // namespace enigma::gfx

namespace enigma {

// SkyBackgroundPass
// =================
// Fullscreen pass that renders the Hillaire 2020 SkyView LUT into pixels
// with no geometry. Reverse-Z depth == 0.0 identifies sky pixels; any
// non-zero depth means geometry is present and the pixel is skipped (discard).
//
// A physically-based sun disk is added using the transmittance LUT.
//
// The pass writes to the HDR intermediate (R16G16B16A16_SFLOAT) with
// LOAD_OP_LOAD so previously rendered geometry pixels are preserved.
class SkyBackgroundPass {
public:
    explicit SkyBackgroundPass(gfx::Device& device);
    ~SkyBackgroundPass();

    SkyBackgroundPass(const SkyBackgroundPass&)            = delete;
    SkyBackgroundPass& operator=(const SkyBackgroundPass&) = delete;
    SkyBackgroundPass(SkyBackgroundPass&&)                 = delete;
    SkyBackgroundPass& operator=(SkyBackgroundPass&&)      = delete;

    void buildPipeline(gfx::ShaderManager& shaderManager,
                       VkDescriptorSetLayout globalSetLayout,
                       VkFormat colorAttachmentFormat);

    void record(VkCommandBuffer cmd,
                VkDescriptorSet globalSet,
                VkExtent2D extent,
                u32 cameraSlot,
                u32 depthSlot,
                u32 skyViewLutSlot,
                u32 transmittanceLutSlot,
                u32 samplerSlot,
                vec4 sunWorldDirIntensity,   // xyz = direction, w = intensity
                vec4 cameraWorldPosKm);      // xyz = km from planet centre

    void registerHotReload(gfx::ShaderHotReload& reloader);

private:
    void rebuildPipeline();

    gfx::Device*          m_device         = nullptr;
    gfx::Pipeline*        m_pipeline       = nullptr;

    gfx::ShaderManager*   m_shaderManager  = nullptr;
    VkDescriptorSetLayout m_globalSetLayout = VK_NULL_HANDLE;
    VkFormat              m_colorFormat     = VK_FORMAT_UNDEFINED;
    std::filesystem::path m_shaderPath;
};

} // namespace enigma
