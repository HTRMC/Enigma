#pragma once

#include "core/Math.h"
#include "core/Types.h"

#include <volk.h>

#include <filesystem>

struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

namespace enigma::gfx {
class Allocator;
class Device;
class Pipeline;
class RTPipeline;
class ShaderHotReload;
class ShaderManager;
} // namespace enigma::gfx

namespace enigma {

// RTReflectionPass
// ================
// Dispatches RT reflections on hardware with ray tracing support,
// or falls back to a screen-space reflection (SSR) raster pass on
// Min-tier GPUs. Writes to an RGBA16F storage image that the
// lighting pass can composite.
class RTReflectionPass {
public:
    explicit RTReflectionPass(gfx::Device& device, gfx::Allocator& allocator);
    ~RTReflectionPass();

    RTReflectionPass(const RTReflectionPass&)            = delete;
    RTReflectionPass& operator=(const RTReflectionPass&) = delete;
    RTReflectionPass(RTReflectionPass&&)                 = delete;
    RTReflectionPass& operator=(RTReflectionPass&&)      = delete;

    // Build the RT pipeline (or SSR raster pipeline on Min tier).
    void buildPipeline(gfx::ShaderManager& shaderManager,
                       VkDescriptorSetLayout globalSetLayout,
                       VkFormat outputFormat);

    // Allocate (or re-allocate) the RGBA16F reflection output image.
    void allocate(VkExtent2D extent);

    // Record the pass into a command buffer.
    // On RT hardware: vkCmdTraceRaysKHR.
    // On Min tier: fullscreen SSR raster pass.
    void record(VkCommandBuffer cmd,
                VkDescriptorSet globalSet,
                VkExtent2D extent,
                u32 normalSlot,
                u32 depthSlot,
                u32 cameraSlot,
                u32 samplerSlot,
                u32 tlasSlot,
                u32 outputSlot,
                u32 skyViewLutSlot,
                u32 transmittanceLutSlot,
                vec4 sunWorldDirIntensity,  // xyz = sun dir, w = intensity
                vec4 cameraWorldPosKm);     // xyz = km from planet centre

    void registerHotReload(gfx::ShaderHotReload& reloader);

    VkImage     outputImage() const { return m_output.image; }
    VkImageView outputView()  const { return m_output.view; }

    // Bindless storage image slot for the reflection output.
    u32 outputSlot = 0;

    static constexpr VkFormat kOutputFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

private:
    struct OutputImage {
        VkImage       image      = VK_NULL_HANDLE;
        VkImageView   view       = VK_NULL_HANDLE;
        VmaAllocation allocation = nullptr;
    };

    void destroyOutput();

    gfx::Device*    m_device    = nullptr;
    gfx::Allocator* m_allocator = nullptr;
    bool            m_useRT     = false;

    // RT path.
    gfx::RTPipeline* m_rtPipeline = nullptr;

    // SSR raster fallback path.
    gfx::Pipeline* m_ssrPipeline = nullptr;

    OutputImage   m_output{};
    VkExtent2D    m_extent{};

    // Hot-reload state.
    gfx::ShaderManager*   m_shaderManager  = nullptr;
    VkDescriptorSetLayout m_globalSetLayout = VK_NULL_HANDLE;
    std::filesystem::path m_rgenPath;
    std::filesystem::path m_rchitPath;
    std::filesystem::path m_rmissPath;
    std::filesystem::path m_ssrPath;
};

} // namespace enigma
