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

// RTShadowPass
// =============
// Trace shadow rays from surface to directional sun light. Soft penumbra
// via cone angle sampling. Falls back to cascaded shadow maps (CSM)
// raster pass on Min tier.
class RTShadowPass {
public:
    explicit RTShadowPass(gfx::Device& device, gfx::Allocator& allocator);
    ~RTShadowPass();

    RTShadowPass(const RTShadowPass&)            = delete;
    RTShadowPass& operator=(const RTShadowPass&) = delete;
    RTShadowPass(RTShadowPass&&)                 = delete;
    RTShadowPass& operator=(RTShadowPass&&)      = delete;

    void buildPipeline(gfx::ShaderManager& shaderManager,
                       VkDescriptorSetLayout globalSetLayout);

    void allocate(VkExtent2D extent);

    // lightDirIntensity.xyz = sun direction, .w = soft shadow cone half-angle (radians)
    void record(VkCommandBuffer cmd,
                VkDescriptorSet globalSet,
                VkExtent2D extent,
                u32 normalSlot,
                u32 depthSlot,
                u32 cameraSlot,
                u32 tlasSlot,
                u32 outputSlot,
                vec4 lightDirIntensity);

    void registerHotReload(gfx::ShaderHotReload& reloader);

    VkImage     outputImage() const { return m_output.image; }
    VkImageView outputView()  const { return m_output.view; }

    // Bindless storage image slot (R16F shadow term).
    u32 outputSlot = 0xFFFFFFFFu;

    static constexpr VkFormat kOutputFormat = VK_FORMAT_R16_SFLOAT;

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

    // CSM fallback path.
    gfx::Pipeline* m_fallbackPipeline = nullptr;

    OutputImage   m_output{};
    VkExtent2D    m_extent{};

    // Hot-reload state.
    gfx::ShaderManager*   m_shaderManager  = nullptr;
    VkDescriptorSetLayout m_globalSetLayout = VK_NULL_HANDLE;
    std::filesystem::path m_rgenPath;
    std::filesystem::path m_rmissPath;
    std::filesystem::path m_csmPath;
};

} // namespace enigma
