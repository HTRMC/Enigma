#pragma once

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

// WetRoadPass
// ===========
// Wet tarmac reflects headlights and environment via RT on Recommended+
// tier. Falls back to screen-space planar reflection on Min tier.
class WetRoadPass {
public:
    explicit WetRoadPass(gfx::Device& device, gfx::Allocator& allocator);
    ~WetRoadPass();

    WetRoadPass(const WetRoadPass&)            = delete;
    WetRoadPass& operator=(const WetRoadPass&) = delete;
    WetRoadPass(WetRoadPass&&)                 = delete;
    WetRoadPass& operator=(WetRoadPass&&)      = delete;

    void buildPipeline(gfx::ShaderManager& shaderManager,
                       VkDescriptorSetLayout globalSetLayout);

    void allocate(VkExtent2D extent);

    void record(VkCommandBuffer cmd,
                VkDescriptorSet globalSet,
                VkExtent2D extent,
                u32 normalSlot,
                u32 depthSlot,
                u32 cameraSlot,
                u32 tlasSlot,
                u32 outputSlot,
                f32 wetnessFactor);

    void registerHotReload(gfx::ShaderHotReload& reloader);

    VkImage     outputImage() const { return m_output.image; }
    VkImageView outputView()  const { return m_output.view; }

    // Bindless storage image slot.
    u32 outputSlot = 0xFFFFFFFFu;

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

    // Screen-space planar reflection fallback.
    gfx::Pipeline* m_fallbackPipeline = nullptr;

    OutputImage   m_output{};
    VkExtent2D    m_extent{};

    // Hot-reload state.
    gfx::ShaderManager*   m_shaderManager  = nullptr;
    VkDescriptorSetLayout m_globalSetLayout = VK_NULL_HANDLE;
    std::filesystem::path m_rgenPath;
    std::filesystem::path m_rchitPath;
    std::filesystem::path m_rmissPath;
    std::filesystem::path m_fallbackPath;
};

} // namespace enigma
