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
class ShaderHotReload;
class ShaderManager;
} // namespace enigma::gfx

namespace enigma {

// Denoiser
// ========
// Spatial-temporal denoiser shared by all RT effects (GI, shadows,
// wet-road reflections). Uses motion vectors from G-buffer for temporal
// reprojection. Two compute passes: spatial (A-trous wavelet, 5
// iterations) then temporal (accumulation with neighborhood clamping).
class Denoiser {
public:
    explicit Denoiser(gfx::Device& device, gfx::Allocator& allocator);
    ~Denoiser();

    Denoiser(const Denoiser&)            = delete;
    Denoiser& operator=(const Denoiser&) = delete;
    Denoiser(Denoiser&&)                 = delete;
    Denoiser& operator=(Denoiser&&)      = delete;

    // effectFormat: input/output image format (VK_FORMAT_R16G16B16A16_SFLOAT or R16_SFLOAT).
    void buildPipelines(gfx::ShaderManager& shaderManager,
                        VkDescriptorSetLayout globalSetLayout,
                        VkFormat effectFormat);

    void allocate(VkExtent2D extent, VkFormat effectFormat);

    // Denoise inputSlot into output storage image.
    // motionVecSlot: G-buffer motion vectors for temporal reprojection.
    // historySlot: previous frame's denoised output (for temporal accumulation).
    void record(VkCommandBuffer cmd,
                VkDescriptorSet globalSet,
                VkExtent2D extent,
                u32 inputSlot,
                u32 motionVecSlot,
                u32 historySlot,
                u32 outputSlot);

    void registerHotReload(gfx::ShaderHotReload& reloader);

    VkImage     outputImage() const { return m_output.image; }
    VkImageView outputView()  const { return m_output.view; }

    // Denoised result storage image slot.
    u32 outputSlot = 0xFFFFFFFFu;

private:
    struct OutputImage {
        VkImage       image      = VK_NULL_HANDLE;
        VkImageView   view       = VK_NULL_HANDLE;
        VmaAllocation allocation = nullptr;
    };

    void destroyOutput();

    gfx::Device*    m_device    = nullptr;
    gfx::Allocator* m_allocator = nullptr;

    gfx::ShaderManager*   m_shaderManager  = nullptr;
    VkDescriptorSetLayout m_globalSetLayout = VK_NULL_HANDLE;
    VkFormat              m_format          = VK_FORMAT_R16G16B16A16_SFLOAT;

    gfx::Pipeline* m_spatialPipeline  = nullptr; // compute
    gfx::Pipeline* m_temporalPipeline = nullptr; // compute

    OutputImage m_output{};

    std::filesystem::path m_spatialPath;
    std::filesystem::path m_temporalPath;
};

} // namespace enigma
