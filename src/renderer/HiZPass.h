#pragma once

#include "core/Types.h"

#include <volk.h>

#include <filesystem>
#include <vector>

// Forward declare VMA handles.
struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T*;
struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

namespace enigma::gfx {
class Device;
class Allocator;
class DescriptorAllocator;
class Pipeline;
class ShaderManager;
class ShaderHotReload;
} // namespace enigma::gfx

namespace enigma {

// HiZPass
// =======
// Builds a hierarchical Z mip-chain from the depth buffer for occlusion
// culling. Each mip is half the resolution of the previous one; values are
// the MINIMUM depth of the covered 2x2 block (conservative for reverse-Z).
//
// The HiZ image is R32_SFLOAT (separate from the D32_SFLOAT depth buffer).
// Mip 0 must be populated externally (copy from depth) before record().
// record() builds mips 1..N-1 by dispatching hiz_build.comp.hlsl per level.
//
// All mip views are registered as storage images (binding 1) so the
// hiz_build compute shader can access them as RWTexture2D<float>.
class HiZPass {
public:
    HiZPass(gfx::Device& device, gfx::Allocator& allocator,
            gfx::DescriptorAllocator& descriptors);
    ~HiZPass();

    HiZPass(const HiZPass&)            = delete;
    HiZPass& operator=(const HiZPass&) = delete;

    // Allocate the HiZ image at the given resolution. Registers all mip
    // views as storage images. Call when the render extent changes.
    void allocate(VkExtent2D extent);

    // Build the pipeline. Must be called before record().
    void buildPipeline(gfx::ShaderManager& shaderManager,
                       VkDescriptorSetLayout globalSetLayout);

    void registerHotReload(gfx::ShaderHotReload& reloader);

    // Record mip downsamples for levels 1..mip_count-1.
    // Assumes mip 0 already contains depth data (caller populated it).
    void record(VkCommandBuffer cmd, VkDescriptorSet globalSet);

    // Bindless storage image slot for a given HiZ mip level (binding 1).
    u32 mip_slot(u32 level) const;

    u32   mip_count()  const { return static_cast<u32>(m_mip_slots.size()); }
    VkImage image()    const { return m_image; }

private:
    void destroyImage();
    void rebuildPipeline();

    gfx::Device*              m_device      = nullptr;
    gfx::Allocator*           m_allocator   = nullptr;
    gfx::DescriptorAllocator* m_descriptors = nullptr;
    gfx::Pipeline*            m_pipeline    = nullptr;
    gfx::ShaderManager*       m_shaderManager  = nullptr;
    VkDescriptorSetLayout     m_globalSetLayout = VK_NULL_HANDLE;
    std::filesystem::path     m_shaderPath;

    VkImage       m_image = VK_NULL_HANDLE;
    VmaAllocation m_alloc = nullptr;

    // Per-mip views and bindless storage image slots.
    std::vector<VkImageView> m_mip_views;
    std::vector<u32>         m_mip_slots;

    VkExtent2D m_extent{};
};

} // namespace enigma
