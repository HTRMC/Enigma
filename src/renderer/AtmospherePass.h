#pragma once

#include "core/Math.h"
#include "core/Types.h"
#include "renderer/AtmosphereSettings.h"

#include <volk.h>
#include <filesystem>

struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

namespace enigma {
struct CameraData;
} // forward decl (used in updatePerFrame)

namespace enigma::gfx {
class Allocator;
class Device;
class DescriptorAllocator;
class Pipeline;
class ShaderManager;
} // namespace enigma::gfx

namespace enigma {

// AtmospherePass
// ==============
// Manages the four Hillaire 2020 LUT images and their compute dispatches.
//
// Lifetime:
//   init()              — called once after device/allocator creation
//   bakeStaticLUTs(cmd) — (re)builds Transmittance + MultiScatter LUTs
//                         on startup and whenever m_sunDirty is set; caller
//                         must surround with vkDeviceWaitIdle
//   updatePerFrame(cmd) — dispatches SkyView + AerialPerspective every frame
//   shutdown()          — destroys all Vulkan objects
//
// LUT formats (matching kLutFormat2D / kLutFormat3D constants):
//   Transmittance  256×64   VK_FORMAT_B10G11R11_UFLOAT_PACK32
//   MultiScatter    32×32   VK_FORMAT_B10G11R11_UFLOAT_PACK32
//   SkyView        192×108  VK_FORMAT_B10G11R11_UFLOAT_PACK32
//   AerialPersp  32×32×32   VK_FORMAT_R16G16B16A16_SFLOAT
class AtmospherePass {
public:
    static constexpr VkFormat    kLutFormat2D = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    static constexpr VkFormat    kLutFormat3D = VK_FORMAT_R16G16B16A16_SFLOAT;
    static constexpr VkExtent2D  kTransmittanceSize{256, 64};
    static constexpr VkExtent2D  kMultiScatterSize{32, 32};
    static constexpr VkExtent2D  kSkyViewSize{192, 108};
    static constexpr VkExtent3D  kAerialPerspectiveSize{32, 32, 32};

    struct InitInfo {
        gfx::Device*              device              = nullptr;
        gfx::Allocator*           allocator           = nullptr;
        gfx::DescriptorAllocator* descriptorAllocator = nullptr;
        gfx::ShaderManager*       shaderManager       = nullptr;
        VkDescriptorSetLayout     globalSetLayout     = VK_NULL_HANDLE;
    };

    void init(const InitInfo& info);

    // Bakes the two static LUTs (Transmittance, MultiScatter).
    // Call when sun changes (guarded by vkDeviceWaitIdle in Renderer).
    void bakeStaticLUTs(VkCommandBuffer cmd,
                        const AtmosphereSettings& settings,
                        const vec3& sunWorldDir,
                        u32 samplerSlot);

    // Per-frame: dispatches SkyView LUT + Aerial Perspective volume.
    void updatePerFrame(VkCommandBuffer cmd,
                        const AtmosphereSettings& settings,
                        const vec3& sunWorldDir,
                        const vec3& cameraWorldPosKm,
                        const mat4& invViewProj,
                        u32 samplerSlot);

    void shutdown();

    // Bindless sampled-image slots for 2D LUTs (used in shader push constants)
    u32 transmittanceLutSlot() const { return m_tlutSampledSlot; }
    u32 multiScatterLutSlot()  const { return m_msLutSampledSlot; }
    u32 skyViewLutSlot()       const { return m_svLutSampledSlot; }

    // Dedicated descriptor sets for the 3D Aerial Perspective volume.
    // PostProcessPass binds m_apReadSet at set=1 when applying AP.
    VkDescriptorSet aerialPerspectiveWriteSet() const { return m_apWriteSet; }
    VkDescriptorSet aerialPerspectiveReadSet()  const { return m_apReadSet; }

    // The pipeline layout used for the AP dispatch (global set 0 + AP set 1).
    // PostProcessPass needs this to bind the read set.
    VkDescriptorSetLayout aerialPerspectiveReadSetLayout() const { return m_apReadSetLayout; }

private:
    void createLut2D(VkExtent2D size, VkFormat fmt,
                     VkImage& outImg, VkImageView& outView,
                     VmaAllocation& outAlloc,
                     u32& outSampledSlot, u32& outStorageSlot);

    void createLut3D();
    void destroyLut3D();

    void transitionImage(VkCommandBuffer cmd, VkImage img,
                         VkImageLayout from, VkImageLayout to,
                         VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                         VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess);

    gfx::Device*              m_device              = nullptr;
    gfx::Allocator*           m_allocator           = nullptr;
    gfx::DescriptorAllocator* m_descriptorAllocator = nullptr;
    gfx::ShaderManager*       m_shaderManager       = nullptr;
    VkDescriptorSetLayout     m_globalSetLayout      = VK_NULL_HANDLE;

    // 2D LUT images
    VkImage       m_transmittanceImg  = VK_NULL_HANDLE;
    VkImageView   m_transmittanceView = VK_NULL_HANDLE;
    VmaAllocation m_transmittanceAlloc= nullptr;
    u32 m_tlutSampledSlot  = UINT32_MAX;
    u32 m_tlutStorageSlot  = UINT32_MAX;

    VkImage       m_multiScatterImg   = VK_NULL_HANDLE;
    VkImageView   m_multiScatterView  = VK_NULL_HANDLE;
    VmaAllocation m_multiScatterAlloc = nullptr;
    u32 m_msLutSampledSlot = UINT32_MAX;
    u32 m_msLutStorageSlot = UINT32_MAX;

    VkImage       m_skyViewImg   = VK_NULL_HANDLE;
    VkImageView   m_skyViewView  = VK_NULL_HANDLE;
    VmaAllocation m_skyViewAlloc = nullptr;
    u32 m_svLutSampledSlot = UINT32_MAX;
    u32 m_svLutStorageSlot = UINT32_MAX;

    // 3D Aerial Perspective volume
    VkImage       m_apImg   = VK_NULL_HANDLE;
    VkImageView   m_apView  = VK_NULL_HANDLE;
    VmaAllocation m_apAlloc = nullptr;

    // Descriptor infrastructure for AP set 1
    VkDescriptorSetLayout m_apWriteSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_apReadSetLayout  = VK_NULL_HANDLE;
    VkDescriptorPool      m_apPool           = VK_NULL_HANDLE;
    VkDescriptorSet       m_apWriteSet       = VK_NULL_HANDLE;
    VkDescriptorSet       m_apReadSet        = VK_NULL_HANDLE;

    // Compute pipelines (2D LUTs use gfx::Pipeline for layout+pipeline in one)
    gfx::Pipeline* m_transmittancePipe = nullptr;
    gfx::Pipeline* m_multiScatterPipe  = nullptr;
    gfx::Pipeline* m_skyViewPipe       = nullptr;

    // AP pipeline: needs 2 descriptor sets so we manage layout+pipeline manually
    VkPipelineLayout m_apPipeLayout = VK_NULL_HANDLE;
    VkPipeline       m_apPipe       = VK_NULL_HANDLE;
};

} // namespace enigma
