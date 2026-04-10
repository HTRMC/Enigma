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
class ShaderHotReload;
class ShaderManager;
} // namespace enigma::gfx

namespace enigma {
struct Scene;

// GBufferPass
// ===========
// Deferred geometry pass. Renders scene geometry into four MRT colour
// targets plus a dedicated depth buffer:
//
//   [0] Albedo       VK_FORMAT_R8G8B8A8_UNORM            rgb=baseColor, a=occlusion
//   [1] Normal       VK_FORMAT_A2B10G10R10_UNORM_PACK32   rgb=world normal [0,1]
//   [2] MetalRough   VK_FORMAT_R8G8_UNORM                 r=metallic, g=roughness
//   [3] MotionVec    VK_FORMAT_R16G16_SFLOAT               rg=NDC velocity
//   [depth]          VK_FORMAT_D32_SFLOAT
//
// Call allocate() after construction and again whenever the swapchain
// is resized. After allocate(), register the image views in the
// DescriptorAllocator, then import each image into the RenderGraph.
class GBufferPass {
public:
    static constexpr VkFormat kAlbedoFormat     = VK_FORMAT_R8G8B8A8_UNORM;
    static constexpr VkFormat kNormalFormat     = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    static constexpr VkFormat kMetalRoughFormat = VK_FORMAT_R8G8_UNORM;
    static constexpr VkFormat kMotionVecFormat  = VK_FORMAT_R16G16_SFLOAT;
    static constexpr VkFormat kDepthFormat      = VK_FORMAT_D32_SFLOAT;

    explicit GBufferPass(gfx::Device& device, gfx::Allocator& allocator);
    ~GBufferPass();

    GBufferPass(const GBufferPass&)            = delete;
    GBufferPass& operator=(const GBufferPass&) = delete;
    GBufferPass(GBufferPass&&)                 = delete;
    GBufferPass& operator=(GBufferPass&&)      = delete;

    // Allocate (or re-allocate on resize) all G-buffer images.
    // Calls vkDeviceWaitIdle internally when re-allocating.
    void allocate(VkExtent2D extent);

    void buildPipeline(gfx::ShaderManager& shaderManager,
                       VkDescriptorSetLayout globalSetLayout);

    void record(VkCommandBuffer cmd,
                VkDescriptorSet globalSet,
                VkExtent2D extent,
                const Scene& scene,
                u32 cameraSlot);

    void registerHotReload(gfx::ShaderHotReload& reloader);

    // Image accessors — valid after allocate().
    VkImage     albedoImage()     const { return m_albedo.image;     }
    VkImageView albedoView()      const { return m_albedo.view;      }
    VkImage     normalImage()     const { return m_normal.image;     }
    VkImageView normalView()      const { return m_normal.view;      }
    VkImage     metalRoughImage() const { return m_metalRough.image; }
    VkImageView metalRoughView()  const { return m_metalRough.view;  }
    VkImage     motionVecImage()  const { return m_motionVec.image;  }
    VkImageView motionVecView()   const { return m_motionVec.view;   }
    VkImage     depthImage()      const { return m_depth.image;      }
    VkImageView depthView()       const { return m_depth.view;       }

private:
    struct GBufferImage {
        VkImage       image      = VK_NULL_HANDLE;
        VkImageView   view       = VK_NULL_HANDLE;
        VmaAllocation allocation = nullptr;
    };

    void createImage(VkFormat format, VkImageUsageFlags usage,
                     VkImageAspectFlags aspect, GBufferImage& out);
    void destroyImage(GBufferImage& img);
    void destroyImages();
    void rebuildPipeline();

    gfx::Device*    m_device    = nullptr;
    gfx::Allocator* m_allocator = nullptr;
    gfx::Pipeline*  m_pipeline  = nullptr;

    GBufferImage m_albedo{};
    GBufferImage m_normal{};
    GBufferImage m_metalRough{};
    GBufferImage m_motionVec{};
    GBufferImage m_depth{};

    VkExtent2D m_extent{};

    // Hot-reload state.
    gfx::ShaderManager*   m_shaderManager  = nullptr;
    VkDescriptorSetLayout m_globalSetLayout = VK_NULL_HANDLE;
    std::filesystem::path m_shaderPath;
};

} // namespace enigma
