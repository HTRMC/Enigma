#pragma once

#include "core/Math.h"
#include "core/Types.h"

#include <volk.h>

#include <filesystem>
#include <vector>

struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

namespace enigma::gfx {
class Allocator;
class DescriptorAllocator;
class Device;
} // namespace enigma::gfx

namespace enigma {

// Describes the on-disk heightmap and the world-space footprint it covers.
// Two binary formats are supported:
//   - .raw  : R16LE — each uint16 is remapped [0,65535] -> [minHeight,maxHeight]
//   - .r32f : raw float32 height in meters (min/max used only for logging)
struct HeightmapDesc {
    std::filesystem::path path;      // .raw (R16LE) or .r32f binary
    u32 sampleCount = 4097;          // samples per side (NxN; 4096+1 for shared edges)
    f32 worldSize   = 4096.0f;       // meters covered (4 km)
    f32 minHeight   = 0.0f;          // for .raw: remap [0,65535] -> [min,max]
    f32 maxHeight   = 512.0f;
};

// HeightmapLoader
// ===============
// Loads a single heightmap tile into both a CPU float array (for physics
// and CPU queries) and a GPU R32_SFLOAT sampled image (for shaders),
// producing a single source of truth for terrain elevation.
//
// Lifecycle:
//   1. Construct with device/allocator/descriptors.
//   2. Call `load(desc, uploadCmd)` where `uploadCmd` is a primary command
//      buffer owned by the caller in the recording state. The loader
//      records a staging->image copy plus layout transitions. The caller
//      submits and waits on the command buffer; once the GPU signals
//      completion, the caller invokes `releaseStaging()` to free the
//      host-visible staging buffer.
//   3. Query heights / bindless slots until destruction.
class HeightmapLoader {
public:
    HeightmapLoader(gfx::Device& device,
                    gfx::Allocator& allocator,
                    gfx::DescriptorAllocator& descriptors);
    ~HeightmapLoader();

    HeightmapLoader(const HeightmapLoader&)            = delete;
    HeightmapLoader& operator=(const HeightmapLoader&) = delete;
    HeightmapLoader(HeightmapLoader&&)                 = delete;
    HeightmapLoader& operator=(HeightmapLoader&&)      = delete;

    // Blocking load + GPU upload recording. Must be called once at startup
    // on the main thread with `uploadCmd` in the recording state.
    // Returns false on file/read failure; in that case m_heights is still
    // filled with `sampleCount^2` zeros and the GPU resources are still
    // created (uploading zeros) so the rest of the engine can run.
    bool load(const HeightmapDesc& desc, VkCommandBuffer uploadCmd);

    // Free the host-visible staging buffer. The caller must guarantee the
    // GPU has finished consuming it (fence signaled after uploadCmd submit).
    void releaseStaging();

    const std::vector<f32>& heights()    const { return m_heights; }
    u32        sampleCount()             const { return m_desc.sampleCount; }
    f32        worldSize()               const { return m_desc.worldSize; }
    vec3       origin()                  const { return m_origin; }
    u32        textureSlot()             const { return m_texSlot; }
    VkSampler  sampler()                 const { return m_sampler; }
    u32        samplerSlot()             const { return m_samplerSlot; }

    // CPU bilinear height sample — single source of truth for both
    // rendering and physics. worldX/worldZ are in meters.
    f32 sampleBilinear(f32 worldX, f32 worldZ) const;

private:
    gfx::Device*              m_device      = nullptr;
    gfx::Allocator*           m_allocator   = nullptr;
    gfx::DescriptorAllocator* m_descriptors = nullptr;

    HeightmapDesc    m_desc{};
    std::vector<f32> m_heights;
    vec3             m_origin{ -2048.0f, 0.0f, -2048.0f };

    VkImage       m_image         = VK_NULL_HANDLE;
    VmaAllocation m_imageAlloc    = nullptr;
    VkImageView   m_imageView     = VK_NULL_HANDLE;
    VkSampler     m_sampler       = VK_NULL_HANDLE;
    u32           m_texSlot       = UINT32_MAX;
    u32           m_samplerSlot   = UINT32_MAX;

    // Staging buffer — alive between load() and releaseStaging().
    VkBuffer      m_staging       = VK_NULL_HANDLE;
    VmaAllocation m_stagingAlloc  = nullptr;
};

} // namespace enigma
