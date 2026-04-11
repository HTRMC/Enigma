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
class Pipeline;
class ShaderHotReload;
class ShaderManager;
} // namespace enigma::gfx

namespace enigma {

// Packed chunk descriptor uploaded to the SSBO. Matches the float4
// layout the shader reads: (worldOffsetX, worldOffsetZ, scale, sinkAmount).
struct TerrainChunkDesc {
    vec2 worldOffset;
    f32  scale;
    f32  sinkAmount;
};
static_assert(sizeof(TerrainChunkDesc) == 16,
              "TerrainChunkDesc must pack into a float4 for the SSBO layout");

// GPU-driven clipmap LOD terrain.
//
// Design points:
//   - All vertices generated from SV_VertexID in the vertex shader (no
//     vertex buffers bound). XZ, UVs and normals are derived on the fly.
//   - One draw call per frame — chunks are instanced via SV_InstanceID
//     indexing into a small SSBO of TerrainChunkDesc entries.
//   - 4 LOD rings (3x3 grid each) with successively doubled chunk size.
//     Outer rings sink below inner rings to hide the LOD seam.
class Terrain {
public:
    static constexpr u32 kChunksPerRing = 9;  // 3x3 grid per LOD ring
    static constexpr u32 kLodLevels     = 4;
    static constexpr u32 kQuadsPerChunk = 32; // N
    static constexpr f32 kBaseChunkSize = 64.0f; // meters

    Terrain(gfx::Device& device,
            gfx::Allocator& allocator,
            gfx::DescriptorAllocator& descriptorAllocator);
    ~Terrain();

    Terrain(const Terrain&)            = delete;
    Terrain& operator=(const Terrain&) = delete;
    Terrain(Terrain&&)                 = delete;
    Terrain& operator=(Terrain&&)      = delete;

    void buildPipeline(gfx::ShaderManager& shaderManager,
                       VkDescriptorSetLayout globalSetLayout,
                       VkFormat colorFormat,
                       VkFormat depthFormat,
                       VkFormat normalFormat,
                       VkFormat metalRoughFormat,
                       VkFormat motionVecFormat);

    void registerHotReload(gfx::ShaderHotReload& reloader);

    // Rebuild chunk positions for this frame given the current camera pos.
    void update(vec3 cameraPosition);

    // Draw the terrain. Must be called inside a render pass whose color
    // targets match the G-buffer MRT layout (albedo/normal/metalRough/
    // motionVec + depth).
    void record(VkCommandBuffer cmd,
                VkExtent2D extent,
                VkDescriptorSet globalSet,
                u32 cameraSlot);

private:
    void uploadChunkSSBO();
    void rebuildPipeline();

    gfx::Device*              m_device              = nullptr;
    gfx::Allocator*           m_allocator           = nullptr;
    gfx::DescriptorAllocator* m_descriptorAllocator = nullptr;
    gfx::Pipeline*            m_pipeline            = nullptr;

    // GPU SSBO: array of TerrainChunkDesc (one per active chunk).
    VkBuffer      m_chunkSSBO  = VK_NULL_HANDLE;
    VmaAllocation m_chunkAlloc = nullptr;
    void*         m_chunkMapped = nullptr;
    u32           m_chunkSlot  = 0;

    std::vector<TerrainChunkDesc> m_chunks;
    u32 m_totalInstances = 0;

    // Hot-reload / pipeline rebuild state.
    gfx::ShaderManager*   m_shaderManager   = nullptr;
    VkDescriptorSetLayout m_globalSetLayout = VK_NULL_HANDLE;
    VkFormat m_colorFormat      = VK_FORMAT_UNDEFINED;
    VkFormat m_depthFormat      = VK_FORMAT_UNDEFINED;
    VkFormat m_normalFormat     = VK_FORMAT_UNDEFINED;
    VkFormat m_metalRoughFormat = VK_FORMAT_UNDEFINED;
    VkFormat m_motionVecFormat  = VK_FORMAT_UNDEFINED;
    std::filesystem::path m_shaderPath;
};

} // namespace enigma
