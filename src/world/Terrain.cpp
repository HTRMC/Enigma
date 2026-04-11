#include "world/Terrain.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "core/Paths.h"
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"
#include "gfx/Pipeline.h"
#include "gfx/ShaderHotReload.h"
#include "gfx/ShaderManager.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable: 4100 4127 4189 4324 4505)
#endif
#include <vk_mem_alloc.h>
#if defined(_MSC_VER)
    #pragma warning(pop)
#endif

#include <cmath>
#include <cstring>

namespace enigma {

// Push constant block — must match terrain_clipmap.hlsl TerrainPC.
struct TerrainPushBlock {
    u32 chunkSSBOSlot;
    u32 cameraSlot;
    u32 quadsPerSide;
    f32 pad;
};
static_assert(sizeof(TerrainPushBlock) == 16);

namespace {

// Snap a world coordinate to the nearest multiple of `step`, toward
// negative infinity. Matches HLSL floor() semantics.
f32 snapFloor(f32 value, f32 step) {
    return std::floor(value / step) * step;
}

} // namespace

// ---------------------------------------------------------------------------

Terrain::Terrain(gfx::Device& device,
                 gfx::Allocator& allocator,
                 gfx::DescriptorAllocator& descriptorAllocator)
    : m_device(&device)
    , m_allocator(&allocator)
    , m_descriptorAllocator(&descriptorAllocator) {

    // Allocate the chunk SSBO sized for the maximum number of active
    // chunks (one 3x3 grid per LOD level). Host-visible + mapped so we
    // can rewrite it every frame without a staging upload.
    constexpr VkDeviceSize kSize =
        kLodLevels * kChunksPerRing * sizeof(TerrainChunkDesc);

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size        = kSize;
    bufInfo.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                    | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo allocResult{};
    ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &bufInfo, &allocInfo,
                                    &m_chunkSSBO, &m_chunkAlloc, &allocResult));
    m_chunkMapped = allocResult.pMappedData;
    ENIGMA_ASSERT(m_chunkMapped != nullptr);

    m_chunkSlot = m_descriptorAllocator->registerStorageBuffer(m_chunkSSBO, kSize);
    m_chunks.reserve(kLodLevels * kChunksPerRing);

    ENIGMA_LOG_INFO("[terrain] chunk SSBO registered at bindless slot {}", m_chunkSlot);
}

Terrain::~Terrain() {
    delete m_pipeline;
    if (m_chunkSSBO != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator->handle(), m_chunkSSBO, m_chunkAlloc);
    }
}

// ---------------------------------------------------------------------------

void Terrain::buildPipeline(gfx::ShaderManager& shaderManager,
                             VkDescriptorSetLayout globalSetLayout,
                             VkFormat colorFormat,
                             VkFormat depthFormat,
                             VkFormat normalFormat,
                             VkFormat metalRoughFormat,
                             VkFormat motionVecFormat) {
    ENIGMA_ASSERT(m_pipeline == nullptr && "Terrain::buildPipeline called twice");

    m_shaderManager    = &shaderManager;
    m_globalSetLayout  = globalSetLayout;
    m_colorFormat      = colorFormat;
    m_depthFormat      = depthFormat;
    m_normalFormat     = normalFormat;
    m_metalRoughFormat = metalRoughFormat;
    m_motionVecFormat  = motionVecFormat;
    m_shaderPath       = Paths::shaderSourceDir() / "terrain_clipmap.hlsl";

    VkShaderModule vert = shaderManager.compile(m_shaderPath, gfx::ShaderManager::Stage::Vertex,   "VSMain");
    VkShaderModule frag = shaderManager.compile(m_shaderPath, gfx::ShaderManager::Stage::Fragment, "PSMain");

    gfx::Pipeline::CreateInfo ci{};
    ci.vertShader              = vert;
    ci.vertEntryPoint          = "VSMain";
    ci.fragShader              = frag;
    ci.fragEntryPoint          = "PSMain";
    ci.globalSetLayout         = globalSetLayout;
    ci.colorAttachmentFormats[0] = colorFormat;
    ci.colorAttachmentFormats[1] = normalFormat;
    ci.colorAttachmentFormats[2] = metalRoughFormat;
    ci.colorAttachmentFormats[3] = motionVecFormat;
    ci.colorAttachmentCount    = 4;
    ci.depthAttachmentFormat   = depthFormat;
    ci.pushConstantSize        = sizeof(TerrainPushBlock);
    ci.depthCompareOp          = VK_COMPARE_OP_GREATER_OR_EQUAL; // reverse-Z
    ci.cullMode                = VK_CULL_MODE_BACK_BIT;

    m_pipeline = new gfx::Pipeline(*m_device, ci);

    vkDestroyShaderModule(m_device->logical(), vert, nullptr);
    vkDestroyShaderModule(m_device->logical(), frag, nullptr);

    ENIGMA_LOG_INFO("[terrain] pipeline built");
}

void Terrain::rebuildPipeline() {
    ENIGMA_ASSERT(m_pipeline != nullptr);
    ENIGMA_ASSERT(m_shaderManager != nullptr);

    VkShaderModule vert = m_shaderManager->tryCompile(m_shaderPath, gfx::ShaderManager::Stage::Vertex,   "VSMain");
    if (vert == VK_NULL_HANDLE) { ENIGMA_LOG_ERROR("[terrain] hot-reload: VS compile failed"); return; }
    VkShaderModule frag = m_shaderManager->tryCompile(m_shaderPath, gfx::ShaderManager::Stage::Fragment, "PSMain");
    if (frag == VK_NULL_HANDLE) {
        ENIGMA_LOG_ERROR("[terrain] hot-reload: PS compile failed");
        vkDestroyShaderModule(m_device->logical(), vert, nullptr);
        return;
    }

    vkDeviceWaitIdle(m_device->logical());
    delete m_pipeline;

    gfx::Pipeline::CreateInfo ci{};
    ci.vertShader              = vert;
    ci.vertEntryPoint          = "VSMain";
    ci.fragShader              = frag;
    ci.fragEntryPoint          = "PSMain";
    ci.globalSetLayout         = m_globalSetLayout;
    ci.colorAttachmentFormats[0] = m_colorFormat;
    ci.colorAttachmentFormats[1] = m_normalFormat;
    ci.colorAttachmentFormats[2] = m_metalRoughFormat;
    ci.colorAttachmentFormats[3] = m_motionVecFormat;
    ci.colorAttachmentCount    = 4;
    ci.depthAttachmentFormat   = m_depthFormat;
    ci.pushConstantSize        = sizeof(TerrainPushBlock);
    ci.depthCompareOp          = VK_COMPARE_OP_GREATER_OR_EQUAL;
    ci.cullMode                = VK_CULL_MODE_BACK_BIT;
    m_pipeline = new gfx::Pipeline(*m_device, ci);

    vkDestroyShaderModule(m_device->logical(), vert, nullptr);
    vkDestroyShaderModule(m_device->logical(), frag, nullptr);

    ENIGMA_LOG_INFO("[terrain] hot-reload: pipeline rebuilt");
}

void Terrain::registerHotReload(gfx::ShaderHotReload& reloader) {
    ENIGMA_ASSERT(m_pipeline != nullptr);
    reloader.watchGroup({m_shaderPath}, [this]() { rebuildPipeline(); });
}

// ---------------------------------------------------------------------------

void Terrain::update(vec3 cameraPosition) {
    m_chunks.clear();

    for (u32 lod = 0; lod < kLodLevels; ++lod) {
        const f32 scale     = static_cast<f32>(1u << lod);
        const f32 chunkSize = kBaseChunkSize * scale;

        // Snap the grid origin to the chunk grid for this LOD so chunks
        // don't jitter as the camera moves.
        const f32 snappedX = snapFloor(cameraPosition.x, chunkSize);
        const f32 snappedZ = snapFloor(cameraPosition.z, chunkSize);

        // 3x3 grid centered on the snapped camera position.
        // Outer rings are sunk below inner rings so the higher-detail
        // closer terrain occludes the LOD seam.
        const f32 sink = (lod == 0) ? 0.0f
                                    : -chunkSize * 0.5f * static_cast<f32>(lod);

        for (i32 dz = -1; dz <= 1; ++dz) {
            for (i32 dx = -1; dx <= 1; ++dx) {
                TerrainChunkDesc c{};
                c.worldOffset = vec2(snappedX + static_cast<f32>(dx) * chunkSize,
                                     snappedZ + static_cast<f32>(dz) * chunkSize);
                c.scale       = scale;
                c.sinkAmount  = sink;
                m_chunks.push_back(c);
            }
        }
    }

    m_totalInstances = static_cast<u32>(m_chunks.size());
    uploadChunkSSBO();
}

void Terrain::uploadChunkSSBO() {
    if (m_chunks.empty()) return;
    std::memcpy(m_chunkMapped, m_chunks.data(),
                m_chunks.size() * sizeof(TerrainChunkDesc));
}

// ---------------------------------------------------------------------------

void Terrain::record(VkCommandBuffer cmd,
                     VkExtent2D extent,
                     VkDescriptorSet globalSet,
                     u32 cameraSlot) {
    ENIGMA_ASSERT(m_pipeline != nullptr && "Terrain::record before buildPipeline");
    if (m_totalInstances == 0) return;

    VkViewport viewport{};
    viewport.width    = static_cast<float>(extent.width);
    viewport.height   = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = extent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline->handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_pipeline->layout(), 0, 1, &globalSet, 0, nullptr);

    TerrainPushBlock pc{};
    pc.chunkSSBOSlot = m_chunkSlot;
    pc.cameraSlot    = cameraSlot;
    pc.quadsPerSide  = kQuadsPerChunk;
    pc.pad           = 0.0f;

    vkCmdPushConstants(cmd, m_pipeline->layout(),
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);

    // 6 vertices per quad, N*N quads per chunk, `m_totalInstances` chunks.
    const u32 vertsPerChunk = 6u * kQuadsPerChunk * kQuadsPerChunk;
    vkCmdDraw(cmd, vertsPerChunk, m_totalInstances, 0, 0);
}

} // namespace enigma
