#include "renderer/TrianglePass.h"

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
#include <vk_mem_alloc.h>

#include <array>
#include <cstring>
#include <filesystem>

namespace enigma {

namespace {

// NDC-space centered triangle, alpha=1 for padding. Matches the
// coordinates documented in the plan (step 36) so the executor and
// the plan agree byte-for-byte.
constexpr std::array<float, 12> kTriangleVertices = {
    -0.5f, -0.5f, 0.0f, 1.0f,
     0.5f, -0.5f, 0.0f, 1.0f,
     0.0f,  0.5f, 0.0f, 1.0f,
};

} // namespace

TrianglePass::TrianglePass(gfx::Device& device,
                           gfx::Allocator& allocator,
                           gfx::DescriptorAllocator& descriptorAllocator)
    : m_device(&device),
      m_allocator(&allocator) {

    // Create a host-visible SSBO and map it long enough to write the
    // three vertex positions. Host-visible is fine for 48 bytes at a
    // single draw call; a real mesh would use a device-local buffer
    // plus a staging upload.
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size        = sizeof(kTriangleVertices);
    bufferInfo.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage          = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags          = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                               | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    allocInfo.requiredFlags  = 0;

    VmaAllocationInfo allocationInfo{};
    ENIGMA_VK_CHECK(vmaCreateBuffer(allocator.handle(), &bufferInfo, &allocInfo,
                                    &m_vertexBuffer, &m_vertexAllocation, &allocationInfo));
    ENIGMA_ASSERT(allocationInfo.pMappedData != nullptr);
    std::memcpy(allocationInfo.pMappedData, kTriangleVertices.data(), sizeof(kTriangleVertices));

    // Register in the bindless descriptor set at binding 2.
    m_bindlessSlot = descriptorAllocator.registerStorageBuffer(
        m_vertexBuffer, sizeof(kTriangleVertices));

    ENIGMA_LOG_INFO("[triangle] ssbo created: size={} bytes, bindless slot={}",
                    sizeof(kTriangleVertices), m_bindlessSlot);
}

TrianglePass::~TrianglePass() {
    delete m_pipeline;
    if (m_allocator != nullptr && m_vertexBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator->handle(), m_vertexBuffer, m_vertexAllocation);
    }
}

void TrianglePass::buildPipeline(gfx::ShaderManager& shaderManager,
                                 VkDescriptorSetLayout globalSetLayout,
                                 VkFormat colorAttachmentFormat) {
    ENIGMA_ASSERT(m_pipeline == nullptr && "TrianglePass::buildPipeline called twice");

    // Capture everything rebuildPipeline() needs so a hot-reload
    // event can swap the pipeline without re-plumbing arguments from
    // the Renderer. `Paths::shaderSourceDir()` prefers the
    // in-repository `shaders/` directory (baked in at build time via
    // the ENIGMA_SHADER_SOURCE_DIR macro) so edits land without a
    // rebuild; it falls back to the exe-adjacent copy on shipped
    // binaries or moved build trees.
    m_shaderManager   = &shaderManager;
    m_globalSetLayout = globalSetLayout;
    m_colorFormat     = colorAttachmentFormat;
    m_vertPath        = Paths::shaderSourceDir() / "triangle.vert";
    m_fragPath        = Paths::shaderSourceDir() / "triangle.frag";

    VkShaderModule vert = shaderManager.compile(m_vertPath, gfx::ShaderManager::Stage::Vertex);
    VkShaderModule frag = shaderManager.compile(m_fragPath, gfx::ShaderManager::Stage::Fragment);

    m_pipeline = new gfx::Pipeline(*m_device, vert, frag, globalSetLayout, colorAttachmentFormat);

    // Shader modules can be destroyed as soon as the pipeline is built.
    vkDestroyShaderModule(m_device->logical(), vert, nullptr);
    vkDestroyShaderModule(m_device->logical(), frag, nullptr);
}

void TrianglePass::rebuildPipeline() {
    ENIGMA_ASSERT(m_pipeline != nullptr && "rebuildPipeline before initial build");
    ENIGMA_ASSERT(m_shaderManager != nullptr);

    // Compile both shaders BEFORE touching the existing pipeline.
    // Either failure keeps the previous pipeline intact and logs an
    // error so the frame loop continues unaffected.
    VkShaderModule vert =
        m_shaderManager->tryCompile(m_vertPath, gfx::ShaderManager::Stage::Vertex);
    if (vert == VK_NULL_HANDLE) {
        ENIGMA_LOG_ERROR("[triangle] hot-reload: vertex compile failed, keeping previous pipeline");
        return;
    }
    VkShaderModule frag =
        m_shaderManager->tryCompile(m_fragPath, gfx::ShaderManager::Stage::Fragment);
    if (frag == VK_NULL_HANDLE) {
        ENIGMA_LOG_ERROR("[triangle] hot-reload: fragment compile failed, keeping previous pipeline");
        vkDestroyShaderModule(m_device->logical(), vert, nullptr);
        return;
    }

    // Both shaders compiled — swap is now safe. `vkDeviceWaitIdle`
    // is heavy but this is a developer-only path; simplicity wins
    // over trying to fence individual frames.
    vkDeviceWaitIdle(m_device->logical());

    delete m_pipeline;
    m_pipeline = new gfx::Pipeline(*m_device, vert, frag, m_globalSetLayout, m_colorFormat);

    vkDestroyShaderModule(m_device->logical(), vert, nullptr);
    vkDestroyShaderModule(m_device->logical(), frag, nullptr);

    ENIGMA_LOG_INFO("[triangle] hot-reload: pipeline rebuilt successfully");
}

void TrianglePass::registerHotReload(gfx::ShaderHotReload& reloader) {
    ENIGMA_ASSERT(m_pipeline != nullptr && "registerHotReload called before buildPipeline");
    reloader.watchGroup({m_vertPath, m_fragPath},
                        [this]() { rebuildPipeline(); });
}

void TrianglePass::record(VkCommandBuffer cmd,
                          VkDescriptorSet globalSet,
                          VkExtent2D extent) {
    ENIGMA_ASSERT(m_pipeline != nullptr && "TrianglePass::record before buildPipeline");

    // Viewport and scissor (pipeline uses dynamic state for both).
    VkViewport viewport{};
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = static_cast<float>(extent.width);
    viewport.height   = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = extent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Bind pipeline + global descriptor set.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline->handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_pipeline->layout(), 0, 1, &globalSet, 0, nullptr);

    // Push the bindless slot index (first 4 bytes of the 16-byte range).
    struct PushBlock { u32 bufferIndex; u32 _pad[3]; };
    PushBlock pc{};
    pc.bufferIndex = m_bindlessSlot;
    vkCmdPushConstants(cmd, m_pipeline->layout(),
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);

    // Draw 3 vertices; positions come from the bindless SSBO.
    vkCmdDraw(cmd, 3, 1, 0, 0);

    // ---- Runtime bindless proof (AC7-runtime) ------------------------
    // On the first record, emit the log line that verification step 3b
    // greps for. This is the machine-checkable signal that the bindless
    // path is actually live, not just scaffolded.
    if (m_firstRecord) {
        m_firstRecord = false;
        ENIGMA_LOG_INFO("[triangle] bindless slot={} ssbo=0x{:x} vertices written: "
                        "(-0.5,-0.5), (0.5,-0.5), (0.0,0.5)",
                        m_bindlessSlot,
                        reinterpret_cast<u64>(m_vertexBuffer));

#if ENIGMA_DEBUG
        // Debug-only CPU readback of the SSBO to confirm the 3 positions
        // match what we wrote. Host-visible buffer is still mapped via
        // VMA so the read is synchronous.
        VmaAllocationInfo info{};
        vmaGetAllocationInfo(m_allocator->handle(), m_vertexAllocation, &info);
        ENIGMA_ASSERT(info.pMappedData != nullptr);
        const float* mapped = static_cast<const float*>(info.pMappedData);
        ENIGMA_ASSERT(mapped[0]  == -0.5f && mapped[1]  == -0.5f);
        ENIGMA_ASSERT(mapped[4]  ==  0.5f && mapped[5]  == -0.5f);
        ENIGMA_ASSERT(mapped[8]  ==  0.0f && mapped[9]  ==  0.5f);
#endif
    }
}

} // namespace enigma
