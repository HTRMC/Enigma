#include "physics/PhysicsDebugRenderer.h"

#ifdef JPH_DEBUG_RENDERER

#include "core/Assert.h"
#include "core/Log.h"
#include "core/Paths.h"
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"
#include "gfx/ShaderManager.h"
#include "physics/PhysicsWorld.h"

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

#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Body/BodyManager.h>
#include <Jolt/Physics/Body/Body.h>

#include <cstring>

namespace enigma {

// ─────────────────────────────────────────────────────────────────────────────
// DynamicBodyFilter
// ─────────────────────────────────────────────────────────────────────────────

bool DynamicBodyFilter::ShouldDraw(const JPH::Body& inBody) const {
    // Skip layer 0 (Static) — heightfield + ground plane would cost ~500k
    // line primitives per frame and swamp the debug overlay.
    return inBody.GetObjectLayer() != static_cast<JPH::ObjectLayer>(PhysicsLayer::Static);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline helper
// ─────────────────────────────────────────────────────────────────────────────

VkPipeline PhysicsDebugRenderer::buildPipeline(VkShaderModule vs, VkShaderModule fs,
                                                bool depthTest, VkFormat colorFmt,
                                                VkFormat depthFmt, VkDevice dev) const {
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs;
    stages[0].pName  = "VSMain";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs;
    stages[1].pName  = "PSMain";

    // Geometry is SSBO-driven via SV_VertexID — no vertex buffers bound.
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode    = VK_CULL_MODE_NONE;
    rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Reverse-Z: closer geometry has larger depth values, so depth-tested
    // lines use GREATER_OR_EQUAL. depthWrite is always false — debug lines
    // must never corrupt the G-buffer depth used by subsequent passes.
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable  = depthTest ? VK_TRUE : VK_FALSE;
    depthStencil.depthWriteEnable = VK_FALSE;
    depthStencil.depthCompareOp   = VK_COMPARE_OP_GREATER_OR_EQUAL;

    VkPipelineColorBlendAttachmentState blendAttach{};
    blendAttach.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                 VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments    = &blendAttach;

    const VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates    = dynStates;

    // Dynamic rendering (VK_KHR_dynamic_rendering, core in Vulkan 1.3).
    const VkFormat depthFmtForPipeline = depthTest ? depthFmt : VK_FORMAT_UNDEFINED;
    VkPipelineRenderingCreateInfo renderingCI{};
    renderingCI.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingCI.colorAttachmentCount    = 1;
    renderingCI.pColorAttachmentFormats = &colorFmt;
    renderingCI.depthAttachmentFormat   = depthFmtForPipeline;

    VkGraphicsPipelineCreateInfo pipelineCI{};
    pipelineCI.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCI.pNext               = &renderingCI;
    pipelineCI.stageCount          = 2;
    pipelineCI.pStages             = stages;
    pipelineCI.pVertexInputState   = &vertexInput;
    pipelineCI.pInputAssemblyState = &inputAssembly;
    pipelineCI.pViewportState      = &viewportState;
    pipelineCI.pRasterizationState = &rasterizer;
    pipelineCI.pMultisampleState   = &multisample;
    pipelineCI.pDepthStencilState  = &depthStencil;
    pipelineCI.pColorBlendState    = &colorBlend;
    pipelineCI.pDynamicState       = &dynamicState;
    pipelineCI.layout              = m_pipelineLayout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    ENIGMA_VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1,
                                               &pipelineCI, nullptr, &pipeline));
    return pipeline;
}

// ─────────────────────────────────────────────────────────────────────────────
// Lifecycle
// ─────────────────────────────────────────────────────────────────────────────

void PhysicsDebugRenderer::init(const PhysicsDebugInitInfo& info) {
    m_device              = info.device;
    m_allocator           = info.allocator;
    m_descriptorAllocator = info.descriptorAllocator;

    const VkDevice dev = m_device->logical();

    // 1. Persistently-mapped GPU SSBO ────────────────────────────────────────
    constexpr VkDeviceSize kSsboBytes = kMaxLineVertices * sizeof(LineVertex);

    VkBufferCreateInfo bufCI{};
    bufCI.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufCI.size        = kSsboBytes;
    bufCI.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO;
    allocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                  | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo allocResult{};
    ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &bufCI, &allocCI,
                                    &m_ssbo, &m_allocation, &allocResult));
    m_mapped = allocResult.pMappedData;
    ENIGMA_ASSERT(m_mapped != nullptr);

    m_ssboSlot = m_descriptorAllocator->registerStorageBuffer(m_ssbo, kSsboBytes);

    // 2. Pipeline layout ──────────────────────────────────────────────────────
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset     = 0;
    pcRange.size       = 16;

    VkPipelineLayoutCreateInfo layoutCI{};
    layoutCI.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCI.setLayoutCount         = 1;
    layoutCI.pSetLayouts            = &info.globalSetLayout;
    layoutCI.pushConstantRangeCount = 1;
    layoutCI.pPushConstantRanges    = &pcRange;
    ENIGMA_VK_CHECK(vkCreatePipelineLayout(dev, &layoutCI, nullptr, &m_pipelineLayout));

    // 3. Compile shader and build two pipeline variants ───────────────────────
    const auto shaderPath = Paths::shaderDir() / "physics_debug.hlsl";
    VkShaderModule vs = info.shaderManager->compile(shaderPath,
                                                      gfx::ShaderManager::Stage::Vertex,
                                                      "VSMain");
    VkShaderModule fs = info.shaderManager->compile(shaderPath,
                                                      gfx::ShaderManager::Stage::Fragment,
                                                      "PSMain");

    m_depthPipeline = buildPipeline(vs, fs, /*depthTest=*/true,
                                     info.colorFormat, info.depthFormat, dev);
    m_xrayPipeline  = buildPipeline(vs, fs, /*depthTest=*/false,
                                     info.colorFormat, info.depthFormat, dev);

    vkDestroyShaderModule(dev, vs, nullptr);
    vkDestroyShaderModule(dev, fs, nullptr);

    m_lineVertices.reserve(4096);

    ENIGMA_LOG_INFO("[physics debug] renderer ready (ssboSlot={}, maxVerts={})",
                    m_ssboSlot, kMaxLineVertices);
}

void PhysicsDebugRenderer::destroy() {
    if (m_device == nullptr) return;
    const VkDevice dev = m_device->logical();

    if (m_depthPipeline  != VK_NULL_HANDLE) { vkDestroyPipeline(dev, m_depthPipeline,  nullptr); m_depthPipeline  = VK_NULL_HANDLE; }
    if (m_xrayPipeline   != VK_NULL_HANDLE) { vkDestroyPipeline(dev, m_xrayPipeline,   nullptr); m_xrayPipeline   = VK_NULL_HANDLE; }
    if (m_pipelineLayout != VK_NULL_HANDLE) { vkDestroyPipelineLayout(dev, m_pipelineLayout, nullptr); m_pipelineLayout = VK_NULL_HANDLE; }
    if (m_ssbo != VK_NULL_HANDLE) {
        m_descriptorAllocator->releaseStorageBuffer(m_ssboSlot);
        vmaDestroyBuffer(m_allocator->handle(), m_ssbo, m_allocation);
        m_ssbo       = VK_NULL_HANDLE;
        m_allocation = nullptr;
        m_mapped     = nullptr;
    }
    m_device = nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-frame interface
// ─────────────────────────────────────────────────────────────────────────────

void PhysicsDebugRenderer::gather(JPH::PhysicsSystem& physicsSystem) {
    m_lineVertices.clear();
    JPH::BodyManager::DrawSettings settings{};
    settings.mDrawShape          = true;
    settings.mDrawShapeWireframe = true;
    physicsSystem.DrawBodies(settings, this, &m_bodyFilter);
}

void PhysicsDebugRenderer::upload() {
    m_uploadedCount = static_cast<u32>(m_lineVertices.size());
    if (m_uploadedCount == 0 || m_mapped == nullptr) return;
    std::memcpy(m_mapped, m_lineVertices.data(), m_uploadedCount * sizeof(LineVertex));
}

void PhysicsDebugRenderer::drawFrame(VkCommandBuffer cmd, VkDescriptorSet globalSet,
                                      VkExtent2D ext, u32 cameraSlot) {
    if (m_uploadedCount == 0) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                       depthTestEnabled ? m_depthPipeline : m_xrayPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout,
                             0, 1, &globalSet, 0, nullptr);

    VkViewport vp{0.f, 0.f, static_cast<float>(ext.width), static_cast<float>(ext.height), 0.f, 1.f};
    vkCmdSetViewport(cmd, 0, 1, &vp);

    VkRect2D scissor{{0, 0}, ext};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    struct { uint32_t lineSSBOSlot, cameraSlot, pad0, pad1; } pc{
        m_ssboSlot, cameraSlot, 0u, 0u
    };
    vkCmdPushConstants(cmd, m_pipelineLayout,
                        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                        0, 16, &pc);

    vkCmdDraw(cmd, m_uploadedCount, 1, 0, 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// JPH::DebugRendererSimple callback
// ─────────────────────────────────────────────────────────────────────────────

void PhysicsDebugRenderer::DrawLine(JPH::RVec3Arg inFrom, JPH::RVec3Arg inTo,
                                     JPH::ColorArg inColor) {
    if (m_lineVertices.size() + 2 > kMaxLineVertices) return;

    const uint32_t packed = static_cast<uint32_t>(inColor.r)
                           | (static_cast<uint32_t>(inColor.g) << 8u)
                           | (static_cast<uint32_t>(inColor.b) << 16u)
                           | (static_cast<uint32_t>(inColor.a) << 24u);

    m_lineVertices.push_back({ static_cast<float>(inFrom.GetX()),
                                static_cast<float>(inFrom.GetY()),
                                static_cast<float>(inFrom.GetZ()), packed });
    m_lineVertices.push_back({ static_cast<float>(inTo.GetX()),
                                static_cast<float>(inTo.GetY()),
                                static_cast<float>(inTo.GetZ()), packed });
}

} // namespace enigma

#endif // JPH_DEBUG_RENDERER
