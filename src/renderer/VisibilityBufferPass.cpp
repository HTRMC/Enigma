#include "renderer/VisibilityBufferPass.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "core/Paths.h"
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"
#include "gfx/ShaderHotReload.h"
#include "gfx/ShaderManager.h"
#include "renderer/GpuMeshletBuffer.h"
#include "renderer/GpuSceneBuffer.h"
#include "renderer/IndirectDrawBuffer.h"
#include "renderer/Meshlet.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vk_mem_alloc.h>

#include <array>
#include <cstring>
#include <vector>

namespace enigma {

// Must match TASK_GROUP_SIZE in visibility_buffer.task.hlsl.
static constexpr u32 TASK_GROUP_SIZE = 32;

// Unified push block — shared by task and mesh stages.
// Must match the PushBlock declaration in both visibility_buffer.task.hlsl
// and visibility_buffer.mesh.hlsl.
struct VBPushBlock {
    u32 instanceBufferSlot;   // GpuInstance[] — both task + mesh
    u32 meshletBufferSlot;    // Meshlet[]     — both task + mesh
    u32 survivingIdsSlot;     // u32[] surviving global meshlet IDs — task only
    u32 meshletVerticesSlot;  // u32[] vertex index remapping — mesh only
    u32 meshletTrianglesSlot; // u8 packed triangle indices — mesh only
    u32 cameraSlot;           // CameraData — both task + mesh
    u32 countBufferSlot;      // u32 actual surviving count (GPU buffer) — task only
    u32 instanceCount;        // number of GpuInstance entries — task only
};

static_assert(sizeof(VBPushBlock) == 32);

// Terrain push block — extends VBPushBlock with the two per-LOD shared
// topology SSBO slots the terrain mesh shader reads patch topology from.
// Must match the PushBlock declaration in terrain_cdlod.task.hlsl and
// terrain_cdlod.mesh.hlsl exactly (same first 8 fields, then 2 extra).
struct VBTerrainPushBlock {
    u32 instanceBufferSlot;
    u32 meshletBufferSlot;
    u32 survivingIdsSlot;
    u32 meshletVerticesSlot;
    u32 meshletTrianglesSlot;
    u32 cameraSlot;
    u32 countBufferSlot;
    u32 instanceCount;
    u32 topologyVerticesSlot;
    u32 topologyTrianglesSlot;
};

static_assert(sizeof(VBTerrainPushBlock) == 40);

VisibilityBufferPass::VisibilityBufferPass(gfx::Device& device,
                                           gfx::Allocator& allocator,
                                           gfx::DescriptorAllocator& descriptors)
    : m_device(&device)
    , m_allocator(&allocator)
    , m_descriptors(&descriptors)
    , m_useMeshShaders(device.supportsMeshShaders())
{}

VisibilityBufferPass::~VisibilityBufferPass() {
    if (m_wireframePipeline       != VK_NULL_HANDLE) vkDestroyPipeline(m_device->logical(), m_wireframePipeline, nullptr);
    if (m_wireframePipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_device->logical(), m_wireframePipelineLayout, nullptr);
    if (m_terrainWireframePipeline       != VK_NULL_HANDLE) vkDestroyPipeline(m_device->logical(), m_terrainWireframePipeline, nullptr);
    if (m_terrainWireframePipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_device->logical(), m_terrainWireframePipelineLayout, nullptr);
    if (m_terrainPipeline       != VK_NULL_HANDLE) vkDestroyPipeline(m_device->logical(), m_terrainPipeline, nullptr);
    if (m_terrainPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_device->logical(), m_terrainPipelineLayout, nullptr);
    if (m_vsFallbackPipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(m_device->logical(), m_vsFallbackPipeline, nullptr);
    if (m_vsFallbackDrawBuffer != VK_NULL_HANDLE)
        vmaDestroyBuffer(m_allocator->handle(), m_vsFallbackDrawBuffer, m_vsFallbackDrawAlloc);
    if (m_pipeline       != VK_NULL_HANDLE) vkDestroyPipeline(m_device->logical(), m_pipeline, nullptr);
    if (m_pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_device->logical(), m_pipelineLayout, nullptr);
    destroyVisImage();
}

void VisibilityBufferPass::destroyVisImage() {
    if (m_vis_slot != 0) {
        m_descriptors->releaseSampledImage(m_vis_slot);
        m_vis_slot = 0;
    }
    if (m_vis_view  != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device->logical(), m_vis_view, nullptr);
        m_vis_view = VK_NULL_HANDLE;
    }
    if (m_vis_image != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator->handle(), m_vis_image, m_vis_alloc);
        m_vis_image = VK_NULL_HANDLE;
        m_vis_alloc = nullptr;
    }
}

void VisibilityBufferPass::allocate(VkExtent2D extent, VkFormat depthFormat) {
    if (m_vis_image != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device->logical());
        destroyVisImage();
    }
    m_extent      = extent;
    m_depthFormat = depthFormat;

    VkImageCreateInfo imageCI{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    imageCI.imageType     = VK_IMAGE_TYPE_2D;
    imageCI.format        = VK_FORMAT_R32_UINT;
    imageCI.extent        = { extent.width, extent.height, 1 };
    imageCI.mipLevels     = 1;
    imageCI.arrayLayers   = 1;
    imageCI.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageCI.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                          | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCI.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    ENIGMA_VK_CHECK(vmaCreateImage(m_allocator->handle(), &imageCI, &allocCI,
                                   &m_vis_image, &m_vis_alloc, nullptr));

    VkImageViewCreateInfo viewCI{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    viewCI.image                           = m_vis_image;
    viewCI.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format                          = VK_FORMAT_R32_UINT;
    viewCI.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    viewCI.subresourceRange.baseMipLevel   = 0;
    viewCI.subresourceRange.levelCount     = 1;
    viewCI.subresourceRange.baseArrayLayer = 0;
    viewCI.subresourceRange.layerCount     = 1;

    ENIGMA_VK_CHECK(vkCreateImageView(m_device->logical(), &viewCI, nullptr, &m_vis_view));

    // Register as sampled image (MaterialEvalPass reads it as Texture2D at binding 0).
    m_vis_slot = m_descriptors->registerSampledImage(m_vis_view,
                                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    ENIGMA_LOG_INFO("[visibility_buffer] allocated {}x{} R32_UINT (slot {})",
                    extent.width, extent.height, m_vis_slot);
}

void VisibilityBufferPass::buildPipeline(gfx::ShaderManager& shaderManager,
                                          VkDescriptorSetLayout globalSetLayout) {
    ENIGMA_ASSERT(m_pipeline == VK_NULL_HANDLE && "VisibilityBufferPass::buildPipeline called twice");

    m_shaderManager   = &shaderManager;
    m_globalSetLayout = globalSetLayout;

    if (!m_useMeshShaders) {
        // VS fallback path — build vertex shader pipeline only.
        m_vsShaderPath = Paths::shaderSourceDir() / "visibility_buffer_vs.hlsl";
        buildVsFallbackPipeline();
        ENIGMA_LOG_INFO("[visibility_buffer] VS fallback pipeline built (no mesh shader support)");
        return;
    }

    m_taskShaderPath  = Paths::shaderSourceDir() / "visibility_buffer.task.hlsl";
    m_meshShaderPath  = Paths::shaderSourceDir() / "visibility_buffer.mesh.hlsl";

    VkShaderModule taskMod = shaderManager.compile(m_taskShaderPath, gfx::ShaderManager::Stage::Task, "ASMain");
    VkShaderModule meshMod = shaderManager.compile(m_meshShaderPath, gfx::ShaderManager::Stage::Mesh, "MSMain");
    VkShaderModule fragMod = shaderManager.compile(m_meshShaderPath, gfx::ShaderManager::Stage::Fragment, "PSMain");

    // Pipeline layout: global bindless set + push constants covering all mesh stages.
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT
                         | VK_SHADER_STAGE_MESH_BIT_EXT
                         | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset     = 0;
    pushRange.size       = sizeof(VBPushBlock);

    VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    layoutInfo.setLayoutCount         = 1;
    layoutInfo.pSetLayouts            = &globalSetLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges    = &pushRange;

    ENIGMA_VK_CHECK(vkCreatePipelineLayout(m_device->logical(), &layoutInfo, nullptr,
                                           &m_pipelineLayout));

    const std::array<VkPipelineShaderStageCreateInfo, 3> stages = {{
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
          VK_SHADER_STAGE_TASK_BIT_EXT, taskMod, "ASMain", nullptr },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
          VK_SHADER_STAGE_MESH_BIT_EXT, meshMod, "MSMain", nullptr },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
          VK_SHADER_STAGE_FRAGMENT_BIT, fragMod, "PSMain", nullptr },
    }};

    // Viewport + scissor: dynamic.
    const std::array<VkDynamicState, 2> dynamicStates = {{
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    }};
    VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamicState.dynamicStateCount = static_cast<u32>(dynamicStates.size());
    dynamicState.pDynamicStates    = dynamicStates.data();

    VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode    = VK_CULL_MODE_NONE; // depth test handles occlusion; Y-flip reverses winding
    rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Single R32_UINT color target — opaque overwrite, no blending.
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.blendEnable    = VK_FALSE;
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlend{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments    = &blendAttachment;

    // Depth: reverse-Z write (clear to 0, near = 1, far = 0).
    VkPipelineDepthStencilStateCreateInfo depthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    depthStencil.depthTestEnable  = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp   = VK_COMPARE_OP_GREATER_OR_EQUAL; // reverse-Z

    const VkFormat visFormat = VK_FORMAT_R32_UINT;
    VkPipelineRenderingCreateInfo renderingInfo{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    renderingInfo.colorAttachmentCount    = 1;
    renderingInfo.pColorAttachmentFormats = &visFormat;
    renderingInfo.depthAttachmentFormat   = m_depthFormat;

    // Mesh shader pipeline: pVertexInputState and pInputAssemblyState MUST be NULL.
    VkGraphicsPipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pipelineInfo.pNext               = &renderingInfo;
    pipelineInfo.stageCount          = static_cast<u32>(stages.size());
    pipelineInfo.pStages             = stages.data();
    pipelineInfo.pVertexInputState   = nullptr; // required null for mesh pipelines
    pipelineInfo.pInputAssemblyState = nullptr; // required null for mesh pipelines
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState   = &multisample;
    pipelineInfo.pDepthStencilState  = &depthStencil;
    pipelineInfo.pColorBlendState    = &colorBlend;
    pipelineInfo.pDynamicState       = &dynamicState;
    pipelineInfo.layout              = m_pipelineLayout;

    ENIGMA_VK_CHECK(vkCreateGraphicsPipelines(m_device->logical(), VK_NULL_HANDLE,
                                              1, &pipelineInfo, nullptr, &m_pipeline));

    vkDestroyShaderModule(m_device->logical(), taskMod, nullptr);
    vkDestroyShaderModule(m_device->logical(), meshMod, nullptr);
    vkDestroyShaderModule(m_device->logical(), fragMod, nullptr);

    ENIGMA_LOG_INFO("[visibility_buffer] mesh shader pipeline built");
}

void VisibilityBufferPass::rebuildPipeline() {
    ENIGMA_ASSERT(m_pipeline != VK_NULL_HANDLE);

    VkShaderModule taskMod = m_shaderManager->tryCompile(m_taskShaderPath, gfx::ShaderManager::Stage::Task, "ASMain");
    if (taskMod == VK_NULL_HANDLE) { ENIGMA_LOG_ERROR("[visibility_buffer] hot-reload: task compile failed"); return; }
    VkShaderModule meshMod = m_shaderManager->tryCompile(m_meshShaderPath, gfx::ShaderManager::Stage::Mesh, "MSMain");
    if (meshMod == VK_NULL_HANDLE) {
        ENIGMA_LOG_ERROR("[visibility_buffer] hot-reload: mesh compile failed");
        vkDestroyShaderModule(m_device->logical(), taskMod, nullptr); return;
    }
    VkShaderModule fragMod = m_shaderManager->tryCompile(m_meshShaderPath, gfx::ShaderManager::Stage::Fragment, "PSMain");
    if (fragMod == VK_NULL_HANDLE) {
        ENIGMA_LOG_ERROR("[visibility_buffer] hot-reload: frag compile failed");
        vkDestroyShaderModule(m_device->logical(), taskMod, nullptr);
        vkDestroyShaderModule(m_device->logical(), meshMod, nullptr); return;
    }

    vkDeviceWaitIdle(m_device->logical());
    vkDestroyPipeline(m_device->logical(), m_pipeline, nullptr);

    // Rebuild using the same pipeline layout (no need to recreate it).
    const std::array<VkPipelineShaderStageCreateInfo, 3> stages = {{
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
          VK_SHADER_STAGE_TASK_BIT_EXT, taskMod, "ASMain", nullptr },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
          VK_SHADER_STAGE_MESH_BIT_EXT, meshMod, "MSMain", nullptr },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
          VK_SHADER_STAGE_FRAGMENT_BIT, fragMod, "PSMain", nullptr },
    }};

    const std::array<VkDynamicState, 2> dynamicStates = {{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR }};
    VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamicState.dynamicStateCount = 2; dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewportState.viewportCount = 1; viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL; rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.blendEnable = VK_FALSE; blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlend{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    colorBlend.attachmentCount = 1; colorBlend.pAttachments = &blendAttachment;

    VkPipelineDepthStencilStateCreateInfo depthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    depthStencil.depthTestEnable = VK_TRUE; depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp  = VK_COMPARE_OP_GREATER_OR_EQUAL;

    const VkFormat visFormat = VK_FORMAT_R32_UINT;
    VkPipelineRenderingCreateInfo renderingInfo{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    renderingInfo.colorAttachmentCount = 1; renderingInfo.pColorAttachmentFormats = &visFormat;
    renderingInfo.depthAttachmentFormat = m_depthFormat;

    VkGraphicsPipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pipelineInfo.pNext = &renderingInfo;
    pipelineInfo.stageCount = 3; pipelineInfo.pStages = stages.data();
    pipelineInfo.pVertexInputState = nullptr; pipelineInfo.pInputAssemblyState = nullptr;
    pipelineInfo.pViewportState = &viewportState; pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisample; pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlend; pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = m_pipelineLayout;

    ENIGMA_VK_CHECK(vkCreateGraphicsPipelines(m_device->logical(), VK_NULL_HANDLE,
                                              1, &pipelineInfo, nullptr, &m_pipeline));

    vkDestroyShaderModule(m_device->logical(), taskMod, nullptr);
    vkDestroyShaderModule(m_device->logical(), meshMod, nullptr);
    vkDestroyShaderModule(m_device->logical(), fragMod, nullptr);

    ENIGMA_LOG_INFO("[visibility_buffer] hot-reload: pipeline rebuilt");
}

void VisibilityBufferPass::buildVsFallbackPipeline() {
    // Pipeline layout: reuse the same layout as the mesh shader path.
    // Push block is the same struct; VS ignores survivingIdsSlot/totalSurviving.
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset     = 0;
    pushRange.size       = sizeof(VBPushBlock);

    VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    layoutInfo.setLayoutCount         = 1;
    layoutInfo.pSetLayouts            = &m_globalSetLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges    = &pushRange;
    ENIGMA_VK_CHECK(vkCreatePipelineLayout(m_device->logical(), &layoutInfo, nullptr, &m_pipelineLayout));

    VkShaderModule vsMod = m_shaderManager->compile(m_vsShaderPath, gfx::ShaderManager::Stage::Vertex, "VSMain");
    VkShaderModule fsMod = m_shaderManager->compile(m_vsShaderPath, gfx::ShaderManager::Stage::Fragment, "PSMain");

    const std::array<VkPipelineShaderStageCreateInfo, 2> stages = {{
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
          VK_SHADER_STAGE_VERTEX_BIT,   vsMod, "VSMain", nullptr },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
          VK_SHADER_STAGE_FRAGMENT_BIT, fsMod, "PSMain", nullptr },
    }};

    // No vertex binding — all data read from SSBOs via SV_VertexID / SV_InstanceID.
    VkPipelineVertexInputStateCreateInfo vertexInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    const std::array<VkDynamicState, 2> dynamicStates = {{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR }};
    VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamicState.dynamicStateCount = 2; dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewportState.viewportCount = 1; viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL; rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.blendEnable = VK_FALSE; blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
    VkPipelineColorBlendStateCreateInfo colorBlend{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    colorBlend.attachmentCount = 1; colorBlend.pAttachments = &blendAttachment;

    VkPipelineDepthStencilStateCreateInfo depthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    depthStencil.depthTestEnable = VK_TRUE; depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp  = VK_COMPARE_OP_GREATER_OR_EQUAL;

    const VkFormat visFormat = VK_FORMAT_R32_UINT;
    VkPipelineRenderingCreateInfo renderingInfo{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    renderingInfo.colorAttachmentCount    = 1;
    renderingInfo.pColorAttachmentFormats = &visFormat;
    renderingInfo.depthAttachmentFormat   = m_depthFormat;

    VkGraphicsPipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pipelineInfo.pNext               = &renderingInfo;
    pipelineInfo.stageCount          = static_cast<u32>(stages.size());
    pipelineInfo.pStages             = stages.data();
    pipelineInfo.pVertexInputState   = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState   = &multisample;
    pipelineInfo.pDepthStencilState  = &depthStencil;
    pipelineInfo.pColorBlendState    = &colorBlend;
    pipelineInfo.pDynamicState       = &dynamicState;
    pipelineInfo.layout              = m_pipelineLayout;

    ENIGMA_VK_CHECK(vkCreateGraphicsPipelines(m_device->logical(), VK_NULL_HANDLE,
                                              1, &pipelineInfo, nullptr, &m_vsFallbackPipeline));

    vkDestroyShaderModule(m_device->logical(), vsMod, nullptr);
    vkDestroyShaderModule(m_device->logical(), fsMod, nullptr);
}

void VisibilityBufferPass::buildVsFallbackDraws(const GpuSceneBuffer& scene,
                                                const GpuMeshletBuffer& meshlets) {
    ENIGMA_ASSERT(!m_useMeshShaders && "buildVsFallbackDraws called on mesh-shader-capable device");

    const auto& instances    = scene.cpu_instances();
    const auto& cpuMeshlets  = meshlets.cpu_meshlets();

    // Generate one VkDrawIndirectCommand per (instance, meshlet_local_idx).
    // firstInstance encodes instance_id in the upper 16 bits and meshlet_local_idx in the lower.
    std::vector<VkDrawIndirectCommand> cmds;
    for (u32 instIdx = 0; instIdx < static_cast<u32>(instances.size()); ++instIdx) {
        const GpuInstance& inst = instances[instIdx];
        for (u32 localIdx = 0; localIdx < inst.meshlet_count; ++localIdx) {
            const u32 globalIdx = inst.meshlet_offset + localIdx;
            if (globalIdx >= static_cast<u32>(cpuMeshlets.size())) break;
            const Meshlet& m = cpuMeshlets[globalIdx];

            VkDrawIndirectCommand cmd{};
            cmd.vertexCount   = m.triangle_count * 3;
            cmd.instanceCount = 1;
            cmd.firstVertex   = 0;
            cmd.firstInstance = (instIdx << 16) | (localIdx & 0xFFFF);
            cmds.push_back(cmd);
        }
    }
    m_vsFallbackDrawCount = static_cast<u32>(cmds.size());

    if (cmds.empty()) {
        ENIGMA_LOG_WARN("[visibility_buffer] VS fallback: no draw commands generated");
        return;
    }

    // Upload to device-local buffer via staging.
    const VkDeviceSize bufSize = cmds.size() * sizeof(VkDrawIndirectCommand);

    VkBuffer      staging;
    VmaAllocation stagingAlloc;
    {
        VkBufferCreateInfo bufCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufCI.size        = bufSize;
        bufCI.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo allocCI{};
        allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        allocCI.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT
                      | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        VmaAllocationInfo info{};
        ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &bufCI, &allocCI,
                                        &staging, &stagingAlloc, &info));
        std::memcpy(info.pMappedData, cmds.data(), static_cast<size_t>(bufSize));
        vmaFlushAllocation(m_allocator->handle(), stagingAlloc, 0, VK_WHOLE_SIZE);
    }

    {
        VkBufferCreateInfo bufCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufCI.size        = bufSize;
        bufCI.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo allocCI{};
        allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &bufCI, &allocCI,
                                        &m_vsFallbackDrawBuffer, &m_vsFallbackDrawAlloc, nullptr));
    }

    // Immediate copy via one-shot command buffer on the graphics queue.
    VkCommandBuffer cmd;
    {
        VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        // Use the internal command pool — borrow from device via a temporary pool.
        VkCommandPoolCreateInfo poolCI{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        poolCI.queueFamilyIndex = m_device->graphicsQueueFamily();
        poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        VkCommandPool tempPool  = VK_NULL_HANDLE;
        ENIGMA_VK_CHECK(vkCreateCommandPool(m_device->logical(), &poolCI, nullptr, &tempPool));

        allocInfo.commandPool        = tempPool;
        allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        ENIGMA_VK_CHECK(vkAllocateCommandBuffers(m_device->logical(), &allocInfo, &cmd));

        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        VkBufferCopy region{ 0, 0, bufSize };
        vkCmdCopyBuffer(cmd, staging, m_vsFallbackDrawBuffer, 1, &region);

        vkEndCommandBuffer(cmd);

        VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers    = &cmd;
        vkQueueSubmit(m_device->graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_device->graphicsQueue());

        vkDestroyCommandPool(m_device->logical(), tempPool, nullptr);
    }

    vmaDestroyBuffer(m_allocator->handle(), staging, stagingAlloc);

    ENIGMA_LOG_INFO("[visibility_buffer] VS fallback: {} draw commands uploaded", m_vsFallbackDrawCount);
}

void VisibilityBufferPass::registerHotReload(gfx::ShaderHotReload& reloader) {
    if (m_useMeshShaders) {
        ENIGMA_ASSERT(m_pipeline != VK_NULL_HANDLE);
        reloader.watchGroup({m_taskShaderPath, m_meshShaderPath},
                            [this]() { rebuildPipeline(); });
    } else {
        ENIGMA_ASSERT(m_vsFallbackPipeline != VK_NULL_HANDLE);
        reloader.watchGroup({m_vsShaderPath}, [this]() {
            // Probe-compile to fail fast before tearing down the existing pipeline.
            VkShaderModule probeVs = m_shaderManager->tryCompile(m_vsShaderPath,
                                         gfx::ShaderManager::Stage::Vertex, "VSMain");
            if (probeVs == VK_NULL_HANDLE) { ENIGMA_LOG_ERROR("[visibility_buffer] VS hot-reload: VS compile failed"); return; }
            VkShaderModule probeFs = m_shaderManager->tryCompile(m_vsShaderPath,
                                         gfx::ShaderManager::Stage::Fragment, "PSMain");
            if (probeFs == VK_NULL_HANDLE) {
                vkDestroyShaderModule(m_device->logical(), probeVs, nullptr);
                ENIGMA_LOG_ERROR("[visibility_buffer] VS hot-reload: PS compile failed"); return;
            }
            vkDestroyShaderModule(m_device->logical(), probeVs, nullptr);
            vkDestroyShaderModule(m_device->logical(), probeFs, nullptr);

            vkDeviceWaitIdle(m_device->logical());
            vkDestroyPipeline(m_device->logical(), m_vsFallbackPipeline, nullptr);
            m_vsFallbackPipeline = VK_NULL_HANDLE;
            buildVsFallbackPipeline();
            ENIGMA_LOG_INFO("[visibility_buffer] VS fallback hot-reload: pipeline rebuilt");
        });
    }
}

void VisibilityBufferPass::buildWireframePipeline(gfx::ShaderManager& shaderManager,
                                                   VkDescriptorSetLayout globalSetLayout,
                                                   VkFormat swapchainFormat) {
    if (!m_useMeshShaders) {
        ENIGMA_LOG_INFO("[visibility_buffer] wireframe skipped (no mesh shader support)");
        return;
    }

    m_wireFragShaderPath = Paths::shaderSourceDir() / "debug_wireframe.frag.hlsl";

    VkShaderModule taskMod = shaderManager.compile(m_taskShaderPath, gfx::ShaderManager::Stage::Task, "ASMain");
    VkShaderModule meshMod = shaderManager.compile(m_meshShaderPath, gfx::ShaderManager::Stage::Mesh, "MSMain");
    VkShaderModule fragMod = shaderManager.compile(m_wireFragShaderPath, gfx::ShaderManager::Stage::Fragment, "PSMain");

    // Two push constant ranges:
    //   Range 0: task+mesh stages — VBPushBlock (32 bytes at offset 0)
    //   Range 1: fragment stage   — wireColor float3 + pad (16 bytes at offset 32)
    const VkPushConstantRange ranges[2] = {
        { VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, 0, 32 },
        { VK_SHADER_STAGE_FRAGMENT_BIT, 32, 16 },
    };

    VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    layoutInfo.setLayoutCount         = 1;
    layoutInfo.pSetLayouts            = &globalSetLayout;
    layoutInfo.pushConstantRangeCount = 2;
    layoutInfo.pPushConstantRanges    = ranges;

    ENIGMA_VK_CHECK(vkCreatePipelineLayout(m_device->logical(), &layoutInfo, nullptr,
                                           &m_wireframePipelineLayout));

    const std::array<VkPipelineShaderStageCreateInfo, 3> stages = {{
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
          VK_SHADER_STAGE_TASK_BIT_EXT, taskMod, "ASMain", nullptr },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
          VK_SHADER_STAGE_MESH_BIT_EXT, meshMod, "MSMain", nullptr },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
          VK_SHADER_STAGE_FRAGMENT_BIT, fragMod, "PSMain", nullptr },
    }};

    const std::array<VkDynamicState, 2> dynamicStates = {{
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
    }};
    VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates    = dynamicStates.data();

    VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rasterizer.polygonMode = VK_POLYGON_MODE_LINE;  // hardware wireframe
    rasterizer.cullMode    = VK_CULL_MODE_NONE;
    rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.blendEnable         = VK_TRUE;
    blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.colorBlendOp        = VK_BLEND_OP_ADD;
    blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    blendAttachment.alphaBlendOp        = VK_BLEND_OP_ADD;
    blendAttachment.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                        | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlend{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments    = &blendAttachment;

    // No depth: wireframe draws on top (no depth test for debug overlay).
    VkPipelineRenderingCreateInfo renderingInfo{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    renderingInfo.colorAttachmentCount    = 1;
    renderingInfo.pColorAttachmentFormats = &swapchainFormat;
    renderingInfo.depthAttachmentFormat   = VK_FORMAT_UNDEFINED;

    VkGraphicsPipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pipelineInfo.pNext               = &renderingInfo;
    pipelineInfo.stageCount          = static_cast<u32>(stages.size());
    pipelineInfo.pStages             = stages.data();
    pipelineInfo.pVertexInputState   = nullptr;
    pipelineInfo.pInputAssemblyState = nullptr;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState   = &multisample;
    pipelineInfo.pDepthStencilState  = nullptr;
    pipelineInfo.pColorBlendState    = &colorBlend;
    pipelineInfo.pDynamicState       = &dynamicState;
    pipelineInfo.layout              = m_wireframePipelineLayout;

    ENIGMA_VK_CHECK(vkCreateGraphicsPipelines(m_device->logical(), VK_NULL_HANDLE,
                                              1, &pipelineInfo, nullptr, &m_wireframePipeline));

    vkDestroyShaderModule(m_device->logical(), taskMod, nullptr);
    vkDestroyShaderModule(m_device->logical(), meshMod, nullptr);
    vkDestroyShaderModule(m_device->logical(), fragMod, nullptr);

    ENIGMA_LOG_INFO("[visibility_buffer] wireframe pipeline built (VK_POLYGON_MODE_LINE)");
}

void VisibilityBufferPass::recordWireframe(VkCommandBuffer cmd,
                                            VkDescriptorSet globalSet,
                                            VkExtent2D extent,
                                            const GpuSceneBuffer& scene,
                                            const GpuMeshletBuffer& meshlets,
                                            const IndirectDrawBuffer& indirect,
                                            u32 cameraSlot,
                                            vec3 wireColor) {
    if (m_wireframePipeline == VK_NULL_HANDLE) return;
    if (!m_useMeshShaders)                      return;
    if (meshlets.total_meshlet_count() == 0)    return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_wireframePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_wireframePipelineLayout, 0, 1, &globalSet, 0, nullptr);

    // Push VBPushBlock for task+mesh stages.
    VBPushBlock pc{};
    pc.instanceBufferSlot   = scene.slot();
    pc.meshletBufferSlot    = meshlets.meshlets_slot();
    pc.survivingIdsSlot     = indirect.surviving_slot();
    pc.meshletVerticesSlot  = meshlets.vertices_slot();
    pc.meshletTrianglesSlot = meshlets.triangles_slot();
    pc.cameraSlot           = cameraSlot;
    pc.countBufferSlot      = indirect.count_slot();
    pc.instanceCount        = static_cast<u32>(scene.instance_count());

    vkCmdPushConstants(cmd, m_wireframePipelineLayout,
                       VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
                       0, sizeof(pc), &pc);

    // Push wireframe color for fragment stage.
    struct WireColorPush { float r, g, b, _pad; };
    WireColorPush wcPush{ wireColor.x, wireColor.y, wireColor.z, 1.0f };
    vkCmdPushConstants(cmd, m_wireframePipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       32, sizeof(wcPush), &wcPush);

    VkViewport viewport{};
    viewport.width    = static_cast<float>(extent.width);
    viewport.height   = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{ {0, 0}, extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    const u32 totalMeshlets = meshlets.total_meshlet_count();
    const u32 taskGroups    = (totalMeshlets + TASK_GROUP_SIZE - 1) / TASK_GROUP_SIZE;
    vkCmdDrawMeshTasksEXT(cmd, taskGroups, 1, 1);
}

void VisibilityBufferPass::record(VkCommandBuffer           cmd,
                                   VkDescriptorSet           globalSet,
                                   VkExtent2D                extent,
                                   VkImageView               depthView,
                                   VkImage                   depthImage,
                                   const GpuSceneBuffer&     scene,
                                   const GpuMeshletBuffer&   meshlets,
                                   const IndirectDrawBuffer& indirect,
                                   u32                       cameraSlot,
                                   bool                      clearFirst) {
    const bool usingMeshShaders = m_useMeshShaders;
    ENIGMA_ASSERT((usingMeshShaders ? m_pipeline != VK_NULL_HANDLE
                                    : m_vsFallbackPipeline != VK_NULL_HANDLE)
                  && "VisibilityBufferPass::record before buildPipeline");
    ENIGMA_ASSERT(m_vis_image != VK_NULL_HANDLE && "VisibilityBufferPass::record before allocate");

    // -------------------------------------------------------------------
    // Pre-pass image layout transitions.
    //   clearFirst=true : UNDEFINED → COLOR/DEPTH  (no prior data this frame)
    //   clearFirst=false: SHADER_READ_ONLY → COLOR/DEPTH  (terrain drew first;
    //     its post-barrier left both images in SHADER_READ_ONLY_OPTIMAL)
    // -------------------------------------------------------------------
    if (clearFirst) {
        const VkImageMemoryBarrier2 preBarriers[2] = {
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, nullptr,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                m_vis_image, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, nullptr,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                depthImage, { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 }
            },
        };
        VkDependencyInfo preDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        preDep.imageMemoryBarrierCount = 2;
        preDep.pImageMemoryBarriers    = preBarriers;
        vkCmdPipelineBarrier2(cmd, &preDep);
    } else {
        // Terrain's post-barrier left both images in SHADER_READ_ONLY_OPTIMAL.
        const VkImageMemoryBarrier2 preBarriers[2] = {
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, nullptr,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                m_vis_image, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, nullptr,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                depthImage, { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 }
            },
        };
        VkDependencyInfo preDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        preDep.imageMemoryBarrierCount = 2;
        preDep.pImageMemoryBarriers    = preBarriers;
        vkCmdPipelineBarrier2(cmd, &preDep);
    }

    // -------------------------------------------------------------------
    // Begin dynamic rendering.
    // -------------------------------------------------------------------
    VkClearValue visClear{};
    visClear.color.uint32[0] = 0xFFFFFFFFu; // INVALID_VIS sentinel

    VkClearValue depthClear{};
    depthClear.depthStencil.depth = 0.0f; // reverse-Z: far = 0

    VkRenderingAttachmentInfo visAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    visAttachment.imageView   = m_vis_view;
    visAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    visAttachment.loadOp      = clearFirst ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    visAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    visAttachment.clearValue  = visClear; // only used when loadOp=CLEAR

    VkRenderingAttachmentInfo depthAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    depthAttachment.imageView   = depthView;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp      = clearFirst ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    depthAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue  = depthClear; // only used when loadOp=CLEAR

    VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
    renderingInfo.renderArea           = { {0, 0}, extent };
    renderingInfo.layerCount           = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments    = &visAttachment;
    renderingInfo.pDepthAttachment     = &depthAttachment;

    vkCmdBeginRendering(cmd, &renderingInfo);

    VkViewport viewport{};
    viewport.width    = static_cast<float>(extent.width);
    viewport.height   = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{ {0, 0}, extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    VBPushBlock pc{};
    pc.instanceBufferSlot   = scene.slot();
    pc.meshletBufferSlot    = meshlets.meshlets_slot();
    pc.survivingIdsSlot     = indirect.surviving_slot();
    pc.meshletVerticesSlot  = meshlets.vertices_slot();
    pc.meshletTrianglesSlot = meshlets.triangles_slot();
    pc.cameraSlot           = cameraSlot;
    pc.countBufferSlot      = indirect.count_slot();
    pc.instanceCount        = static_cast<u32>(scene.instance_count());

    if (usingMeshShaders) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                m_pipelineLayout, 0, 1, &globalSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_pipelineLayout,
                           VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT
                           | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(pc), &pc);
        // Dispatch ceil(totalMeshlets / TASK_GROUP_SIZE) task groups. Each group
        // processes up to TASK_GROUP_SIZE surviving meshlets read from the
        // surviving IDs buffer, using the GPU count buffer for the valid range.
        const u32 totalMeshlets = meshlets.total_meshlet_count();
        const u32 taskGroups    = (totalMeshlets + TASK_GROUP_SIZE - 1) / TASK_GROUP_SIZE;
        vkCmdDrawMeshTasksEXT(cmd, taskGroups, 1, 1);
    } else {
        // VS fallback: draw each meshlet as a direct indexed triangle draw.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_vsFallbackPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                m_pipelineLayout, 0, 1, &globalSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(pc), &pc);
        if (m_vsFallbackDrawBuffer != VK_NULL_HANDLE && m_vsFallbackDrawCount > 0) {
            vkCmdDrawIndirect(cmd, m_vsFallbackDrawBuffer, 0,
                              m_vsFallbackDrawCount, sizeof(VkDrawIndirectCommand));
        }
    }

    vkCmdEndRendering(cmd);

    // -------------------------------------------------------------------
    // Post-pass transitions:
    //   vis buffer  COLOR_ATTACHMENT_OPTIMAL  → SHADER_READ_ONLY_OPTIMAL
    //   depth buffer DEPTH_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
    // Both are needed by MaterialEvalPass (compute) and other downstream
    // passes that sample them. Depth must also be declared correctly in the
    // Renderer render-graph importImage calls.
    // -------------------------------------------------------------------
    const VkImageMemoryBarrier2 postBarriers[2] = {
        {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_vis_image, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
        },
        {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            depthImage, { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 }
        },
    };

    VkDependencyInfo postDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    postDep.imageMemoryBarrierCount = 2;
    postDep.pImageMemoryBarriers    = postBarriers;
    vkCmdPipelineBarrier2(cmd, &postDep);
}

// ---------------------------------------------------------------------------
// CDLOD terrain pipeline — identical pipeline layout semantics to record()'s
// scene pipeline but different shaders and a larger push-constant range.
// ---------------------------------------------------------------------------

void VisibilityBufferPass::buildTerrainPipeline(gfx::ShaderManager& shaderManager,
                                                 VkDescriptorSetLayout globalSetLayout) {
    ENIGMA_ASSERT(m_terrainPipeline == VK_NULL_HANDLE &&
                  "VisibilityBufferPass::buildTerrainPipeline called twice");
    if (!m_useMeshShaders) {
        ENIGMA_LOG_WARN("[visibility_buffer] terrain pipeline skipped (no mesh shader support)");
        return;
    }

    m_shaderManager         = &shaderManager;
    m_globalSetLayout       = globalSetLayout;
    m_terrainTaskShaderPath = Paths::shaderSourceDir() / "terrain_cdlod.task.hlsl";
    m_terrainMeshShaderPath = Paths::shaderSourceDir() / "terrain_cdlod.mesh.hlsl";

    VkShaderModule taskMod = shaderManager.compile(m_terrainTaskShaderPath, gfx::ShaderManager::Stage::Task, "ASMain");
    VkShaderModule meshMod = shaderManager.compile(m_terrainMeshShaderPath, gfx::ShaderManager::Stage::Mesh, "MSMain");
    VkShaderModule fragMod = shaderManager.compile(m_terrainMeshShaderPath, gfx::ShaderManager::Stage::Fragment, "PSMain");

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT
                         | VK_SHADER_STAGE_MESH_BIT_EXT
                         | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset     = 0;
    pushRange.size       = sizeof(VBTerrainPushBlock);

    VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    layoutInfo.setLayoutCount         = 1;
    layoutInfo.pSetLayouts            = &globalSetLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges    = &pushRange;

    ENIGMA_VK_CHECK(vkCreatePipelineLayout(m_device->logical(), &layoutInfo, nullptr,
                                           &m_terrainPipelineLayout));

    const std::array<VkPipelineShaderStageCreateInfo, 3> stages = {{
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
          VK_SHADER_STAGE_TASK_BIT_EXT, taskMod, "ASMain", nullptr },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
          VK_SHADER_STAGE_MESH_BIT_EXT, meshMod, "MSMain", nullptr },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
          VK_SHADER_STAGE_FRAGMENT_BIT, fragMod, "PSMain", nullptr },
    }};

    const std::array<VkDynamicState, 2> dynamicStates = {{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR }};
    VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamicState.dynamicStateCount = 2; dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewportState.viewportCount = 1; viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL; rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.blendEnable = VK_FALSE; blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
    VkPipelineColorBlendStateCreateInfo colorBlend{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    colorBlend.attachmentCount = 1; colorBlend.pAttachments = &blendAttachment;

    VkPipelineDepthStencilStateCreateInfo depthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    depthStencil.depthTestEnable = VK_TRUE; depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp  = VK_COMPARE_OP_GREATER_OR_EQUAL; // reverse-Z: scene now runs after terrain so GREATER_OR_EQUAL is correct

    const VkFormat visFormat = VK_FORMAT_R32_UINT;
    VkPipelineRenderingCreateInfo renderingInfo{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    renderingInfo.colorAttachmentCount = 1; renderingInfo.pColorAttachmentFormats = &visFormat;
    renderingInfo.depthAttachmentFormat = m_depthFormat;

    VkGraphicsPipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pipelineInfo.pNext               = &renderingInfo;
    pipelineInfo.stageCount          = static_cast<u32>(stages.size());
    pipelineInfo.pStages             = stages.data();
    pipelineInfo.pVertexInputState   = nullptr;
    pipelineInfo.pInputAssemblyState = nullptr;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState   = &multisample;
    pipelineInfo.pDepthStencilState  = &depthStencil;
    pipelineInfo.pColorBlendState    = &colorBlend;
    pipelineInfo.pDynamicState       = &dynamicState;
    pipelineInfo.layout              = m_terrainPipelineLayout;

    ENIGMA_VK_CHECK(vkCreateGraphicsPipelines(m_device->logical(), VK_NULL_HANDLE,
                                              1, &pipelineInfo, nullptr, &m_terrainPipeline));

    vkDestroyShaderModule(m_device->logical(), taskMod, nullptr);
    vkDestroyShaderModule(m_device->logical(), meshMod, nullptr);
    vkDestroyShaderModule(m_device->logical(), fragMod, nullptr);

    ENIGMA_LOG_INFO("[visibility_buffer] terrain pipeline built");
}

void VisibilityBufferPass::recordTerrain(VkCommandBuffer           cmd,
                                          VkDescriptorSet           globalSet,
                                          VkExtent2D                extent,
                                          VkImageView               depthView,
                                          VkImage                   depthImage,
                                          const GpuSceneBuffer&     scene,
                                          const GpuMeshletBuffer&   meshlets,
                                          const IndirectDrawBuffer& indirect,
                                          u32                       cameraSlot,
                                          u32                       topologyVerticesSlot,
                                          u32                       topologyTrianglesSlot,
                                          u32                       survivingMeshletCount) {
    if (m_terrainPipeline == VK_NULL_HANDLE) return;
    if (!m_useMeshShaders)                   return;
    if (survivingMeshletCount == 0)          return;
    ENIGMA_ASSERT(m_vis_image != VK_NULL_HANDLE && "recordTerrain before allocate");

    // --------- Pre-pass: UNDEFINED → COLOR/DEPTH_ATTACHMENT_OPTIMAL -----------
    // recordTerrain() now runs FIRST this frame (terrain drawn before scene so
    // the scene pass renders on top and always wins coplanar depth fights).
    // UNDEFINED old-layout discards prior contents — fine because we CLEAR.
    const VkImageMemoryBarrier2 preBarriers[2] = {
        {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_vis_image, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
        },
        {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
            | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            depthImage, { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 }
        },
    };
    VkDependencyInfo preDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    preDep.imageMemoryBarrierCount = 2;
    preDep.pImageMemoryBarriers    = preBarriers;
    vkCmdPipelineBarrier2(cmd, &preDep);

    // --------- Begin dynamic rendering — CLEAR initialises vis + depth ------
    VkClearValue visClear{};
    visClear.color.uint32[0] = 0xFFFFFFFFu;
    VkClearValue depthClear{};
    depthClear.depthStencil.depth = 0.0f;

    VkRenderingAttachmentInfo visAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    visAttachment.imageView   = m_vis_view;
    visAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    visAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    visAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    visAttachment.clearValue  = visClear;

    VkRenderingAttachmentInfo depthAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    depthAttachment.imageView   = depthView;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue  = depthClear;

    VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
    renderingInfo.renderArea           = { {0, 0}, extent };
    renderingInfo.layerCount           = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments    = &visAttachment;
    renderingInfo.pDepthAttachment     = &depthAttachment;

    vkCmdBeginRendering(cmd, &renderingInfo);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_terrainPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_terrainPipelineLayout, 0, 1, &globalSet, 0, nullptr);

    VkViewport viewport{};
    viewport.width    = static_cast<float>(extent.width);
    viewport.height   = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{ {0, 0}, extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    VBTerrainPushBlock pc{};
    pc.instanceBufferSlot    = scene.slot();
    pc.meshletBufferSlot     = meshlets.meshlets_slot();
    pc.survivingIdsSlot      = indirect.surviving_slot();
    pc.meshletVerticesSlot   = meshlets.vertices_slot();
    pc.meshletTrianglesSlot  = meshlets.triangles_slot();
    pc.cameraSlot            = cameraSlot;
    pc.countBufferSlot       = indirect.count_slot();
    pc.instanceCount         = static_cast<u32>(scene.instance_count());
    pc.topologyVerticesSlot  = topologyVerticesSlot;
    pc.topologyTrianglesSlot = topologyTrianglesSlot;

    vkCmdPushConstants(cmd, m_terrainPipelineLayout,
                       VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT
                       | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);

    // Dispatch ceil(survivingMeshletCount / TASK_GROUP_SIZE) task groups.
    // The task shader also reads the actual surviving count from the count
    // buffer — the CPU-side argument here is only the dispatch size.
    const u32 taskGroups = (survivingMeshletCount + TASK_GROUP_SIZE - 1) / TASK_GROUP_SIZE;
    vkCmdDrawMeshTasksEXT(cmd, taskGroups, 1, 1);

    vkCmdEndRendering(cmd);

    // --------- Post-pass: COLOR/DEPTH_ATTACHMENT → SHADER_READ_ONLY ---------
    // Matches the post state left by record() so MaterialEvalPass (the reader
    // after this function) can sample both vis and depth as textures.
    const VkImageMemoryBarrier2 postBarriers[2] = {
        {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_vis_image, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
        },
        {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            depthImage, { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 }
        },
    };
    VkDependencyInfo postDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    postDep.imageMemoryBarrierCount = 2;
    postDep.pImageMemoryBarriers    = postBarriers;
    vkCmdPipelineBarrier2(cmd, &postDep);
}

// ---------------------------------------------------------------------------
// buildTerrainWireframePipeline / recordTerrainWireframe
// ---------------------------------------------------------------------------

void VisibilityBufferPass::buildTerrainWireframePipeline(gfx::ShaderManager& shaderManager,
                                                          VkDescriptorSetLayout globalSetLayout,
                                                          VkFormat swapchainFormat) {
    if (!m_useMeshShaders) {
        ENIGMA_LOG_INFO("[visibility_buffer] terrain wireframe skipped (no mesh shader support)");
        return;
    }
    // Requires buildTerrainPipeline() to have run first (sets terrain shader paths).
    if (m_terrainTaskShaderPath.empty()) {
        ENIGMA_LOG_WARN("[visibility_buffer] buildTerrainWireframePipeline: call buildTerrainPipeline first");
        return;
    }

    // Terrain wireframe uses a dedicated frag shader with [[vk::offset(40)]] so
    // the color push constant aligns with VBTerrainPushBlock (40 bytes).
    const auto terrainWireFragPath = Paths::shaderSourceDir() / "debug_wireframe_terrain.frag.hlsl";

    VkShaderModule taskMod = shaderManager.compile(m_terrainTaskShaderPath, gfx::ShaderManager::Stage::Task, "ASMain");
    VkShaderModule meshMod = shaderManager.compile(m_terrainMeshShaderPath, gfx::ShaderManager::Stage::Mesh, "MSMain");
    VkShaderModule fragMod = shaderManager.compile(terrainWireFragPath,     gfx::ShaderManager::Stage::Fragment, "PSMain");

    // Two push constant ranges:
    //   Range 0: task+mesh — VBTerrainPushBlock (40 bytes at offset 0)
    //   Range 1: fragment  — wireColor float3 + pad (16 bytes at offset 40)
    const VkPushConstantRange ranges[2] = {
        { VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, 0, 40 },
        { VK_SHADER_STAGE_FRAGMENT_BIT, 40, 16 },
    };

    VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    layoutInfo.setLayoutCount         = 1;
    layoutInfo.pSetLayouts            = &globalSetLayout;
    layoutInfo.pushConstantRangeCount = 2;
    layoutInfo.pPushConstantRanges    = ranges;

    ENIGMA_VK_CHECK(vkCreatePipelineLayout(m_device->logical(), &layoutInfo, nullptr,
                                           &m_terrainWireframePipelineLayout));

    const std::array<VkPipelineShaderStageCreateInfo, 3> stages = {{
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
          VK_SHADER_STAGE_TASK_BIT_EXT, taskMod, "ASMain", nullptr },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
          VK_SHADER_STAGE_MESH_BIT_EXT, meshMod, "MSMain", nullptr },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
          VK_SHADER_STAGE_FRAGMENT_BIT, fragMod, "PSMain", nullptr },
    }};

    const std::array<VkDynamicState, 2> dynamicStates = {{
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
    }};
    VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates    = dynamicStates.data();

    VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
    rasterizer.cullMode    = VK_CULL_MODE_NONE;
    rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.blendEnable         = VK_TRUE;
    blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.colorBlendOp        = VK_BLEND_OP_ADD;
    blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    blendAttachment.alphaBlendOp        = VK_BLEND_OP_ADD;
    blendAttachment.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                        | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlend{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments    = &blendAttachment;

    VkPipelineRenderingCreateInfo renderingInfo{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    renderingInfo.colorAttachmentCount    = 1;
    renderingInfo.pColorAttachmentFormats = &swapchainFormat;
    renderingInfo.depthAttachmentFormat   = VK_FORMAT_UNDEFINED;

    VkGraphicsPipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pipelineInfo.pNext               = &renderingInfo;
    pipelineInfo.stageCount          = static_cast<u32>(stages.size());
    pipelineInfo.pStages             = stages.data();
    pipelineInfo.pVertexInputState   = nullptr;
    pipelineInfo.pInputAssemblyState = nullptr;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState   = &multisample;
    pipelineInfo.pDepthStencilState  = nullptr;
    pipelineInfo.pColorBlendState    = &colorBlend;
    pipelineInfo.pDynamicState       = &dynamicState;
    pipelineInfo.layout              = m_terrainWireframePipelineLayout;

    ENIGMA_VK_CHECK(vkCreateGraphicsPipelines(m_device->logical(), VK_NULL_HANDLE,
                                              1, &pipelineInfo, nullptr, &m_terrainWireframePipeline));

    vkDestroyShaderModule(m_device->logical(), taskMod, nullptr);
    vkDestroyShaderModule(m_device->logical(), meshMod, nullptr);
    vkDestroyShaderModule(m_device->logical(), fragMod, nullptr);

    ENIGMA_LOG_INFO("[visibility_buffer] terrain wireframe pipeline built (VK_POLYGON_MODE_LINE)");
}

void VisibilityBufferPass::recordTerrainWireframe(VkCommandBuffer           cmd,
                                                   VkDescriptorSet           globalSet,
                                                   VkExtent2D                extent,
                                                   const GpuSceneBuffer&     scene,
                                                   const GpuMeshletBuffer&   meshlets,
                                                   const IndirectDrawBuffer& terrainIndirect,
                                                   u32                       cameraSlot,
                                                   u32                       topologyVerticesSlot,
                                                   u32                       topologyTrianglesSlot,
                                                   u32                       terrainMeshletCount,
                                                   vec3                      wireColor) {
    if (m_terrainWireframePipeline == VK_NULL_HANDLE) return;
    if (!m_useMeshShaders)                             return;
    if (terrainMeshletCount == 0)                      return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_terrainWireframePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_terrainWireframePipelineLayout, 0, 1, &globalSet, 0, nullptr);

    VBTerrainPushBlock pc{};
    pc.instanceBufferSlot    = scene.slot();
    pc.meshletBufferSlot     = meshlets.meshlets_slot();
    pc.survivingIdsSlot      = terrainIndirect.surviving_slot();
    pc.meshletVerticesSlot   = meshlets.vertices_slot();
    pc.meshletTrianglesSlot  = meshlets.triangles_slot();
    pc.cameraSlot            = cameraSlot;
    pc.countBufferSlot       = terrainIndirect.count_slot();
    pc.instanceCount         = static_cast<u32>(scene.instance_count());
    pc.topologyVerticesSlot  = topologyVerticesSlot;
    pc.topologyTrianglesSlot = topologyTrianglesSlot;

    vkCmdPushConstants(cmd, m_terrainWireframePipelineLayout,
                       VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
                       0, sizeof(pc), &pc);

    struct WireColorPush { float r, g, b, _pad; };
    WireColorPush wcPush{ wireColor.x, wireColor.y, wireColor.z, 1.0f };
    vkCmdPushConstants(cmd, m_terrainWireframePipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       40, sizeof(wcPush), &wcPush);

    VkViewport viewport{};
    viewport.width    = static_cast<float>(extent.width);
    viewport.height   = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{ {0, 0}, extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    const u32 taskGroups = (terrainMeshletCount + TASK_GROUP_SIZE - 1) / TASK_GROUP_SIZE;
    vkCmdDrawMeshTasksEXT(cmd, taskGroups, 1, 1);
}

} // namespace enigma
