#include "renderer/Renderer.h"

#include "core/Assert.h"
#include "core/Log.h"
#include <imgui.h>
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"
#include "gfx/FrameContext.h"
#include "gfx/Instance.h"
#include "gfx/GpuProfiler.h"
#include "gfx/RenderGraph.h"
#include "gfx/ShaderHotReload.h"
#include "gfx/ShaderManager.h"
#include "gfx/Swapchain.h"
#include "gfx/Validation.h"
#include "platform/Window.h"
#include "renderer/GBufferFormats.h"
#include "renderer/LightingPass.h"
#include "renderer/MeshPass.h"
#include "renderer/RTReflectionPass.h"
#include "renderer/RTGIPass.h"
#include "renderer/RTShadowPass.h"
#include "renderer/WetRoadPass.h"
#include "renderer/Denoiser.h"
#include "renderer/ClusteredForwardPass.h"
#include "renderer/GpuCullPass.h"
#include "renderer/GpuMeshletBuffer.h"
#include "renderer/GpuSceneBuffer.h"
#include "renderer/HiZPass.h"
#include "renderer/IndirectDrawBuffer.h"
#include "renderer/MaterialEvalPass.h"
#include "renderer/TrianglePass.h"
#include "renderer/UpscalerFactory.h"
#include "renderer/DebugVisualizationPass.h"
#include "renderer/VisibilityBufferPass.h"
#include "renderer/micropoly/MicropolyCapability.h"
#include "renderer/micropoly/MicropolyCullPass.h"
#include "renderer/micropoly/MicropolyStreaming.h"
#include "asset/MpAssetReader.h"
#include "asset/MpPathUtils.h"
#include "scene/Camera.h"
#include "scene/Scene.h"

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

#include <algorithm>
#include <cmath>
#include <cstring>

namespace enigma {

namespace {

// Emits the COMPUTE -> FRAGMENT vis-image memory barrier that every
// micropoly debug overlay needs (HW raster writes in FRAGMENT, SW raster
// writes in COMPUTE; the overlays all read the vis image in FRAGMENT).
//
// Must run OUTSIDE any dynamic-rendering instance (Vulkan
// VUID-vkCmdPipelineBarrier2-None-09553). The render graph only wraps a
// pass in vkCmdBeginRendering when colorTargets/depthTarget is populated,
// so we enqueue a barrier-only pass with NO attachments; the graph
// treats it as compute and skips the render-pass wrapper.
//
// Call BEFORE the raster pass that reads the vis image.
void addMicropolyVisBarrierPass(gfx::RenderGraph& graph) {
    gfx::RenderGraph::RasterPassDesc barrier{};
    barrier.name = "MicropolyVisBarrierCompute2Fragment";
    // Intentionally no colorTargets / depthTarget — no render-pass wrapper.
    barrier.execute = [](VkCommandBuffer cmd, VkExtent2D /*ext*/) {
        VkMemoryBarrier2 visToFrag{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        visToFrag.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        visToFrag.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                                | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        visToFrag.dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        visToFrag.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.memoryBarrierCount = 1u;
        dep.pMemoryBarriers    = &visToFrag;
        vkCmdPipelineBarrier2(cmd, &dep);
    };
    graph.addRasterPass(std::move(barrier));
}

// Convert spherical sun angles (degrees) to a unit world-space direction
// pointing FROM the surface TO the sun (Y-up world, azimuth clockwise from Z+).
// This is the ONLY place az/el are used — all shaders receive the resulting vec3.
vec3 fromAzimuthElevation(f32 azimuthDeg, f32 elevationDeg) {
    const f32 az = glm::radians(azimuthDeg);
    const f32 el = glm::radians(elevationDeg);
    return glm::normalize(vec3{
        glm::cos(el) * glm::sin(az),
        glm::sin(el),
        glm::cos(el) * glm::cos(az)
    });
}

// Halton sequence value for a given index and base (typically 2 or 3).
// Returns a f32 in [0, 1).
f32 haltonJitter(u32 index, u32 base) {
    f32 result = 0.0f;
    f32 f = 1.0f;
    u32 i = index;
    while (i > 0) {
        f /= static_cast<f32>(base);
        result += f * static_cast<f32>(i % base);
        i /= base;
    }
    return result;
}

} // namespace

Renderer::Renderer(Window& window)
    : Renderer(window, MicropolyConfig{}) {}

Renderer::Renderer(Window& window, MicropolyConfig micropolyConfig)
    : m_window(window)
    , m_instance(std::make_unique<gfx::Instance>())
    , m_device(std::make_unique<gfx::Device>(*m_instance))
    , m_allocator(std::make_unique<gfx::Allocator>(*m_instance, *m_device))
    , m_swapchain(std::make_unique<gfx::Swapchain>(*m_instance, *m_device, *m_allocator, m_window))
    , m_frames(std::make_unique<gfx::FrameContextSet>(*m_device))
    , m_descriptorAllocator(std::make_unique<gfx::DescriptorAllocator>(*m_device))
    , m_gpuProfiler(std::make_unique<gfx::GpuProfiler>(*m_device))
    , m_renderGraph(std::make_unique<gfx::RenderGraph>())
    , m_shaderManager(std::make_unique<gfx::ShaderManager>(*m_device))
    , m_shaderHotReload(std::make_unique<gfx::ShaderHotReload>())
    , m_trianglePass(std::make_unique<TrianglePass>(*m_device, *m_allocator, *m_descriptorAllocator))
    , m_meshPass(std::make_unique<MeshPass>(*m_device))
    , m_micropolyConfig(std::move(micropolyConfig)) {

    // Micropoly scaffolding (M0a) — construct the pass shell after Device
    // so its constructor can probe capabilities and log the chosen vis
    // image format. MicropolyConfig::enabled defaults to false; when off,
    // MicropolyPass::record() is an early-return no-op and no GPU resources
    // are touched (Principle 1: bit-identical-when-disabled).
    m_micropolyPass = std::make_unique<MicropolyPass>(*m_device, m_micropolyConfig);

    // M2.4a: bring up the streaming orchestrator IFF enabled + capability-
    // gated on. When disabled we construct NOTHING here; no file IO, no VMA
    // allocations, no semaphores — the Renderer output is bit-identical to
    // the pre-micropoly path (Principle 1; screenshot_diff maxDelta=0).
    if (m_micropolyConfig.enabled) {
        const MicropolyCaps caps = micropolyCaps(*m_device);
        if (caps.row != HwMatrixRow::Disabled) {
            // M3.2 closeout fix #1: shared path-validation helper. Identical
            // rules are applied in AsyncIOWorker.cpp's worker thread as a
            // defence-in-depth check. See src/asset/MpPathUtils.h for the
            // canonical rule list (absolute; no UNC; no NT device namespace;
            // length bound). Previously this block inlined a lambda that
            // drifted against the worker's version — the helper eliminates
            // the duplication.
            std::string pathDetail;
            if (m_micropolyConfig.mpaFilePath.empty()) {
                ENIGMA_LOG_ERROR("[micropoly] streaming: enabled=true but mpaFilePath is empty; streaming disabled");
            } else if (!asset::isSafeMpaPath(m_micropolyConfig.mpaFilePath, pathDetail)) {
                ENIGMA_LOG_ERROR(
                    "[micropoly] streaming: mpaFilePath rejected: {}",
                    pathDetail);
            } else {
                auto reader = std::make_unique<asset::MpAssetReader>();
                auto openRes = reader->open(m_micropolyConfig.mpaFilePath);
                if (!openRes.has_value()) {
                    ENIGMA_LOG_ERROR("[micropoly] streaming: failed to open .mpa '{}'",
                                     m_micropolyConfig.mpaFilePath.string());
                } else {
                    renderer::micropoly::MicropolyStreamingOptions opts{};
                    opts.mpaFilePath             = m_micropolyConfig.mpaFilePath;
                    opts.reader                  = reader.get();
                    opts.residency.capacityBytes =
                        static_cast<u64>(m_micropolyConfig.pageCacheMB) * 1024ull * 1024ull;
                    opts.pageCache.poolBytes     =
                        static_cast<u64>(m_micropolyConfig.pageCacheMB) * 1024ull * 1024ull;
                    opts.pageCache.slotBytes     = 128u * 1024u;
                    opts.requestQueue.capacity   = 4096u;
                    opts.asyncIO.maxInflightRequests = 64u;
                    opts.debugName               = "micropoly.streaming";

                    auto made = renderer::micropoly::MicropolyStreaming::create(
                        *m_device, *m_allocator, std::move(opts));
                    if (!made) {
                        ENIGMA_LOG_ERROR(
                            "[micropoly] streaming: create failed: {} / {}",
                            renderer::micropoly::micropolyStreamingErrorKindString(made.error().kind),
                            made.error().detail);
                    } else {
                        m_micropolyStreaming = std::move(*made);
                        m_micropolyReader    = std::move(reader);
                        ENIGMA_LOG_INFO("[micropoly] streaming: initialised on '{}'",
                                        m_micropolyConfig.mpaFilePath.string());

                        // Register the RequestQueue buffer as a bindless
                        // UAV RWByteAddressBuffer. Without this the cull
                        // shader's emitPageReq() dereferences UINT32_MAX
                        // which silently no-ops on NVIDIA, so no page is
                        // ever streamed in and every cluster stays
                        // non-resident — streaming appears frozen.
                        auto& rq = m_micropolyStreaming->requestQueue();
                        const u32 rqSlot = m_descriptorAllocator->registerUavBuffer(
                            rq.buffer(), rq.bufferBytes());
                        if (rqSlot == UINT32_MAX) {
                            ENIGMA_LOG_ERROR(
                                "[micropoly] requestQueue bindless registration failed "
                                "— streaming will stall (emitPageReq no-ops)");
                        } else {
                            rq.setBindlessSlot(rqSlot);
                            ENIGMA_LOG_INFO(
                                "[micropoly] requestQueue bindless registered (slot={})",
                                rqSlot);
                        }

                        // Register PageCache VkBuffer as a bindless
                        // RWByteAddressBuffer. PageCache::create leaves
                        // bindlessBindingIndex at UINT32_MAX — without this
                        // step the HW raster task/mesh shaders read from
                        // g_rwBuffers[UINT32_MAX] and every Load returns
                        // zero, so cluster.vertexCount reads as 0 for every
                        // cluster and all debug overlays stay black.
                        auto& pc = m_micropolyStreaming->pageCache();
                        const VkDeviceSize pcBytes =
                            static_cast<VkDeviceSize>(pc.totalSlots()) *
                            static_cast<VkDeviceSize>(pc.slotBytes());
                        const u32 pcSlot = m_descriptorAllocator->registerUavBuffer(
                            pc.buffer(), pcBytes);
                        if (pcSlot == UINT32_MAX) {
                            ENIGMA_LOG_ERROR(
                                "[micropoly] pageCache bindless registration failed "
                                "— HW raster will read zeros and produce no output");
                        } else {
                            pc.setBindlessSlot(pcSlot);
                            ENIGMA_LOG_INFO(
                                "[micropoly] pageCache bindless registered (slot={}, bytes={})",
                                pcSlot,
                                static_cast<unsigned long long>(pcBytes));
                        }
                    }
                }
            }

            // M5a: per-asset-proxy BLAS for micropoly shadow casting.
            // Double-gated on Device::supportsRayTracing() AND the
            // micropoly reader having opened a valid .mpa above. On a
            // non-RT device we skip construction entirely so no RT
            // extension entrypoint is ever touched (Principle 1). On an
            // RT device we still guard on m_micropolyReader — without an
            // asset there's nothing to build.
            if (m_device->supportsRayTracing() && m_micropolyReader != nullptr) {
                auto blasMade = renderer::micropoly::MicropolyBlasManager::create(
                    *m_device, *m_allocator);
                if (!blasMade) {
                    ENIGMA_LOG_ERROR(
                        "[micropoly] BLAS manager: create failed: {} / {}",
                        renderer::micropoly::micropolyBlasErrorKindString(
                            blasMade.error().kind),
                        blasMade.error().detail);
                } else {
                    m_micropolyBlasManager = std::move(*blasMade);
                    auto built = m_micropolyBlasManager->buildForAsset(
                        *m_micropolyReader,
                        renderer::micropoly::kMpBlasDefaultDagLodLevel);
                    if (!built) {
                        // NoLevel3Clusters is soft — continue without
                        // micropoly shadows rather than fail the ctor.
                        ENIGMA_LOG_ERROR(
                            "[micropoly] BLAS manager: buildForAsset failed: "
                            "{} / {}",
                            renderer::micropoly::micropolyBlasErrorKindString(
                                built.error().kind),
                            built.error().detail);
                    } else {
                        ENIGMA_LOG_INFO(
                            "[micropoly] BLAS manager: built {} instance(s)",
                            m_micropolyBlasManager->instances().size());
                    }
                }
            }

            // M3.2: cluster cull compute pass. Constructed alongside the
            // streaming orchestrator so the two share the same enable
            // gating. Pipeline build failures log but do NOT abort the
            // Renderer — the pass stays null and dispatch() is skipped,
            // keeping the rest of the frame functional.
            auto cullMade = renderer::micropoly::MicropolyCullPass::create(
                *m_device, *m_allocator, *m_descriptorAllocator, *m_shaderManager);
            if (!cullMade) {
                ENIGMA_LOG_ERROR(
                    "[micropoly] cull pass: create failed: {} / {}",
                    renderer::micropoly::micropolyCullErrorKindString(cullMade.error().kind),
                    cullMade.error().detail);
            } else {
                m_micropolyCullPass = std::make_unique<renderer::micropoly::MicropolyCullPass>(
                    std::move(*cullMade));
                m_micropolyCullPass->registerHotReload(*m_shaderHotReload);
                ENIGMA_LOG_INFO("[micropoly] cull pass: initialised");
            }

            // M3.3: HW raster pass. Gated on mesh-shader + shaderImageInt64
            // device capabilities. Gate at the call site too so non-capable
            // devices don't log "raster pass: create failed" at every boot —
            // Principle 1 requires zero observable side effects when the
            // subsystem is inert.
            if (m_device->supportsMeshShaders() && m_device->supportsShaderImageInt64()) {
                auto rasterMade = renderer::micropoly::MicropolyRasterPass::create(
                    *m_device, *m_descriptorAllocator, *m_shaderManager);
                if (!rasterMade) {
                    ENIGMA_LOG_ERROR(
                        "[micropoly] raster pass: create failed: {} / {}",
                        renderer::micropoly::micropolyRasterErrorKindString(rasterMade.error().kind),
                        rasterMade.error().detail);
                } else {
                    m_micropolyRasterPass = std::make_unique<renderer::micropoly::MicropolyRasterPass>(
                        std::move(*rasterMade));
                    m_micropolyRasterPass->registerHotReload(*m_shaderHotReload);
                    ENIGMA_LOG_INFO("[micropoly] raster pass: initialised");
                }

                // M4.2: SW raster binning pass. Same capability umbrella as
                // the HW raster pass for now — see MicropolySwRasterPass.h
                // banner. Dispatched after the HW raster so M4.3's fragment
                // compute can consume the bins in the same frame.
                auto swRasterMade = renderer::micropoly::MicropolySwRasterPass::create(
                    *m_device, *m_allocator, *m_descriptorAllocator,
                    *m_shaderManager, m_swapchain->extent());
                if (!swRasterMade) {
                    ENIGMA_LOG_ERROR(
                        "[micropoly] sw raster pass: create failed: {} / {}",
                        renderer::micropoly::micropolySwRasterErrorKindString(swRasterMade.error().kind),
                        swRasterMade.error().detail);
                } else {
                    m_micropolySwRasterPass = std::move(*swRasterMade);
                    m_micropolySwRasterPass->registerHotReload(*m_shaderHotReload);
                    ENIGMA_LOG_INFO("[micropoly] sw raster pass: initialised (bin + fragment dispatch)");
                }
            }

            // M3.3: attach the pageId -> slotIndex GPU mirror that the
            // raster task shader indexes via pageId. Sized for the .mpa's
            // pageCount. Needs the reader to be live — skip cleanly when
            // either streaming init or reader open failed. Renderer owns
            // the bindless registration side so MicropolyStreaming stays
            // free of DescriptorAllocator coupling (simplifies the smaller
            // test binaries' link graphs).
            if (m_micropolyStreaming != nullptr && m_micropolyReader != nullptr) {
                const u32 pageCount = m_micropolyReader->header().pageCount;
                if (pageCount > 0u) {
                    const bool ok = m_micropolyStreaming->attachPageToSlotBuffer(pageCount);
                    if (!ok) {
                        ENIGMA_LOG_ERROR(
                            "[micropoly] pageToSlot buffer attach failed (pageCount={})",
                            pageCount);
                    } else {
                        const u32 slot = m_descriptorAllocator->registerStorageBuffer(
                            m_micropolyStreaming->pageToSlotBuffer(),
                            m_micropolyStreaming->pageToSlotBufferBytes());
                        m_micropolyStreaming->setPageToSlotBindless(slot);
                        ENIGMA_LOG_INFO(
                            "[micropoly] pageToSlot buffer attached (pageCount={}, bindless={})",
                            pageCount, slot);

                        // M4.5 multi-cluster page support: DEVICE_LOCAL SSBO
                        // holding firstDagNodeIdx per page. Shaders compute
                        // localClusterIdx = globalDagNodeIdx - firstDagIdx
                        // so every cluster in a multi-cluster page renders.
                        // One-shot staging upload at asset load time.
                        const auto firstDagIdx =
                            m_micropolyReader->firstDagNodeIndices();
                        const bool fdOk = m_micropolyStreaming
                            ->attachPageFirstDagNodeBuffer(
                                std::span<const u32>(firstDagIdx));
                        if (!fdOk) {
                            ENIGMA_LOG_ERROR(
                                "[micropoly] pageFirstDagNode buffer attach failed "
                                "(pageCount={})", pageCount);
                        } else {
                            const u32 fdSlot = m_descriptorAllocator->registerStorageBuffer(
                                m_micropolyStreaming->pageFirstDagNodeBuffer(),
                                m_micropolyStreaming->pageFirstDagNodeBufferBytes());
                            m_micropolyStreaming->setPageFirstDagNodeBindless(fdSlot);
                            ENIGMA_LOG_INFO(
                                "[micropoly] pageFirstDagNode buffer attached "
                                "(pageCount={}, bindless={})",
                                pageCount, fdSlot);
                        }

                        // M3.3-deferred DAG SSBO: assemble the 48 B runtime
                        // DAG node array (one entry per on-disk MpDagNode,
                        // joining cone data from per-page ClusterOnDisk with
                        // MpDagNode.pageId). This unblocks cull + raster +
                        // sw_raster + M6.1 heatmaps — the shader's
                        // `loadDagNode` reads 3×float4 per node with cone
                        // fields that do NOT exist in the 36 B on-disk
                        // MpDagNode (they live in ClusterOnDisk).
                        auto runtimeDag =
                            m_micropolyReader->assembleRuntimeDagNodes();
                        if (!runtimeDag.has_value()) {
                            ENIGMA_LOG_ERROR(
                                "[micropoly] assembleRuntimeDagNodes failed: {} / {}",
                                asset::mpReadErrorKindString(runtimeDag.error().kind),
                                runtimeDag.error().detail);
                        } else if (runtimeDag->empty()) {
                            ENIGMA_LOG_INFO(
                                "[micropoly] DAG node buffer skipped (empty DAG)");
                        } else {
                            const u8*  dagBytes = reinterpret_cast<const u8*>(
                                runtimeDag->data());
                            const u64  dagByteCount =
                                runtimeDag->size() * sizeof(decltype(*runtimeDag->data()));
                            const bool dagOk = m_micropolyStreaming
                                ->attachDagNodeBuffer(
                                    std::span<const u8>(dagBytes,
                                                        static_cast<std::size_t>(dagByteCount)));
                            if (!dagOk) {
                                ENIGMA_LOG_ERROR(
                                    "[micropoly] DAG node buffer attach failed "
                                    "(dagNodeCount={})", runtimeDag->size());
                            } else {
                                const u32 dagSlot = m_descriptorAllocator->registerStorageBuffer(
                                    m_micropolyStreaming->dagNodeBuffer(),
                                    m_micropolyStreaming->dagNodeBufferBytes());
                                m_micropolyStreaming->setDagNodeBufferBindless(dagSlot);
                                ENIGMA_LOG_INFO(
                                    "[micropoly] DAG node buffer attached "
                                    "(dagNodeCount={}, bindless={})",
                                    runtimeDag->size(), dagSlot);
                            }
                        }
                    }
                }
            }
        } else {
            ENIGMA_LOG_INFO("[micropoly] streaming: disabled (capability row=Disabled)");
        }
    }

    m_trianglePass->buildPipeline(*m_shaderManager,
                                  m_descriptorAllocator->layout(),
                                  m_swapchain->format(),
                                  m_swapchain->depthFormat());
    m_trianglePass->registerHotReload(*m_shaderHotReload);

    m_meshPass->buildPipeline(*m_shaderManager,
                              m_descriptorAllocator->layout(),
                              m_swapchain->format(),
                              m_swapchain->depthFormat());
    m_meshPass->registerHotReload(*m_shaderHotReload);

    // G-buffer images owned directly by Renderer.
    createGBufferImages(m_swapchain->extent());

    // Register G-buffer textures as bindless sampled images (read path:
    // LightingPass, RT passes, Denoiser, atmosphere, post-process).
    m_gbufAlbedoSlot     = m_descriptorAllocator->registerSampledImage(
        m_gbufAlbedo.view,     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_gbufNormalSlot     = m_descriptorAllocator->registerSampledImage(
        m_gbufNormal.view,     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_gbufMetalRoughSlot = m_descriptorAllocator->registerSampledImage(
        m_gbufMetalRough.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_gbufMotionVecSlot  = m_descriptorAllocator->registerSampledImage(
        m_gbufMotionVec.view,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_gbufDepthSlot      = m_descriptorAllocator->registerSampledImage(
        m_gbufDepth.view,      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Register G-buffer colour targets as bindless storage images (write path:
    // MaterialEvalPass compute shader writes PBR evaluation results via
    // imageStore()). Depth cannot be a storage image in Vulkan — no slot needed.
    m_gbufAlbedoStorageSlot     = m_descriptorAllocator->registerStorageImage(
        m_gbufAlbedo.view);
    m_gbufNormalStorageSlot     = m_descriptorAllocator->registerStorageImage(
        m_gbufNormal.view);
    m_gbufMetalRoughStorageSlot = m_descriptorAllocator->registerStorageImage(
        m_gbufMetalRough.view);
    m_gbufMotionVecStorageSlot  = m_descriptorAllocator->registerStorageImage(
        m_gbufMotionVec.view);

    // M3.1: 64-bit visibility image. createVisImage is a no-op when
    // MicropolyConfig::enabled=false (Principle 1 invariant) — the call is
    // safe to make unconditionally. When enabled, allocates the image at
    // swapchain extent and registers it as a bindless storage image.
    m_micropolyPass->createVisImage(
        m_swapchain->extent(), *m_allocator, *m_descriptorAllocator);

    // Nearest-neighbour sampler for G-buffer reads in the lighting pass.
    VkSamplerCreateInfo samplerCI{};
    samplerCI.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCI.magFilter    = VK_FILTER_NEAREST;
    samplerCI.minFilter    = VK_FILTER_NEAREST;
    samplerCI.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ENIGMA_VK_CHECK(vkCreateSampler(m_device->logical(), &samplerCI, nullptr, &m_gbufferSampler));
    m_gbufferSamplerSlot = m_descriptorAllocator->registerSampler(m_gbufferSampler);

    // Trilinear clamp sampler — used for the AP volume and atmosphere LUTs
    // so that the low-resolution (32³) froxel grid blends smoothly instead of
    // stepping at each froxel boundary and producing horizontal bands.
    VkSamplerCreateInfo linearCI{};
    linearCI.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    linearCI.magFilter    = VK_FILTER_LINEAR;
    linearCI.minFilter    = VK_FILTER_LINEAR;
    linearCI.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    linearCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    linearCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    linearCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ENIGMA_VK_CHECK(vkCreateSampler(m_device->logical(), &linearCI, nullptr, &m_linearSampler));
    m_linearSamplerSlot = m_descriptorAllocator->registerSampler(m_linearSampler);

    // Deferred lighting pass.
    m_lightingPass = std::make_unique<LightingPass>(*m_device);
    m_lightingPass->buildPipeline(*m_shaderManager,
                                   m_descriptorAllocator->layout(),
                                   VK_FORMAT_R16G16B16A16_SFLOAT);
    m_lightingPass->registerHotReload(*m_shaderHotReload);

    // Sky background pass — renders SkyView LUT into sky pixels after lighting.
    m_skyPass = std::make_unique<SkyBackgroundPass>(*m_device);
    m_skyPass->buildPipeline(*m_shaderManager,
                              m_descriptorAllocator->layout(),
                              VK_FORMAT_R16G16B16A16_SFLOAT);
    m_skyPass->registerHotReload(*m_shaderHotReload);

    // Atmosphere pass — must be initialized before PostProcessPass reads its AP layout.
    m_atmospherePass = std::make_unique<AtmospherePass>();
    {
        AtmospherePass::InitInfo atmInfo{};
        atmInfo.device              = m_device.get();
        atmInfo.allocator           = m_allocator.get();
        atmInfo.descriptorAllocator = m_descriptorAllocator.get();
        atmInfo.shaderManager       = m_shaderManager.get();
        atmInfo.globalSetLayout     = m_descriptorAllocator->layout();
        m_atmospherePass->init(atmInfo);
    }

    // Post-process pass — AP apply + bloom + tonemap → swapchain.
    // Uses the swapchain format since it writes the final display-ready image.
    m_postProcessPass = std::make_unique<PostProcessPass>(*m_device);
    m_postProcessPass->buildPipeline(*m_shaderManager,
                                      m_descriptorAllocator->layout(),
                                      m_atmospherePass->aerialPerspectiveReadSetLayout(),
                                      m_swapchain->format());
    m_postProcessPass->registerHotReload(*m_shaderHotReload);

    // SMAA anti-aliasing pass — allocates edge + weight textures, builds pipelines.
    m_smaaPass = std::make_unique<SMAAPass>(*m_device, *m_allocator);
    m_smaaPass->allocate(m_swapchain->extent(), *m_descriptorAllocator);
    m_smaaPass->buildPipelines(*m_shaderManager,
                                m_descriptorAllocator->layout(),
                                m_swapchain->format());
    m_smaaPass->registerHotReload(*m_shaderHotReload);

    // RT reflection pass.
    m_rtReflectionPass = std::make_unique<RTReflectionPass>(*m_device, *m_allocator);
    m_rtReflectionPass->allocate(m_swapchain->extent());
    m_rtReflectionPass->buildPipeline(*m_shaderManager,
                                       m_descriptorAllocator->layout(),
                                       RTReflectionPass::kOutputFormat);
    m_rtReflectionPass->registerHotReload(*m_shaderHotReload);

    // Register the reflection output as a bindless storage image.
    m_reflectionSlot = m_descriptorAllocator->registerStorageImage(m_rtReflectionPass->outputView());
    m_rtReflectionPass->outputSlot = m_reflectionSlot;

    // RT GI pass.
    m_giPass = std::make_unique<RTGIPass>(*m_device, *m_allocator);
    m_giPass->allocate(m_swapchain->extent());
    m_giPass->buildPipeline(*m_shaderManager, m_descriptorAllocator->layout());
    m_giPass->registerHotReload(*m_shaderHotReload);
    m_giSlot = m_descriptorAllocator->registerStorageImage(m_giPass->outputView());
    m_giPass->outputSlot = m_giSlot;

    // RT Shadow pass.
    m_shadowPass = std::make_unique<RTShadowPass>(*m_device, *m_allocator);
    m_shadowPass->allocate(m_swapchain->extent());
    m_shadowPass->buildPipeline(*m_shaderManager, m_descriptorAllocator->layout());
    m_shadowPass->registerHotReload(*m_shaderHotReload);
    m_shadowSlot = m_descriptorAllocator->registerStorageImage(m_shadowPass->outputView());
    m_shadowPass->outputSlot = m_shadowSlot;

    // Wet Road pass.
    m_wetRoadPass = std::make_unique<WetRoadPass>(*m_device, *m_allocator);
    m_wetRoadPass->allocate(m_swapchain->extent());
    m_wetRoadPass->buildPipeline(*m_shaderManager, m_descriptorAllocator->layout());
    m_wetRoadPass->registerHotReload(*m_shaderHotReload);
    m_wetRoadSlot = m_descriptorAllocator->registerStorageImage(m_wetRoadPass->outputView());
    m_wetRoadPass->outputSlot = m_wetRoadSlot;

    // Denoisers (one per RT effect).
    m_giDenoiser = std::make_unique<Denoiser>(*m_device, *m_allocator);
    m_giDenoiser->allocate(m_swapchain->extent(), RTGIPass::kOutputFormat);
    m_giDenoiser->buildPipelines(*m_shaderManager, m_descriptorAllocator->layout(), RTGIPass::kOutputFormat);
    m_giDenoiser->registerHotReload(*m_shaderHotReload);
    m_giDenoiseSlot = m_descriptorAllocator->registerStorageImage(m_giDenoiser->outputView());
    m_giDenoiser->outputSlot = m_giDenoiseSlot;

    m_shadowDenoiser = std::make_unique<Denoiser>(*m_device, *m_allocator);
    m_shadowDenoiser->allocate(m_swapchain->extent(), RTShadowPass::kOutputFormat);
    m_shadowDenoiser->buildPipelines(*m_shaderManager, m_descriptorAllocator->layout(), RTShadowPass::kOutputFormat);
    m_shadowDenoiser->registerHotReload(*m_shaderHotReload);
    m_shadowDenoiseSlot = m_descriptorAllocator->registerStorageImage(m_shadowDenoiser->outputView());
    m_shadowDenoiser->outputSlot = m_shadowDenoiseSlot;

    m_reflectionDenoiser = std::make_unique<Denoiser>(*m_device, *m_allocator);
    m_reflectionDenoiser->allocate(m_swapchain->extent(), RTReflectionPass::kOutputFormat);
    m_reflectionDenoiser->buildPipelines(*m_shaderManager, m_descriptorAllocator->layout(), RTReflectionPass::kOutputFormat);
    m_reflectionDenoiser->registerHotReload(*m_shaderHotReload);
    m_reflDenoiseSlot = m_descriptorAllocator->registerStorageImage(m_reflectionDenoiser->outputView());
    m_reflectionDenoiser->outputSlot = m_reflDenoiseSlot;

    // Visibility buffer pipeline — GPU-driven mesh-shader path.
    m_gpuScene       = std::make_unique<GpuSceneBuffer>(*m_device, *m_allocator, *m_descriptorAllocator);
    m_gpuMeshlets    = std::make_unique<GpuMeshletBuffer>(*m_device, *m_allocator, *m_descriptorAllocator);
    m_indirectBuffer = std::make_unique<IndirectDrawBuffer>(*m_device, *m_allocator, *m_descriptorAllocator);
    m_indirectBuffer->resize(65536); // reserve for up to 64K meshlets
    m_terrainWireIndirectBuffer = std::make_unique<IndirectDrawBuffer>(*m_device, *m_allocator, *m_descriptorAllocator);
    m_terrainWireIndirectBuffer->resize(65536);

    m_hizPass = std::make_unique<HiZPass>(*m_device, *m_allocator, *m_descriptorAllocator);
    m_hizPass->allocate(m_swapchain->extent());
    m_hizPass->buildPipeline(*m_shaderManager, m_descriptorAllocator->layout());
    m_hizPass->registerHotReload(*m_shaderHotReload);

    m_gpuCullPass = std::make_unique<GpuCullPass>(*m_device);
    m_gpuCullPass->buildPipeline(*m_shaderManager, m_descriptorAllocator->layout());
    m_gpuCullPass->registerHotReload(*m_shaderHotReload);

    m_visibilityPass = std::make_unique<VisibilityBufferPass>(*m_device, *m_allocator, *m_descriptorAllocator);
    m_visibilityPass->allocate(m_swapchain->extent(), gbuf::kDepthFormat);
    m_visibilityPass->buildPipeline(*m_shaderManager, m_descriptorAllocator->layout());
    m_visibilityPass->buildTerrainPipeline(*m_shaderManager, m_descriptorAllocator->layout());
    m_visibilityPass->registerHotReload(*m_shaderHotReload);

    // Wireframe pipeline (conditional on fillModeNonSolid support).
    if (m_device->fillModeNonSolidSupported() && m_visibilityPass) {
        m_visibilityPass->buildWireframePipeline(*m_shaderManager,
                                                  m_descriptorAllocator->layout(),
                                                  m_swapchain->format());
        m_visibilityPass->buildTerrainWireframePipeline(*m_shaderManager,
                                                         m_descriptorAllocator->layout(),
                                                         m_swapchain->format());
    }

    // Debug visualization pass — fullscreen debug modes.
    m_debugVisPass = std::make_unique<DebugVisualizationPass>(*m_device);
    m_debugVisPass->buildPipelines(*m_shaderManager,
                                    m_descriptorAllocator->layout(),
                                    m_swapchain->format());
    m_debugVisPass->registerHotReload(*m_shaderHotReload);

    m_materialEvalPass = std::make_unique<MaterialEvalPass>(*m_device);
    m_materialEvalPass->buildPipeline(*m_shaderManager, m_descriptorAllocator->layout());
    m_materialEvalPass->registerHotReload(*m_shaderHotReload);
    m_materialEvalPass->prepare(
        m_swapchain->extent(),
        m_gbufAlbedo.image,     m_gbufNormal.image,
        m_gbufMetalRough.image, m_gbufMotionVec.image,
        m_gbufDepth.image,
        m_gbufAlbedoStorageSlot,     m_gbufNormalStorageSlot,
        m_gbufMetalRoughStorageSlot, m_gbufMotionVecStorageSlot,
        m_gbufDepthSlot);

    // Clustered forward pass — renders transparent geometry after opaque lighting.
    m_clusteredForwardPass = std::make_unique<ClusteredForwardPass>(*m_device);
    m_clusteredForwardPass->buildPipeline(*m_shaderManager,
                                          m_descriptorAllocator->layout(),
                                          VK_FORMAT_R16G16B16A16_SFLOAT);
    m_clusteredForwardPass->registerHotReload(*m_shaderHotReload);

    // Create per-frame camera SSBOs (host-visible, persistently mapped).
    // 13 float4s = 208 bytes per camera buffer.
    constexpr VkDeviceSize kCameraBufferSize = sizeof(GpuCameraData);
    for (u32 i = 0; i < gfx::MAX_FRAMES_IN_FLIGHT; ++i) {
        VkBufferCreateInfo bufInfo{};
        bufInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size        = kCameraBufferSize;
        bufInfo.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                        | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo allocResult{};
        ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &bufInfo, &allocInfo,
                                        &m_cameraBuffers[i].buffer,
                                        &m_cameraBuffers[i].allocation,
                                        &allocResult));
        m_cameraBuffers[i].mapped = allocResult.pMappedData;
        ENIGMA_ASSERT(m_cameraBuffers[i].mapped != nullptr);

        m_cameraBuffers[i].bindlessSlot =
            m_descriptorAllocator->registerStorageBuffer(
                m_cameraBuffers[i].buffer, kCameraBufferSize);
    }

    vkGetPhysicalDeviceMemoryProperties(m_device->physical(), &m_memoryProperties);

    ENIGMA_LOG_INFO("[renderer] constructed (camera slots: {}, {})",
                    m_cameraBuffers[0].bindlessSlot,
                    m_cameraBuffers[1].bindlessSlot);

    // Upscaler — auto-select based on vendor/tier, init at swapchain display extent.
    m_upscaler = UpscalerFactory::create(m_device->properties(), m_device->gpuTier());
    m_upscaler->init(m_device->logical(), m_device->physical(),
                     m_swapchain->extent(), m_upscalerSettings.quality);

    // ImGui overlay — must be created after swapchain (needs format + imageCount).
    m_imguiLayer = std::make_unique<gfx::ImGuiLayer>(
        *m_device,
        m_instance->handle(),
        m_window.handle(),
        m_swapchain->format(),
        m_swapchain->imageCount());

    // Physics debug wireframe renderer.
    PhysicsDebugInitInfo dbgInfo{};
    dbgInfo.device              = m_device.get();
    dbgInfo.allocator           = m_allocator.get();
    dbgInfo.descriptorAllocator = m_descriptorAllocator.get();
    dbgInfo.shaderManager       = m_shaderManager.get();
    dbgInfo.globalSetLayout     = m_descriptorAllocator->layout();
    dbgInfo.colorFormat         = VK_FORMAT_R16G16B16A16_SFLOAT;
    dbgInfo.depthFormat         = gbuf::kDepthFormat;
    m_physicsDebugRenderer.init(dbgInfo);

    createHdrColor(m_swapchain->extent());
    createSmaaLdr(m_swapchain->extent());

    // Bake Transmittance + MultiScatter LUTs on startup (one-shot command buffer).
    {
        VkDevice dev = m_device->logical();

        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandPoolCreateInfo poolCI{};
        poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolCI.queueFamilyIndex = m_device->graphicsQueueFamily();
        ENIGMA_VK_CHECK(vkCreateCommandPool(dev, &poolCI, nullptr, &pool));

        VkCommandBuffer bakeCmd = VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo cbInfo{};
        cbInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbInfo.commandPool        = pool;
        cbInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbInfo.commandBufferCount = 1;
        ENIGMA_VK_CHECK(vkAllocateCommandBuffers(dev, &cbInfo, &bakeCmd));

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        ENIGMA_VK_CHECK(vkBeginCommandBuffer(bakeCmd, &beginInfo));

        m_atmospherePass->bakeStaticLUTs(bakeCmd, m_atmosphereSettings,
                                          m_sunWorldDir, m_gbufferSamplerSlot);

        ENIGMA_VK_CHECK(vkEndCommandBuffer(bakeCmd));

        VkFence fence = VK_NULL_HANDLE;
        VkFenceCreateInfo fenceCI{};
        fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        ENIGMA_VK_CHECK(vkCreateFence(dev, &fenceCI, nullptr, &fence));

        VkSubmitInfo submitInfo{};
        submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers    = &bakeCmd;
        ENIGMA_VK_CHECK(vkQueueSubmit(m_device->graphicsQueue(), 1, &submitInfo, fence));
        ENIGMA_VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));

        vkDestroyFence(dev, fence, nullptr);
        vkDestroyCommandPool(dev, pool, nullptr); // frees bakeCmd implicitly

        m_sunDirty = false;
        ENIGMA_LOG_INFO("[renderer] initial atmosphere LUT bake complete");
    }
}

Renderer::~Renderer() {
    if (m_device) {
        vkDeviceWaitIdle(m_device->logical());
    }

    // CDLOD terrain + heightmap own VMA buffers/images and bindless slots —
    // destroy before VMA / DescriptorAllocator teardown.
    m_terrain.reset();
    m_heightmapLoader.reset();

    // Shut down ImGui before destroying Vulkan resources.
    m_imguiLayer.reset();

    // Shut down the upscaler before device destruction.
    if (m_upscaler) {
        m_upscaler->shutdown();
    }

    // Clean up samplers (pass objects destroy their own images/pipelines).
    if (m_gbufferSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device->logical(), m_gbufferSampler, nullptr);
    }
    if (m_linearSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device->logical(), m_linearSampler, nullptr);
    }
    // Renderer-owned runtime-rebuilt material sampler. The original sampler
    // created by GltfLoader still lives in scene.ownedSamplers and is freed
    // by Scene::destroy — we only own the replacement created by
    // applyTextureFilterSettings().
    if (m_materialSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device->logical(), m_materialSampler, nullptr);
        m_materialSampler = VK_NULL_HANDLE;
    }

    // Atmosphere pass owns LUT images + pipelines.
    if (m_atmospherePass) {
        m_atmospherePass->shutdown();
    }

    // SMAA intermediate textures — free before SMAAPass is destroyed.
    if (m_smaaPass && m_smaaPass->isAllocated()) {
        m_smaaPass->free(*m_descriptorAllocator);
    }
    destroySmaaLdr();

    // HDR intermediate — destroyed before VMA teardown.
    destroyHdrColor();
    destroyGBufferImages();

    // M5a: release per-asset-proxy BLAS manager before MicropolyStreaming
    // destructs. The manager does CPU-side decompression at build time so
    // it doesn't reference PageCache pages owned by MicropolyStreaming,
    // but we tear down explicitly here to keep the destruction order
    // readable: BLAS manager -> streaming -> vis image -> VMA / Device.
    // Safe to reset when null (non-RT devices never constructed it).
    m_micropolyBlasManager.reset();

    // M3.3: release the pageId->slot bindless slot before we let
    // MicropolyStreaming destruct (its destructor frees the VkBuffer, but
    // the bindless slot is Renderer-owned so we release it here).
    if (m_micropolyStreaming != nullptr
        && m_micropolyStreaming->pageToSlotBufferBindless() != UINT32_MAX) {
        m_descriptorAllocator->releaseStorageBuffer(
            m_micropolyStreaming->pageToSlotBufferBindless());
        m_micropolyStreaming->setPageToSlotBindless(UINT32_MAX);
    }

    // Release the RequestQueue bindless slot. Registered as a UAV byte
    // buffer at streaming init (see create block in ctor); the VkBuffer
    // itself is owned by RequestQueue inside MicropolyStreaming and is
    // released in its destructor below.
    if (m_micropolyStreaming != nullptr
        && m_micropolyStreaming->requestQueue().bindlessSlot() != UINT32_MAX) {
        m_descriptorAllocator->releaseUavBuffer(
            m_micropolyStreaming->requestQueue().bindlessSlot());
        m_micropolyStreaming->requestQueue().setBindlessSlot(UINT32_MAX);
    }

    // M4.5: release the pageFirstDagNode bindless slot (same ownership
    // model as pageToSlot — DescriptorAllocator owned by Renderer,
    // VkBuffer owned by MicropolyStreaming).
    if (m_micropolyStreaming != nullptr
        && m_micropolyStreaming->pageFirstDagNodeBufferBindless() != UINT32_MAX) {
        m_descriptorAllocator->releaseStorageBuffer(
            m_micropolyStreaming->pageFirstDagNodeBufferBindless());
        m_micropolyStreaming->setPageFirstDagNodeBindless(UINT32_MAX);
    }

    // M3.3-deferred: release the DAG node bindless slot. Same ownership
    // model — DescriptorAllocator owned by Renderer, VkBuffer owned by
    // MicropolyStreaming.
    if (m_micropolyStreaming != nullptr
        && m_micropolyStreaming->dagNodeBufferBindless() != UINT32_MAX) {
        m_descriptorAllocator->releaseStorageBuffer(
            m_micropolyStreaming->dagNodeBufferBindless());
        m_micropolyStreaming->setDagNodeBufferBindless(UINT32_MAX);
    }

    // M3.1: Micropoly 64-bit vis image. destroyVisImage is a no-op when the
    // pass never allocated one (disabled configs), so this call is safe
    // unconditionally. Must run before DescriptorAllocator / VMA teardown
    // since it releases a bindless slot + VMA allocation.
    if (m_micropolyPass) {
        m_micropolyPass->destroyVisImage(*m_allocator, *m_descriptorAllocator);
    }

    // Physics debug renderer owns a VMA buffer — must be destroyed before VMA teardown.
    m_physicsDebugRenderer.destroy();

    // Clean up camera buffers.
    for (auto& cb : m_cameraBuffers) {
        if (cb.buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(), cb.buffer, cb.allocation);
        }
    }

    const u32 validationCount = gfx::getValidationCounter();
    ENIGMA_LOG_INFO("[renderer] shutdown g_validationCounter = {}", validationCount);
    ENIGMA_ASSERT(validationCount == 0);

    ENIGMA_LOG_INFO("[renderer] shutdown");
}

void Renderer::uploadCameraData() {
    const auto& cb = m_cameraBuffers[m_frameIndex];
    const auto extent = m_swapchain->extent();
    const f32 aspect = (extent.height > 0)
        ? static_cast<f32>(extent.width) / static_cast<f32>(extent.height)
        : 1.0f;

    GpuCameraData data{};
    if (m_camera != nullptr) {
        data = m_camera->gpuData(aspect);
        // Cache camera basis vectors and FOV tangents for the AP bake shader.
        m_cameraRight   = m_camera->right();
        m_cameraUp      = m_camera->up();
        m_cameraForward = m_camera->forward();
        m_tanHalfFovY   = std::tan(m_camera->fovY * 0.5f);
        m_tanHalfFovX   = m_tanHalfFovY * aspect;
    } else {
        // Identity camera fallback.
        data.view        = mat4{1.0f};
        data.proj        = mat4{1.0f};
        data.viewProj    = mat4{1.0f};
        data.invViewProj = mat4{1.0f};
        data.worldPos    = vec4{0.0f, 0.0f, 0.0f, 1.0f};
        // Basis vectors stay at their initialized defaults.
    }

    // Inject previous frame's viewProj for motion vector computation.
    data.prevViewProj = m_prevViewProj;
    std::memcpy(cb.mapped, &data, sizeof(data));
    vmaFlushAllocation(m_allocator->handle(), cb.allocation, 0, sizeof(data));

    // Save current viewProj so next frame can use it as prevViewProj.
    m_prevViewProj    = data.viewProj;
    m_invViewProj     = data.invViewProj;
    m_cameraWorldPos  = vec3(data.worldPos);
}

void Renderer::drawFrame() {
    {
        const auto fb = m_window.framebufferSize();
        if (fb.width == 0 || fb.height == 0) {
            m_window.waitEvents();
            return;
        }
    }

    // CPU frame time.
    {
        const auto now = std::chrono::steady_clock::now();
        if (m_lastFrameTime.time_since_epoch().count() != 0) {
            m_cpuFrameTimeMs = std::chrono::duration<f32, std::milli>(now - m_lastFrameTime).count();
        }
        m_lastFrameTime = now;
    }

    m_shaderHotReload->poll();

    // Compute canonical sun direction once per frame from UI az/el.
    // This is the single source of truth fanned to all pass push constants.
    {
        const vec3 newDir = fromAzimuthElevation(
            m_atmosphereSettings.sunAzimuth,
            m_atmosphereSettings.sunElevation);
        if (glm::any(glm::notEqual(newDir, m_sunWorldDir))) {
            m_sunWorldDir = newDir;
            m_sunDirty    = true;
        }
    }

    // When the sun direction changes, rebake the static LUTs (Transmittance +
    // MultiScatter). These are slow to compute and only change when the sun
    // moves, so we idle the device and rebuild before opening this frame's
    // command buffer. SkyView + AP are rebuilt every frame in updatePerFrame.
    if (m_sunDirty && m_atmospherePass) {
        vkDeviceWaitIdle(m_device->logical());

        VkCommandPool bakePool = VK_NULL_HANDLE;
        VkCommandPoolCreateInfo bakePoolCI{};
        bakePoolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        bakePoolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        bakePoolCI.queueFamilyIndex = m_device->graphicsQueueFamily();
        ENIGMA_VK_CHECK(vkCreateCommandPool(m_device->logical(), &bakePoolCI, nullptr, &bakePool));

        VkCommandBuffer bakeCmd = VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo bakeCbInfo{};
        bakeCbInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        bakeCbInfo.commandPool        = bakePool;
        bakeCbInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        bakeCbInfo.commandBufferCount = 1;
        ENIGMA_VK_CHECK(vkAllocateCommandBuffers(m_device->logical(), &bakeCbInfo, &bakeCmd));

        VkCommandBufferBeginInfo bakeBeginInfo{};
        bakeBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bakeBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        ENIGMA_VK_CHECK(vkBeginCommandBuffer(bakeCmd, &bakeBeginInfo));

        m_atmospherePass->bakeStaticLUTs(bakeCmd, m_atmosphereSettings,
                                          m_sunWorldDir, m_gbufferSamplerSlot);

        ENIGMA_VK_CHECK(vkEndCommandBuffer(bakeCmd));

        VkFence bakeFence = VK_NULL_HANDLE;
        VkFenceCreateInfo bakeFenceCI{};
        bakeFenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        ENIGMA_VK_CHECK(vkCreateFence(m_device->logical(), &bakeFenceCI, nullptr, &bakeFence));

        VkSubmitInfo bakeSubmit{};
        bakeSubmit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        bakeSubmit.commandBufferCount = 1;
        bakeSubmit.pCommandBuffers    = &bakeCmd;
        ENIGMA_VK_CHECK(vkQueueSubmit(m_device->graphicsQueue(), 1, &bakeSubmit, bakeFence));
        ENIGMA_VK_CHECK(vkWaitForFences(m_device->logical(), 1, &bakeFence, VK_TRUE, UINT64_MAX));

        vkDestroyFence(m_device->logical(), bakeFence, nullptr);
        vkDestroyCommandPool(m_device->logical(), bakePool, nullptr); // frees bakeCmd

        m_sunDirty = false;
    }

    VkDevice dev = m_device->logical();
    gfx::FrameContext& frame = m_frames->get(m_frameIndex);

    if (frame.frameValue > 0) {
        VkSemaphoreWaitInfo waitInfo{};
        waitInfo.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores    = &frame.inFlight;
        waitInfo.pValues        = &frame.frameValue;
        ENIGMA_VK_CHECK(vkWaitSemaphores(dev, &waitInfo, UINT64_MAX));
    }

    u32 imageIndex = 0;
    {
        const VkResult acquireResult = vkAcquireNextImageKHR(
            dev, m_swapchain->handle(), UINT64_MAX,
            frame.imageAvailable, VK_NULL_HANDLE, &imageIndex);
        if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
            const auto [w, h] = m_window.framebufferSize();
            m_swapchain->recreate(w, h);
            resizeGBuffer(m_swapchain->extent());
            return;
        }
        if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
            ENIGMA_VK_CHECK(acquireResult);
        }
    }

    // Upload camera data for this frame.
    uploadCameraData();

    // Process pending deformation: upload deformed vertices and update BLASes.
    if (m_deformationPending && m_scene != nullptr) {
        constexpr VkDeviceSize kVertexStride = 3 * sizeof(vec4); // 48 bytes

        for (u32 i = 0; i < static_cast<u32>(m_scene->primitives.size()); ++i) {
            auto& prim = m_scene->primitives[i];
            if (!prim.blas.has_value()) continue;

            // uploadDeformedPositions no-ops for unregistered primitives.
            m_deformationSystem.uploadDeformedPositions(i, prim.vertexBuffer, *m_device, *m_allocator);

            if (m_deformationSystem.requiresBlasRebuild(i)) {
                prim.blas->rebuild(*m_device, *m_allocator,
                                   prim.vertexBuffer, prim.vertexCount, kVertexStride,
                                   prim.indexBuffer, prim.indexCount);
            } else {
                prim.blas->refit(*m_device, *m_allocator,
                                 prim.vertexBuffer, prim.vertexCount, kVertexStride,
                                 prim.indexBuffer, prim.indexCount);
            }
        }

        m_deformationPending = false;
    }

    ENIGMA_VK_CHECK(vkResetCommandPool(dev, frame.commandPool, 0));

    // Read back GPU timings from the previous frame before resetting the pool.
    // (The previous frame's submissions are guaranteed done by the timeline wait above.)
    if (frame.frameValue > 0) {
        m_lastGpuTimings = m_gpuProfiler->readback();
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ENIGMA_VK_CHECK(vkBeginCommandBuffer(frame.commandBuffer, &beginInfo));

    m_gpuProfiler->reset(frame.commandBuffer);

    // Per-frame atmosphere LUT update: SkyView + Aerial Perspective volume.
    // Recorded directly into the main command buffer before the render graph
    // executes so RT passes, sky background, lighting, and post-process all
    // read fresh LUTs this frame. Camera position converted from world-units
    // to km (engine uses metres: 1 world unit = 1 m → / 1000).
    if (m_atmospherePass) {
        m_atmospherePass->updatePerFrame(
            frame.commandBuffer,
            m_atmosphereSettings,
            m_sunWorldDir,
            m_cameraWorldPos * 0.001f, // metres → km
            m_cameraRight,
            m_cameraUp,
            m_cameraForward,
            m_tanHalfFovX,
            m_tanHalfFovY,
            m_gbufferSamplerSlot);
    }

    // Debug mode hotkeys (F1-F6).
    // ImGui::IsKeyPressed works even outside ImGui window scope.
    if (ImGui::IsKeyPressed(ImGuiKey_F1)) m_debugMode = DebugMode::Lit;
    if (ImGui::IsKeyPressed(ImGuiKey_F2)) m_debugMode = DebugMode::Unlit;
    if (ImGui::IsKeyPressed(ImGuiKey_F3) &&
        m_device->fillModeNonSolidSupported() &&
        m_visibilityPass && m_visibilityPass->hasWireframePipeline())
        m_debugMode = DebugMode::Wireframe;
    if (ImGui::IsKeyPressed(ImGuiKey_F4) &&
        m_device->fillModeNonSolidSupported() &&
        m_visibilityPass && m_visibilityPass->hasWireframePipeline())
        m_debugMode = DebugMode::LitWireframe;
    if (ImGui::IsKeyPressed(ImGuiKey_F6)) m_debugMode = DebugMode::DetailLighting;
    if (ImGui::IsKeyPressed(ImGuiKey_F7))
        m_debugMode = DebugMode::Clusters;
    // M4.6: MicropolyRasterClass — availability gated on Int64 storage-image
    // support + an active HW raster pass (both SW and HW write to the same
    // vis image, so one capability-gate covers both).
    if (ImGui::IsKeyPressed(ImGuiKey_F8) &&
        m_device->supportsShaderImageInt64() &&
        m_micropolyRasterPass != nullptr)
        m_debugMode = DebugMode::MicropolyRasterClass;
    // M6.1: LOD + Residency heatmaps — same capability gate as RasterClass.
    if (ImGui::IsKeyPressed(ImGuiKey_F9) &&
        m_device->supportsShaderImageInt64() &&
        m_micropolyRasterPass != nullptr)
        m_debugMode = DebugMode::MicropolyLodHeatmap;
    if (ImGui::IsKeyPressed(ImGuiKey_F10) &&
        m_device->supportsShaderImageInt64() &&
        m_micropolyRasterPass != nullptr)
        m_debugMode = DebugMode::MicropolyResidencyHeatmap;

    // ImGui new frame -- must be called before any ImGui:: window calls this frame.
    if (m_imguiLayer) {
        m_imguiLayer->newFrame();

        // Query device-local VRAM budget for the perf panel.
        f32 vramUsedMb = 0.f, vramBudgetMb = 0.f;
        {
            std::vector<VmaBudget> budgets(m_memoryProperties.memoryHeapCount);
            vmaGetHeapBudgets(m_allocator->handle(), budgets.data());
            for (u32 i = 0; i < m_memoryProperties.memoryHeapCount; ++i) {
                if (m_memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                    vramUsedMb   += static_cast<f32>(budgets[i].usage)  / (1024.f * 1024.f);
                    vramBudgetMb += static_cast<f32>(budgets[i].budget) / (1024.f * 1024.f);
                }
            }
        }

        m_imguiLayer->drawGpuTimings(m_lastGpuTimings, m_cpuFrameTimeMs, vramUsedMb, vramBudgetMb);
        m_imguiLayer->drawSceneInfo(
            m_scene ? static_cast<u32>(m_scene->primitives.size()) : 0u,
            m_tlasSlot > 0 ? 1u : 0u);
        m_imguiLayer->drawUpscalerSettings(m_upscalerSettings);
        m_imguiLayer->drawPhysicsStats(0.0f, 0u); // wired up later when physics is exposed
        m_imguiLayer->drawPhysicsDebugPanel(m_physicsDebugRenderer.enabled,
                                             m_physicsDebugRenderer.depthTestEnabled);

        // Settings panel — sun light + scene knobs.
        ImGui::SetNextWindowPos({310.f, 10.f}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({300.f, 260.f}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Settings")) {
            if (ImGui::CollapsingHeader("Atmosphere", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::SliderFloat("Sun Azimuth",   &m_atmosphereSettings.sunAzimuth,   0.f,   360.f, "%.1f deg");
                ImGui::SliderFloat("Sun Elevation", &m_atmosphereSettings.sunElevation, -10.f,  90.f, "%.1f deg");
                ImGui::SliderFloat("Sun Intensity", &m_atmosphereSettings.sunIntensity,  0.f,   20.f);
                ImGui::Separator();
                ImGui::SliderFloat("Exposure EV",        &m_atmosphereSettings.exposureEV,      -6.f, 6.f, "%.2f EV");
                ImGui::SliderFloat("Bloom Threshold",    &m_atmosphereSettings.bloomThreshold,   0.f, 4.f);
                ImGui::SliderFloat("Bloom Intensity",    &m_atmosphereSettings.bloomIntensity,   0.f, 2.f);
                ImGui::Checkbox("Bloom",              &m_atmosphereSettings.bloomEnabled);
                ImGui::SameLine();
                ImGui::Checkbox("Aerial Perspective", &m_atmosphereSettings.aerialPerspectiveEnabled);
                if (m_atmosphereSettings.aerialPerspectiveEnabled)
                    ImGui::SliderFloat("AP Strength", &m_atmosphereSettings.aerialPerspectiveStrength, 0.0f, 1.0f);
                const char* tonemapItems[] = {"AgX", "ACES"};
                ImGui::Combo("Tonemap", &m_atmosphereSettings.tonemapMode, tonemapItems, 2);
            }
            if (ImGui::CollapsingHeader("Sun Light")) {
                ImGui::ColorEdit3 ("Color",    &m_light.color.x);
            }
            if (ImGui::CollapsingHeader("Environment")) {
                ImGui::SliderFloat("Wetness", &m_wetnessFactor, 0.f, 1.f);
            }
            if (ImGui::CollapsingHeader("Renderer")) {
                ImGui::Checkbox("SMAA", &m_aaSettings.smaaEnabled);

                ImGui::SeparatorText("Texture Filtering");
                bool tfDirty = false;
                tfDirty |= ImGui::Checkbox("Mipmaps", &m_textureFilterSettings.mipmapsEnabled);
                static const char* kAnisoLabels[] = { "1x (off)", "2x", "4x", "8x", "16x" };
                static const u32   kAnisoValues[] = { 1u, 2u, 4u, 8u, 16u };
                int anisoIdx = 0;
                for (int i = 0; i < 5; ++i) {
                    if (kAnisoValues[i] == m_textureFilterSettings.anisotropy) { anisoIdx = i; break; }
                }
                const f32 devMaxAniso = m_device->properties().limits.maxSamplerAnisotropy;
                const bool anisoSupported = devMaxAniso > 1.0f;
                if (!anisoSupported) ImGui::BeginDisabled();
                if (ImGui::Combo("Anisotropic Filtering", &anisoIdx, kAnisoLabels, 5)) {
                    m_textureFilterSettings.anisotropy = kAnisoValues[anisoIdx];
                    tfDirty = true;
                }
                if (!anisoSupported) ImGui::EndDisabled();
                if (tfDirty) applyTextureFilterSettings();
                ImGui::Separator();
            }
            if (ImGui::CollapsingHeader("Graphics Quality")) {
                const bool rtAvailable = m_device->gpuTier() >= gfx::GpuTier::Recommended;
                if (!rtAvailable) {
                    ImGui::TextDisabled("RT effects require Recommended tier GPU");
                    ImGui::BeginDisabled();
                }
                ImGui::Checkbox("RT Reflections", &m_rtReflectionsEnabled);
                ImGui::Checkbox("RT Global Illumination", &m_rtGIEnabled);
                ImGui::Checkbox("RT Shadows", &m_rtShadowsEnabled);
                ImGui::Checkbox("Wet Road Effect", &m_wetRoadEnabled);
                ImGui::Checkbox("Denoising", &m_denoiseEnabled);
                if (!rtAvailable) {
                    ImGui::EndDisabled();
                }
            }
        }
        ImGui::End();

        // Debug Views panel.
        const bool wireAvail  = m_device->fillModeNonSolidSupported() &&
                                m_visibilityPass && m_visibilityPass->hasWireframePipeline();
        const bool clustAvail = true;
        // M4.6 / M6.1: MicropolyRasterClass, LodHeatmap, ResidencyHeatmap
        // all require Int64 storage images AND an active HW raster pass
        // (so the vis image has meaningful writes). Share one availability
        // flag across all three modes.
        const bool mpClassAvail = m_device->supportsShaderImageInt64()
                               && m_micropolyRasterPass != nullptr;
        // M6.2b: bounds overlay iterates the DAG and projects bounding
        // spheres — it does NOT read the 64-bit vis image, so no Int64
        // storage-image requirement. All it needs is a wired DAG SSBO.
        const bool mpBoundsAvail = m_micropolyStreaming
                               && m_micropolyStreaming->dagNodeBufferBindless() != UINT32_MAX
                               && m_micropolyReader
                               && m_micropolyReader->header().dagNodeCount > 0u;
        // M6 plan §3.M6: BinOverflow heat reads the SW raster tile-bin SSBOs.
        // They only exist when the SW raster pass is constructed (which itself
        // requires mesh-shaders + shaderImageInt64 + micropoly enabled). No
        // vis-image dependency — this overlay is a read-side consumer of the
        // binning SSBOs only.
        const bool mpBinOverflowAvail = m_micropolySwRasterPass != nullptr;

        ImGui::SetNextWindowPos({620.f, 10.f}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({280.f, 140.f}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Debug Views")) {
            static const char* kModeNames[] = {
                "Lit", "Unlit", "Wireframe", "Lit Wireframe", "Detail Lighting", "Clusters",
                "Micropoly RasterClass", "Micropoly LOD Heatmap",
                "Micropoly Residency Heatmap",
                "Micropoly Bounds",
                "Micropoly BinOverflow"
            };
            int modeInt = static_cast<int>(m_debugMode);
            if (ImGui::Combo("Mode", &modeInt, kModeNames,
                             static_cast<int>(sizeof(kModeNames) / sizeof(kModeNames[0])))) {
                DebugMode selected = static_cast<DebugMode>(modeInt);
                if (!wireAvail && (selected == DebugMode::Wireframe || selected == DebugMode::LitWireframe))
                    selected = DebugMode::Lit;
                if (!clustAvail && selected == DebugMode::Clusters)
                    selected = DebugMode::Lit;
                if (!mpClassAvail && (selected == DebugMode::MicropolyRasterClass
                                      || selected == DebugMode::MicropolyLodHeatmap
                                      || selected == DebugMode::MicropolyResidencyHeatmap))
                    selected = DebugMode::Lit;
                if (!mpBoundsAvail && selected == DebugMode::MicropolyBounds)
                    selected = DebugMode::Lit;
                if (!mpBinOverflowAvail && selected == DebugMode::MicropolyBinOverflowHeat)
                    selected = DebugMode::Lit;
                m_debugMode = selected;
            }
            if (m_debugMode == DebugMode::Wireframe || m_debugMode == DebugMode::LitWireframe)
                ImGui::ColorEdit3("Wire Color", &m_wireframeColor.x);
            ImGui::Separator();
            ImGui::TextDisabled("F1=Lit F2=Unlit F3=Wire F4=LitWire F6=Detail F7=Cluster F8=MpRaster");
            if (!wireAvail)
                ImGui::TextDisabled("Wireframe: requires mesh shaders + fillModeNonSolid");
            if (!mpClassAvail)
                ImGui::TextDisabled("Micropoly RasterClass: requires shaderImageInt64 + HW raster pass");
        }
        ImGui::End();

        // M6.2a: Micropoly runtime settings panel — one window for the
        // enable toggle, page-cache slider, force-HW/SW debug flags, and
        // the overlay radio. Toggled by the checkbox exposed here (no
        // separate main-menu bar in this build); the F11 key provides a
        // quick show/hide shortcut alongside the existing F1..F10 debug
        // hotkeys.
        if (ImGui::IsKeyPressed(ImGuiKey_F11)) {
            m_mpSettingsPanelOpen = !m_mpSettingsPanelOpen;
        }
        // Feed the panel the live cull counters when the pass exists.
        // readbackStats() is cheap (7 u32 loads from a persistent-mapped
        // pointer) so unconditionally snapshotting here is fine. When
        // micropoly is disabled m_micropolyCullPass is null and we pass
        // nullptr so the panel skips the stats section entirely.
        renderer::micropoly::CullStats mpStatsSnapshot{};
        const renderer::micropoly::CullStats* mpStatsPtr = nullptr;
        if (m_micropolyCullPass) {
            mpStatsSnapshot = m_micropolyCullPass->readbackStats();
            mpStatsPtr      = &mpStatsSnapshot;
        }
        m_mpSettingsPanel.draw(&m_mpSettingsPanelOpen,
                               *m_device,
                               m_micropolyConfig,
                               m_debugMode,
                               mpStatsPtr);
    }

    VkImage     targetImage = m_swapchain->image(imageIndex);
    VkImageView targetView  = m_swapchain->view(imageIndex);
    VkImage     depthImage  = m_swapchain->depthImage();
    VkImageView depthView   = m_swapchain->depthView();
    const VkExtent2D extent = m_swapchain->extent();

    // Build the render graph for this frame. All barriers and attachment
    // setup are handled by the graph; drawFrame() holds no hand-coded barriers.
    m_renderGraph->reset();

    const auto colorHandle = m_renderGraph->importImage(
        "swapchain_color",
        targetImage, targetView, m_swapchain->format(),
        VK_IMAGE_LAYOUT_UNDEFINED,       // starts undefined each frame
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, // must end ready for present
        VK_IMAGE_ASPECT_COLOR_BIT);

    // HDR intermediate — only needed for Lit and LitWireframe modes.
    const bool needsHdr = (m_debugMode == DebugMode::Lit || m_debugMode == DebugMode::LitWireframe);
    gfx::RGImageHandle hdrColorHandle{};
    if (needsHdr) {
        hdrColorHandle = m_renderGraph->importImage(
            "hdr_color",
            m_hdrColor, m_hdrColorView, VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_LAYOUT_UNDEFINED,                  // re-written every frame
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,   // upscaler reads as SRV
            VK_IMAGE_ASPECT_COLOR_BIT);
    }

    const u32 cameraSlot = m_cameraBuffers[m_frameIndex].bindlessSlot;

    // ---- Visibility buffer pipeline (direct cmd recording, before render graph) ----
    if (m_scene != nullptr && m_visibilityPass && m_materialEvalPass) {
        m_gpuScene->begin_frame();
        for (const auto& node : m_scene->nodes) {
            for (u32 primIdx : node.primitiveIndices) {
                const auto& prim = m_scene->primitives[primIdx];
                if (prim.meshletOffset == UINT32_MAX) { ENIGMA_ASSERT(false && "primitive missing meshlet data — asset pipeline bug"); continue; }
                GpuInstance inst{};
                inst.transform          = node.worldTransform;
                inst.meshlet_offset     = prim.meshletOffset;
                inst.meshlet_count      = static_cast<u32>(prim.meshlets.meshlets.size());
                inst.material_index     = prim.materialIndex < 0 ? 0u : static_cast<u32>(prim.materialIndex);
                inst.vertex_buffer_slot = prim.vertexBufferSlot;
                inst.vertex_base_offset = 0; // Non-terrain meshes index from vertex 0; CDLOD terrain overrides.
                // inst._pad is zero-initialized by GpuInstance inst{}.
                m_gpuScene->add_instance(inst);
            }
        }
        // Remember the split point: scene instances are [0, sceneInstanceCount),
        // terrain instances (appended below) live at [sceneInstanceCount, total).
        // Used by the two-pass cull + VB draw sequence further down so scene
        // meshlets go through visibility_buffer.mesh.hlsl and terrain meshlets
        // go through terrain_cdlod.mesh.hlsl.
        const u32 sceneInstanceCount  = static_cast<u32>(m_gpuScene->instance_count());
        // Use scene_meshlet_count() — frozen at upload()/reserveCapacity() time —
        // not total_meshlet_count(), which grows as terrain patches activate and
        // would pull terrain-meshlet indices into the scene cull range, causing
        // findInstanceAndLocal to fall through to instanceId=0 → corruption.
        const u32 sceneMeshletCount   = m_gpuMeshlets->scene_meshlet_count();

        // Per-frame CDLOD terrain traversal — activates/deactivates patches,
        // emits a GpuInstance per active patch into the scene buffer, and
        // records incremental vertex + meshlet uploads on this frame's cmd.
        if (m_terrain != nullptr && m_terrain->isEnabled()) {
            m_terrain->update(m_cameraWorldPos, frame.commandBuffer,
                              m_frameIndex, *m_gpuScene);
        }
        const u32 totalInstanceCount  = static_cast<u32>(m_gpuScene->instance_count());
        const u32 totalMeshletCount   = m_gpuMeshlets->total_meshlet_count();
        const u32 terrainInstanceCount = totalInstanceCount - sceneInstanceCount;
        const u32 terrainMeshletCount  = totalMeshletCount - sceneMeshletCount;

        m_gpuScene->upload(frame.commandBuffer, m_frameIndex, totalMeshletCount);

        // ---- Micropoly streaming per-frame pump (M2.4a) ----
        // Call site per plan §3.M2 (ralplan-micropolygon.md line 325): drain
        // the GPU request queue, dispatch async IO, collect completions,
        // allocate PageCache slots, record transfer-queue uploads, submit
        // with a timeline semaphore. When m_micropolyStreaming is null
        // (config disabled or capability-gated off), this is a NO-OP and
        // the frame is bit-identical to the pre-micropoly path.
        //
        // BARRIER NOTE: when M3 adds the compute producer that fills the
        // RequestQueue, a pipeline barrier with dstStageMask=HOST_BIT and
        // dstAccessMask=HOST_READ_BIT | HOST_WRITE_BIT must be emitted on
        // frame.commandBuffer BEFORE this call (the barrier makes the
        // GPU writes visible to the host-side drain below). In M2.4a
        // there is no producer yet — no barrier is needed today, but the
        // call is placed here so the barrier has a natural home later.
        if (m_micropolyStreaming) {
            // Clear the 64-bit vis image to the kMpVisEmpty sentinel (0)
            // before this frame's cull/raster writes. First-frame call also
            // handles the UNDEFINED → GENERAL layout transition. Missing
            // this clear is the reason all atomic-max writes from the HW +
            // SW raster paths were silently discarded — the image stayed in
            // UNDEFINED layout and the driver no-ops storage writes there.
            if (m_micropolyPass) {
                m_micropolyPass->clearVisImage(frame.commandBuffer);

                // Transition clear write (TRANSFER_WRITE) to the storage
                // accesses the raster paths need: FRAGMENT_SHADER for HW
                // PSMain's InterlockedMax, COMPUTE_SHADER for SW raster
                // fragment compute. The image stays in GENERAL.
                VkMemoryBarrier2 postClear{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
                postClear.srcStageMask  = VK_PIPELINE_STAGE_2_CLEAR_BIT
                                        | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                postClear.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                postClear.dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                                        | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                postClear.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                                        | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                dep.memoryBarrierCount = 1u;
                dep.pMemoryBarriers    = &postClear;
                vkCmdPipelineBarrier2(frame.commandBuffer, &dep);
            }
            (void)m_micropolyStreaming->beginFrame();
            // The graphics-queue submit at the end of drawFrame() waits on
            // m_micropolyStreaming->uploadDoneSemaphore() at value
            // uploadCounter(); see the vkQueueSubmit call site.
        }

        // M3.2: cluster cull compute dispatch. Runs AFTER streaming.beginFrame
        // (so newly-uploaded pages are observable by the residency bitmap)
        // and BEFORE any downstream HW/SW raster consumer (M3.3+). Zero-
        // cluster dispatch is legal and is a no-op. The pass emits a
        // compute-write → (DRAW_INDIRECT | TASK | MESH) barrier on the
        // indirect-draw buffer + a (HOST | COMPUTE) barrier on the cull-
        // stats buffer; M3.3 can bind the indirect-draw buffer directly.
        //
        // M3.2 scope: the pass is constructed but the DAG / residency-
        // bitmap SSBO bindless slots are not yet plumbed through (those
        // ride in on M3.3 MicropolyPass rewire). Until then we call
        // resetCounters() every frame so the indirect-draw header stays
        // zero — the pass contributes exactly zero draws. Principle 1
        // holds because when m_micropolyCullPass is null (disabled
        // config), none of this runs at all.
        if (m_micropolyCullPass) {
            m_micropolyCullPass->resetCounters(frame.commandBuffer);
            // Reset→compute barrier so the shader sees a zeroed header.
            VkMemoryBarrier2 cullResetBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            cullResetBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            cullResetBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            cullResetBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            cullResetBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                                            | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            VkDependencyInfo cullResetDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            cullResetDep.memoryBarrierCount = 1u;
            cullResetDep.pMemoryBarriers    = &cullResetBarrier;
            vkCmdPipelineBarrier2(frame.commandBuffer, &cullResetDep);
            // Dispatch with zero clusters until the DAG plumbing lands in
            // M3.3. The pass treats totalClusterCount==0 as a legal no-op
            // (early return, no dispatch, no barrier), matching the
            // gpu_cull pattern. When M3.3 wires in a real DAG SSBO the
            // cluster count will come from MicropolyStreaming's resident
            // page set.
            renderer::micropoly::MicropolyCullPass::DispatchInputs cin{};
            cin.cmd                         = frame.commandBuffer;
            cin.globalSet                   = m_descriptorAllocator->globalSet();
            // Cull processes one DAG node per thread — totalClusterCount is
            // the full DAG size. The cluster-cull shader's leaf-LOD gate
            // (lodLevel != 0) skips non-leaf nodes before any draw emission,
            // so passing dagNodeCount here is correct even though many nodes
            // are parent groups. Zero when the DAG SSBO isn't attached.
            cin.totalClusterCount           =
                (m_micropolyStreaming && m_micropolyStreaming->dagNodeBufferBindless() != UINT32_MAX
                 && m_micropolyReader)
                    ? m_micropolyReader->header().dagNodeCount : 0u;
            cin.dagBufferBindlessIndex      =
                m_micropolyStreaming ? m_micropolyStreaming->dagNodeBufferBindless()
                                     : UINT32_MAX;
            cin.cameraSlot                  = cameraSlot;
            cin.hiZMipCount                 =
                m_hizPass ? static_cast<f32>(m_hizPass->mip_count()) : 1.0f;
            cin.hiZBindlessIndex            =
                m_hizPass ? m_hizPass->mip_slot(0u) : UINT32_MAX;
            cin.screenSpaceErrorThreshold   = m_micropolyConfig.lodScale;
            // Security HIGH-2: pageCount bounds-checks residency-bitmap reads.
            // Zero is legal for totalClusterCount=0 dispatches (early-out).
            cin.pageCount                   =
                m_micropolyReader ? m_micropolyReader->header().pageCount : 0u;
            // M4.4: classifier inputs. Page cache slots + stride feed the
            // per-cluster triangleCount lookup; screenHeight converts world
            // radius into pixel radius for the projected-area threshold.
            if (m_micropolyStreaming) {
                cin.pageToSlotBufferBindlessIndex =
                    m_micropolyStreaming->pageToSlotBufferBindless();
                cin.pageCacheBufferBindlessIndex  =
                    m_micropolyStreaming->pageCache().bindlessIndex();
                cin.pageSlotBytes                 =
                    m_micropolyStreaming->pageCache().slotBytes();
                // M4.5 multi-cluster page support: classifier reads
                // firstDagNodeIdx to derive per-page local cluster index.
                cin.pageFirstDagNodeBufferBindlessIndex =
                    m_micropolyStreaming->pageFirstDagNodeBufferBindless();
                // Bindless slot for the RequestQueue SSBO — the cull shader
                // writes pageId requests here via emitPageReq(). Without
                // this the streaming pump never sees work and every page
                // stays non-resident (cluster cull sees pageToSlot == -1
                // forever, survivors stay at 0).
                cin.requestQueueBindlessIndex =
                    m_micropolyStreaming->requestQueue().bindlessSlot();
            }
            cin.screenHeight                =
                m_swapchain ? m_swapchain->extent().height : 0u;
            m_gpuProfiler->beginZone(frame.commandBuffer, "MicropolyCull");
            m_micropolyCullPass->dispatch(cin);
            m_gpuProfiler->endZone(frame.commandBuffer);

            // M3.3: HW raster dispatch. Consumes the cull pass's indirect-
            // draw buffer (barrier is already emitted by MicropolyCullPass::
            // dispatch: COMPUTE_SHADER_BIT -> DRAW_INDIRECT | TASK | MESH).
            // The vis image is kept in VK_IMAGE_LAYOUT_GENERAL across the
            // frame — MicropolyPass::clearVisImage handled the initial
            // UNDEFINED -> GENERAL transition, and atomic-min writes are
            // legal in GENERAL layout.
            //
            // M3.3 zero-cluster safety: the cull pass emits zero draws
            // until the DAG SSBO is wired (tracked as plan follow-up
            // M3.3a). vkCmdDrawMeshTasksIndirectCountEXT with count==0 is
            // a legal no-op — no draws dispatched. This keeps Principle 1
            // safe even though the raster pass IS bound.
            if (m_micropolyRasterPass && m_micropolyStreaming) {
                // Make beginFrame()'s host-coherent writes to pageToSlotBuffer
                // visible to TASK/MESH shader reads. Persistent-mapped host-
                // coherent memory still requires an explicit HOST_WRITE ->
                // SHADER_READ memory dependency per Vulkan spec 7.1.1.
                // M4.2: the SW raster binning compute (sw_raster_bin.comp.hlsl)
                // also reads pageToSlotBuffer via bindless, so COMPUTE_SHADER
                // must be in dstStageMask alongside the TASK/MESH stages.
                VkMemoryBarrier2 hostBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                hostBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT;
                hostBarrier.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
                hostBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT
                                          | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT
                                          | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                hostBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                VkDependencyInfo hostDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                hostDep.memoryBarrierCount = 1u;
                hostDep.pMemoryBarriers    = &hostBarrier;
                vkCmdPipelineBarrier2(frame.commandBuffer, &hostDep);

                renderer::micropoly::MicropolyRasterPass::DispatchInputs rin{};
                rin.cmd                            = frame.commandBuffer;
                rin.globalSet                      = m_descriptorAllocator->globalSet();
                rin.indirectBuffer                 = m_micropolyCullPass->indirectDrawBuffer();
                rin.indirectBufferBindlessIndex    = m_micropolyCullPass->indirectDrawBindlessSlot();
                // M3.3-deferred DAG SSBO wire-up: replaces the UINT32_MAX
                // stub. The assembled runtime DAG (MicropolyStreaming's
                // DEVICE_LOCAL SSBO) carries one 48 B node per on-disk
                // MpDagNode, joining cone data from per-page ClusterOnDisk.
                rin.dagBufferBindlessIndex         =
                    m_micropolyStreaming->dagNodeBufferBindless();
                rin.pageToSlotBufferBindlessIndex  = m_micropolyStreaming->pageToSlotBufferBindless();
                rin.pageCacheBufferBindlessIndex   = m_micropolyStreaming->pageCache().bindlessIndex();
                rin.cameraSlot                     = cameraSlot;
                rin.visImageBindlessIndex          = m_micropolyPass ? m_micropolyPass->visBindlessSlot() : UINT32_MAX;
                rin.extent                         = m_swapchain->extent();
                rin.pageSlotBytes                  = m_micropolyStreaming->pageCache().slotBytes();
                rin.pageCount                      = m_micropolyStreaming->pageToSlotPageCount();
                rin.dagNodeCount                   =
                    m_micropolyReader ? m_micropolyReader->header().dagNodeCount : 0u;
                rin.maxClusters                    = renderer::micropoly::kMpMaxIndirectDrawClusters;
                // M4.4 dispatcher classifier tag SSBO — task shader gates
                // DispatchMesh(0,...) on clusters classified SW.
                rin.rasterClassBufferBindlessIndex =
                    m_micropolyCullPass->rasterClassBufferBindlessSlot();
                // M4.5 multi-cluster: task shader reads firstDagNodeIdx to
                // compute per-cluster localClusterIdx, then forwards it to
                // the mesh shader via the task->mesh payload.
                rin.pageFirstDagNodeBufferBindlessIndex =
                    m_micropolyStreaming->pageFirstDagNodeBufferBindless();
                m_gpuProfiler->beginZone(frame.commandBuffer, "MicropolyHwRaster");
                m_micropolyRasterPass->record(rin);
                m_gpuProfiler->endZone(frame.commandBuffer);

                // M4.2: SW raster binning. Shares the same cull indirect
                // buffer + DAG/page bindings as the HW raster. The HW
                // raster pass already emitted a FRAGMENT_SHADER ->
                // COMPUTE_SHADER barrier on the vis image; we pile on a
                // separate set of SSBO writes (tileBinCount / Entries /
                // spill / dispatchIndirect) that no other pass reads yet.
                // The binning dispatch is pure additive — when M4.3 lands
                // the fragment compute just consumes these outputs.
                if (m_micropolySwRasterPass) {
                    renderer::micropoly::MicropolySwRasterPass::DispatchInputs sin{};
                    sin.cmd                            = frame.commandBuffer;
                    sin.globalSet                      = m_descriptorAllocator->globalSet();
                    sin.indirectBuffer                 = m_micropolyCullPass->indirectDrawBuffer();
                    sin.indirectBufferBindlessIndex    = m_micropolyCullPass->indirectDrawBindlessSlot();
                    sin.dagBufferBindlessIndex         = rin.dagBufferBindlessIndex;
                    sin.pageToSlotBufferBindlessIndex  = rin.pageToSlotBufferBindlessIndex;
                    sin.pageCacheBufferBindlessIndex   = rin.pageCacheBufferBindlessIndex;
                    sin.cameraSlot                     = rin.cameraSlot;
                    // M4.3: feed the R64_UINT vis image's bindless slot so
                    // the fragment raster pipeline can InterlockedMax into
                    // the same image the HW raster's PSMain writes. Both
                    // paths co-exist on a single vis image (reverse-Z).
                    sin.visImage64Bindless             = rin.visImageBindlessIndex;
                    sin.extent                         = rin.extent;
                    sin.pageSlotBytes                  = rin.pageSlotBytes;
                    sin.pageCount                      = rin.pageCount;
                    sin.dagNodeCount                   = rin.dagNodeCount;
                    // M4.4 dispatcher classifier tag — bin shader's
                    // workgroup-thread-0 early-outs when cluster is HW.
                    sin.rasterClassBufferBindlessIndex = rin.rasterClassBufferBindlessIndex;
                    // M4.5 multi-cluster: same firstDagNodeIdx slot both
                    // bin + raster compute paths consume.
                    sin.pageFirstDagNodeBufferBindlessIndex =
                        rin.pageFirstDagNodeBufferBindlessIndex;
                    m_gpuProfiler->beginZone(frame.commandBuffer, "MicropolySwRaster");
                    m_micropolySwRasterPass->record(sin);
                    m_gpuProfiler->endZone(frame.commandBuffer);
                }
            }
        }

        // Throttled per-second cull-stats diagnostic. Helps catch
        // "is cull producing any survivors?" without opening the ImGui
        // panel. readbackStats() is cheap (7 u32 loads from a persistent-
        // mapped pointer); stats reflect the PREVIOUS frame (the one whose
        // submit has fenced). Guarded on micropoly enabled + pass alive so
        // there's zero cost on disabled/non-capable paths.
        if (m_micropolyConfig.enabled && m_micropolyCullPass != nullptr) {
            static auto sCullStatLastLog = std::chrono::steady_clock::now()
                                         - std::chrono::seconds(2); // force first-hit log
            const auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<f64>(now - sCullStatLastLog).count() > 1.0) {
                sCullStatLastLog = now;
                const auto s = m_micropolyCullPass->readbackStats();
                ENIGMA_LOG_INFO(
                    "[micropoly_cull] dispatched={} visible={} "
                    "culled(LOD={} resid={} frus={} back={} hiz={})",
                    s.totalDispatched, s.visible, s.culledLOD,
                    s.culledResidency, s.culledFrustum,
                    s.culledBackface, s.culledHiZ);
                // Surface per-pass GPU timings on the same cadence so we can
                // attribute 5-fps regressions without the ImGui overlay.
                f32 cullMs = 0.0f, hwMs = 0.0f, swMs = 0.0f;
                for (const auto& z : m_lastGpuTimings) {
                    if (z.name == "MicropolyCull")      cullMs = z.durationMs;
                    else if (z.name == "MicropolyHwRaster") hwMs = z.durationMs;
                    else if (z.name == "MicropolySwRaster") swMs = z.durationMs;
                }
                ENIGMA_LOG_INFO(
                    "[micropoly_gpu] cull={:.3f}ms hwRaster={:.3f}ms swRaster={:.3f}ms",
                    cullMs, hwMs, swMs);
            }
        }

        m_indirectBuffer->reset_count(frame.commandBuffer);

        // Barrier: all TRANSFER writes (scene SSBO upload + indirect count reset) →
        //   DRAW_INDIRECT  (count buffer read by vkCmdDrawMeshTasksEXT — unused now but harmless)
        //   COMPUTE_SHADER (cull pass reads scene SSBO)
        //   TASK_SHADER    (task shader reads scene SSBO in findInstanceAndLocal)
        //   MESH_SHADER    (mesh shader reads scene SSBO in loadInstance for transform +
        //                   vertex_buffer_slot; task→mesh ordering only propagates task writes,
        //                   not third-party TRANSFER writes, so explicit coverage is required)
        // Must be unconditional — the reset happens even if there are no meshlets.
        {
            VkMemoryBarrier2 transferBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            transferBarrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            transferBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            transferBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            transferBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT
                                          | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                          | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT
                                          | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
            transferBarrier.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT
                                          | VK_ACCESS_2_SHADER_READ_BIT;
            VkDependencyInfo transferDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            transferDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            transferDep.memoryBarrierCount = 1;
            transferDep.pMemoryBarriers    = &transferBarrier;
            vkCmdPipelineBarrier2(frame.commandBuffer, &transferDep);
        }

        if (totalMeshletCount > 0) {
            // Helper: compute→task/mesh barrier on the surviving-IDs +
            // count buffer so the task shader sees the cull output.
            auto emitCullToTaskBarrier = [&]() {
                VkMemoryBarrier2 cullBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                cullBarrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                cullBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                cullBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
                cullBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT;
                cullBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                VkDependencyInfo cullDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                cullDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                cullDep.memoryBarrierCount = 1;
                cullDep.pMemoryBarriers    = &cullBarrier;
                vkCmdPipelineBarrier2(frame.commandBuffer, &cullDep);
            };

            bool terrainRan = false;

            // ---------- Pass B: Terrain meshlets (runs FIRST — clears vis+depth) ----------
            // Terrain is drawn before scene so the scene pass (LOAD) renders on top
            // and always wins at coplanar depth. recordTerrain() CLEARs both images.
            // Skip in pure Wireframe mode so scene survivors remain in the indirect
            // buffer for the wireframe overlay.
            if (m_terrain != nullptr && m_terrain->isEnabled()
                && terrainMeshletCount > 0 && terrainInstanceCount > 0
                && m_visibilityPass->hasTerrainPipeline()
                && m_debugMode != DebugMode::Wireframe)
            {
                // Initial reset_count + transfer barrier above already ran.
                m_gpuProfiler->beginZone(frame.commandBuffer, "CullTerrain");
                m_gpuCullPass->record(frame.commandBuffer,
                                       m_descriptorAllocator->globalSet(),
                                       *m_gpuScene, *m_gpuMeshlets, *m_indirectBuffer,
                                       cameraSlot,
                                       sceneInstanceCount, terrainInstanceCount,
                                       sceneMeshletCount,  terrainMeshletCount);
                emitCullToTaskBarrier();
                m_gpuProfiler->endZone(frame.commandBuffer);

                const auto terrainTopo = m_terrain->sharedTopologyHandle();
                m_gpuProfiler->beginZone(frame.commandBuffer, "DrawTerrainVB");
                m_visibilityPass->recordTerrain(
                    frame.commandBuffer, m_descriptorAllocator->globalSet(), extent,
                    m_gbufDepth.view, m_gbufDepth.image,
                    *m_gpuScene, *m_gpuMeshlets, *m_indirectBuffer, cameraSlot,
                    terrainTopo.topologyVerticesSlot,
                    terrainTopo.topologyTrianglesSlot,
                    terrainMeshletCount);
                m_gpuProfiler->endZone(frame.commandBuffer);

                terrainRan = true;
            }

            // ---------- Pass A: Scene meshlets (runs SECOND — LOADs vis+depth) ----------
            // When terrainRan==true the images are in SHADER_READ_ONLY_OPTIMAL after
            // recordTerrain()'s post-barriers; record(clearFirst=false) transitions
            // them back to COLOR_ATTACHMENT with LOAD so scene fragments overwrite
            // terrain at equal or greater depth (GREATER_OR_EQUAL, reverse-Z).
            // When terrainRan==false (Wireframe or no terrain), clearFirst=true
            // initialises both images from UNDEFINED with CLEAR.
            if (sceneMeshletCount > 0 && sceneInstanceCount > 0) {
                if (terrainRan) {
                    // Terrain task/mesh reads → transfer+compute writes for the reset.
                    VkMemoryBarrier2 readToWrite{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                    readToWrite.srcStageMask  = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT
                                              | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
                    readToWrite.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                    readToWrite.dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT
                                              | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    readToWrite.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT
                                              | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                    dep.memoryBarrierCount = 1;
                    dep.pMemoryBarriers    = &readToWrite;
                    vkCmdPipelineBarrier2(frame.commandBuffer, &dep);

                    m_indirectBuffer->reset_count(frame.commandBuffer);

                    VkMemoryBarrier2 resetBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                    resetBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                    resetBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                    resetBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    resetBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                                               | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    VkDependencyInfo resetDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                    resetDep.memoryBarrierCount = 1;
                    resetDep.pMemoryBarriers    = &resetBarrier;
                    vkCmdPipelineBarrier2(frame.commandBuffer, &resetDep);
                }

                m_gpuProfiler->beginZone(frame.commandBuffer, "CullScene");
                m_gpuCullPass->record(frame.commandBuffer,
                                       m_descriptorAllocator->globalSet(),
                                       *m_gpuScene, *m_gpuMeshlets, *m_indirectBuffer,
                                       cameraSlot,
                                       /*instanceOffset*/ 0, sceneInstanceCount,
                                       /*meshletOffset*/  0, sceneMeshletCount);

                m_indirectBuffer->record_count_readback(frame.commandBuffer);

                emitCullToTaskBarrier();
                m_gpuProfiler->endZone(frame.commandBuffer);

                m_gpuProfiler->beginZone(frame.commandBuffer, "DrawSceneVB");
                m_visibilityPass->record(frame.commandBuffer, m_descriptorAllocator->globalSet(),
                                          extent,
                                          m_gbufDepth.view, m_gbufDepth.image,
                                          *m_gpuScene, *m_gpuMeshlets, *m_indirectBuffer, cameraSlot,
                                          /*clearFirst=*/!terrainRan);
                m_gpuProfiler->endZone(frame.commandBuffer);
            }

            // --- Terrain wireframe cull (Wireframe + LitWireframe debug modes) ---
            // Culls terrain meshlets into m_terrainWireIndirectBuffer so the wireframe
            // overlay pass can draw terrain lines independently of m_indirectBuffer
            // (which at this point holds scene survivors only).
            if ((m_debugMode == DebugMode::Wireframe || m_debugMode == DebugMode::LitWireframe)
                && m_terrain != nullptr && m_terrain->isEnabled()
                && terrainMeshletCount > 0 && terrainInstanceCount > 0
                && m_visibilityPass->hasTerrainWireframePipeline()
                && m_terrainWireIndirectBuffer)
            {
                m_terrainWireIndirectBuffer->reset_count(frame.commandBuffer);

                VkMemoryBarrier2 wireResetBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                wireResetBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                wireResetBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                wireResetBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                wireResetBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                VkDependencyInfo wireResetDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                wireResetDep.memoryBarrierCount = 1;
                wireResetDep.pMemoryBarriers    = &wireResetBarrier;
                vkCmdPipelineBarrier2(frame.commandBuffer, &wireResetDep);

                m_gpuProfiler->beginZone(frame.commandBuffer, "CullTerrainWire");
                m_gpuCullPass->record(frame.commandBuffer,
                                       m_descriptorAllocator->globalSet(),
                                       *m_gpuScene, *m_gpuMeshlets, *m_terrainWireIndirectBuffer,
                                       cameraSlot,
                                       sceneInstanceCount, terrainInstanceCount,
                                       sceneMeshletCount,  terrainMeshletCount);

                emitCullToTaskBarrier();
                m_gpuProfiler->endZone(frame.commandBuffer);
            }

            // MaterialEvalPass reads vis buffer + depth, writes G-buffer to SHADER_READ_ONLY_OPTIMAL.
            // M3.4: also consumes the 64-bit Micropoly vis image via MP_ENABLE
            // spec constant. enableMp mirrors the capability gate used for
            // constructing MicropolyCullPass/MicropolyRasterPass so the three
            // passes agree on whether the micropoly path is live this frame.
            // The raster pass emits its own FRAGMENT_SHADER -> COMPUTE_SHADER
            // memory barrier on the vis image before returning from record(),
            // so no extra Renderer-level barrier is required here.
            const u32  vis64Slot =
                m_micropolyPass ? m_micropolyPass->visBindlessSlot() : UINT32_MAX;
            const bool enableMp  =
                (m_micropolyCullPass != nullptr)
             && (m_micropolyRasterPass != nullptr)
             && (m_micropolyStreaming != nullptr)
             && (vis64Slot != UINT32_MAX);
            // M5 material resolution — thread the mp geometry bindless slots
            // so the mpWins branch can walk DAG -> page -> vertices and
            // output real world-space normals instead of the magenta stamp.
            MaterialEvalPass::MpResolveInputs mpResolve{};
            if (enableMp) {
                mpResolve.dagBufferSlot        = m_micropolyStreaming->dagNodeBufferBindless();
                mpResolve.pageToSlotSlot       = m_micropolyStreaming->pageToSlotBufferBindless();
                mpResolve.pageCacheSlot        = m_micropolyStreaming->pageCache().bindlessIndex();
                mpResolve.pageFirstDagNodeSlot = m_micropolyStreaming->pageFirstDagNodeBufferBindless();
                mpResolve.pageSlotBytes        = m_micropolyStreaming->pageCache().slotBytes();
                mpResolve.pageCount            = m_micropolyStreaming->pageToSlotPageCount();
                mpResolve.dagNodeCount         =
                    m_micropolyReader ? m_micropolyReader->header().dagNodeCount : 0u;
            }
            m_gpuProfiler->beginZone(frame.commandBuffer, "MaterialEval");
            m_materialEvalPass->record(frame.commandBuffer, m_descriptorAllocator->globalSet(),
                                        extent, m_visibilityPass->vis_buffer_slot(),
                                        *m_gpuScene, *m_gpuMeshlets,
                                        m_scene->materialBufferSlot, cameraSlot,
                                        vis64Slot, enableMp, mpResolve);
            m_gpuProfiler->endZone(frame.commandBuffer);

            // Micropoly scaffold (M0a) — record() is a no-op whenever
            // MicropolyConfig::enabled is false, which is the default. This
            // call only reserves the render-graph slot for future work;
            // with enabled=false it emits no commands, no barriers, no
            // descriptor updates. Moving this line or wrapping it in a
            // different predicate breaks the Principle 1 invariant.
            //
            // m_micropolyPass is always non-null after Renderer construction
            // (see ctor: constructed unconditionally, gated internally via
            // active()). Calling record() directly matches the idiom used
            // by every other mandatory pass (m_materialEvalPass, m_lightingPass,
            // m_postProcessPass, etc.).
            m_micropolyPass->record(frame.commandBuffer);
        }
    }

    if (m_scene != nullptr) {
        // G-buffer images are in SHADER_READ_ONLY_OPTIMAL from MaterialEvalPass.
        const auto gbufAlbedo = m_renderGraph->importImage(
            "gbuf_albedo", m_gbufAlbedo.image, m_gbufAlbedo.view,
            gbuf::kAlbedoFormat,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_ASPECT_COLOR_BIT);
        const auto gbufNormal = m_renderGraph->importImage(
            "gbuf_normal", m_gbufNormal.image, m_gbufNormal.view,
            gbuf::kNormalFormat,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_ASPECT_COLOR_BIT);
        const auto gbufMetalRough = m_renderGraph->importImage(
            "gbuf_metalrough", m_gbufMetalRough.image, m_gbufMetalRough.view,
            gbuf::kMetalRoughFormat,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_ASPECT_COLOR_BIT);
        const auto gbufMotionVec = m_renderGraph->importImage(
            "gbuf_motionvec", m_gbufMotionVec.image, m_gbufMotionVec.view,
            gbuf::kMotionVecFormat,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_ASPECT_COLOR_BIT);
        const auto gbufDepth = m_renderGraph->importImage(
            "gbuf_depth", m_gbufDepth.image, m_gbufDepth.view,
            gbuf::kDepthFormat,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_ASPECT_DEPTH_BIT);

        // CDLOD terrain is drawn through the visibility-buffer pipeline —
        // per-patch GpuInstances emitted in update() are culled and rasterised
        // alongside scene meshes. No dedicated raster pass is required here.

        // ---- Mode-dependent render graph passes ----
        if (m_debugMode == DebugMode::Lit) {

        // RT reflection pass — runs between G-buffer and lighting.
        // On RT hardware: dispatches vkCmdTraceRaysKHR.
        // On Min tier: no-op (SSR integration deferred to Phase 2).
        if (m_rtReflectionsEnabled && m_device->gpuTier() >= gfx::GpuTier::Recommended && m_scene->tlas.has_value()) {
            // RT reflection pass records its own barriers and is not a
            // raster pass, so we inject it as a custom execute callback
            // in a dummy raster pass with no color targets.
            gfx::RenderGraph::RasterPassDesc rtReflectDesc{};
            rtReflectDesc.name    = "RTReflectionPass";
            rtReflectDesc.sampledInputs = {gbufNormal, gbufDepth};
            rtReflectDesc.execute = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "RTReflectionPass");
                m_rtReflectionPass->record(cmd, m_descriptorAllocator->globalSet(), ext,
                                            m_gbufNormalSlot, m_gbufDepthSlot,
                                            cameraSlot, m_gbufferSamplerSlot,
                                            m_tlasSlot, m_reflectionSlot,
                                            m_atmospherePass->skyViewLutSlot(),
                                            m_atmospherePass->transmittanceLutSlot(),
                                            vec4{m_sunWorldDir, m_atmosphereSettings.sunIntensity},
                                            vec4{m_cameraWorldPos * 0.001f, 0.0f});
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(rtReflectDesc));
        }

        // RT GI pass — conditional on gpuTier >= Recommended.
        if (m_rtGIEnabled && m_device->gpuTier() >= gfx::GpuTier::Recommended && m_scene->tlas.has_value()) {
            gfx::RenderGraph::RasterPassDesc rtGIDesc{};
            rtGIDesc.name    = "RTGIPass";
            rtGIDesc.sampledInputs = {gbufNormal, gbufDepth};
            rtGIDesc.execute = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "RTGIPass");
                m_giPass->record(cmd, m_descriptorAllocator->globalSet(), ext,
                                  m_gbufNormalSlot, m_gbufDepthSlot,
                                  cameraSlot, m_gbufferSamplerSlot, m_tlasSlot, m_giSlot,
                                  m_atmospherePass->skyViewLutSlot(),
                                  m_atmospherePass->transmittanceLutSlot(),
                                  vec4{m_sunWorldDir, m_atmosphereSettings.sunIntensity},
                                  vec4{m_cameraWorldPos * 0.001f, 0.0f});
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(rtGIDesc));
        }

        // RT Shadow pass — conditional on gpuTier >= Recommended.
        if (m_rtShadowsEnabled && m_device->gpuTier() >= gfx::GpuTier::Recommended && m_scene->tlas.has_value()) {
            gfx::RenderGraph::RasterPassDesc rtShadowDesc{};
            rtShadowDesc.name    = "RTShadowPass";
            rtShadowDesc.sampledInputs = {gbufNormal, gbufDepth};
            rtShadowDesc.execute = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "RTShadowPass");
                m_shadowPass->record(cmd, m_descriptorAllocator->globalSet(), ext,
                                      m_gbufNormalSlot, m_gbufDepthSlot,
                                      cameraSlot, m_tlasSlot, m_shadowSlot,
                                      vec4{m_sunWorldDir, 0.02f}); // 0.02 rad cone angle
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(rtShadowDesc));
        }

        // Wet Road pass — conditional on gpuTier >= Recommended.
        if (m_wetRoadEnabled && m_device->gpuTier() >= gfx::GpuTier::Recommended && m_scene->tlas.has_value()) {
            gfx::RenderGraph::RasterPassDesc wetRoadDesc{};
            wetRoadDesc.name    = "WetRoadPass";
            wetRoadDesc.sampledInputs = {gbufNormal, gbufDepth};
            wetRoadDesc.execute = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "WetRoadPass");
                m_wetRoadPass->record(cmd, m_descriptorAllocator->globalSet(), ext,
                                       m_gbufNormalSlot, m_gbufDepthSlot,
                                       cameraSlot, m_gbufferSamplerSlot, m_tlasSlot, m_wetRoadSlot,
                                       m_atmospherePass->skyViewLutSlot(),
                                       m_atmospherePass->transmittanceLutSlot(),
                                       vec4{m_sunWorldDir, m_atmosphereSettings.sunIntensity},
                                       vec4{m_cameraWorldPos * 0.001f, 0.0f},
                                       m_wetnessFactor);
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(wetRoadDesc));
        }

        // Denoise RT effects — conditional on gpuTier >= Recommended.
        if (m_denoiseEnabled && m_device->gpuTier() >= gfx::GpuTier::Recommended && m_scene->tlas.has_value()) {
            gfx::RenderGraph::RasterPassDesc denoiseDesc{};
            denoiseDesc.name = "DenoisePass";
            denoiseDesc.execute = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "DenoisePass");
                m_reflectionDenoiser->record(cmd, m_descriptorAllocator->globalSet(), ext,
                                              m_reflectionSlot, m_gbufMotionVecSlot,
                                              m_reflDenoiseSlot, m_reflDenoiseSlot);
                m_giDenoiser->record(cmd, m_descriptorAllocator->globalSet(), ext,
                                      m_giSlot, m_gbufMotionVecSlot,
                                      m_giDenoiseSlot, m_giDenoiseSlot);
                m_shadowDenoiser->record(cmd, m_descriptorAllocator->globalSet(), ext,
                                          m_shadowSlot, m_gbufMotionVecSlot,
                                          m_shadowDenoiseSlot, m_shadowDenoiseSlot);
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(denoiseDesc));
        }

        gfx::RenderGraph::RasterPassDesc lightPassDesc{};
        lightPassDesc.name          = "LightingPass";
        lightPassDesc.colorTargets  = {hdrColorHandle};
        lightPassDesc.sampledInputs = {gbufAlbedo, gbufNormal, gbufMetalRough, gbufDepth};
        lightPassDesc.clearColor    = {{0.02f, 0.02f, 0.05f, 1.0f}};
        lightPassDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
            m_gpuProfiler->beginZone(cmd, "LightingPass");
            m_lightingPass->record(cmd, m_descriptorAllocator->globalSet(), ext,
                                    m_gbufAlbedoSlot, m_gbufNormalSlot,
                                    m_gbufMetalRoughSlot, m_gbufDepthSlot,
                                    cameraSlot, m_gbufferSamplerSlot,
                                    vec4{m_sunWorldDir, m_atmosphereSettings.sunIntensity},
                                    vec4{m_light.color, 0.0f});
            m_gpuProfiler->endZone(cmd);
        };
        m_renderGraph->addRasterPass(std::move(lightPassDesc));

        // Sky background — overwrites sky pixels (depth==0) with the SkyView LUT.
        // Must run after LightingPass (LOAD_OP_LOAD preserves geometry pixels)
        // and before PhysicsDebug (which re-attaches gbufDepth as depth target).
        // gbufDepth stays in SHADER_READ_ONLY_OPTIMAL from the LightingPass,
        // so we list it in sampledInputs to let the graph verify the layout.
        {
            gfx::RenderGraph::RasterPassDesc skyDesc{};
            skyDesc.name         = "SkyBackgroundPass";
            skyDesc.colorTargets = {hdrColorHandle};
            skyDesc.colorLoadOp  = VK_ATTACHMENT_LOAD_OP_LOAD;
            skyDesc.sampledInputs = {gbufDepth};
            skyDesc.execute      = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "SkyBackgroundPass");
                m_skyPass->record(cmd,
                    m_descriptorAllocator->globalSet(), ext,
                    cameraSlot,
                    m_gbufDepthSlot,
                    m_atmospherePass->skyViewLutSlot(),
                    m_atmospherePass->transmittanceLutSlot(),
                    m_gbufferSamplerSlot,
                    vec4{m_sunWorldDir, m_atmosphereSettings.sunIntensity},
                    vec4{m_cameraWorldPos * 0.001f, 0.0f}); // metres → km
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(skyDesc));
        }

        // Physics debug wireframe overlay — inserted after lighting so it
        // composites on top of the final lit image. Only active when enabled
        // and there are lines to draw (upload() sets the count).
        if (m_physicsDebugRenderer.enabled) {
            m_physicsDebugRenderer.upload();

            gfx::RenderGraph::RasterPassDesc debugDesc{};
            debugDesc.name        = "PhysicsDebugPass";
            debugDesc.colorTargets = {hdrColorHandle};
            debugDesc.colorLoadOp  = VK_ATTACHMENT_LOAD_OP_LOAD;
            if (m_physicsDebugRenderer.depthTestEnabled) {
                // Re-attach the G-buffer depth read-only so lines are occluded
                // by scene geometry. depthWriteEnable=false in the pipeline
                // ensures the depth buffer is never modified.
                debugDesc.depthTarget = gbufDepth;
                debugDesc.depthLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            }
            debugDesc.execute = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "PhysicsDebugPass");
                m_physicsDebugRenderer.drawFrame(cmd, m_descriptorAllocator->globalSet(),
                                                  ext, cameraSlot);
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(debugDesc));
        }

        // Clustered forward pass — transparent geometry blended onto HDR after sky/debug.
        if (m_scene != nullptr && m_clusteredForwardPass) {
            gfx::RenderGraph::RasterPassDesc fwdDesc{};
            fwdDesc.name         = "ClusteredForwardPass";
            fwdDesc.colorTargets = {hdrColorHandle};
            fwdDesc.colorLoadOp  = VK_ATTACHMENT_LOAD_OP_LOAD;
            fwdDesc.depthTarget  = gbufDepth;
            fwdDesc.depthLoadOp  = VK_ATTACHMENT_LOAD_OP_LOAD;
            fwdDesc.execute      = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "ClusteredForwardPass");
                m_clusteredForwardPass->record(
                    cmd, m_descriptorAllocator->globalSet(), ext, *m_scene,
                    cameraSlot, m_scene->materialBufferSlot,
                    m_sunWorldDir, vec3(1.0f), m_atmosphereSettings.sunIntensity);
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(fwdDesc));
        }

        // PostProcessPass + optional SMAA.
        // When SMAA is on: PostProcess → LDR intermediate, then 3 SMAA passes → swapchain.
        // When SMAA is off: PostProcess → swapchain directly.
        const bool smaaActive = m_aaSettings.smaaEnabled
                             && m_smaaPass
                             && m_smaaPass->isAllocated()
                             && m_smaaLdrImage != VK_NULL_HANDLE;

        if (smaaActive) {
            const auto ldrHandle = m_renderGraph->importImage(
                "smaa_ldr",
                m_smaaLdrImage, m_smaaLdrView, m_swapchain->format(),
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT);
            const auto smaaEdgeHandle = m_renderGraph->importImage(
                "smaa_edge",
                m_smaaPass->edgeImage(), m_smaaPass->edgeView(), VK_FORMAT_R8G8_UNORM,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT);
            const auto smaaWeightHandle = m_renderGraph->importImage(
                "smaa_weight",
                m_smaaPass->weightImage(), m_smaaPass->weightView(), VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT);

            // PostProcessPass → LDR intermediate.
            {
                gfx::RenderGraph::RasterPassDesc ppDesc{};
                ppDesc.name          = "PostProcessPass";
                ppDesc.colorTargets  = {ldrHandle};
                ppDesc.sampledInputs = {hdrColorHandle, gbufDepth};
                ppDesc.clearColor    = {{0.0f, 0.0f, 0.0f, 1.0f}};
                ppDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                    m_gpuProfiler->beginZone(cmd, "PostProcessPass");
                    m_postProcessPass->record(cmd,
                        m_descriptorAllocator->globalSet(),
                        m_atmospherePass->aerialPerspectiveReadSet(),
                        ext,
                        m_hdrColorSampledSlot,
                        m_gbufDepthSlot,
                        cameraSlot,
                        m_gbufferSamplerSlot,
                        m_linearSamplerSlot,
                        m_atmosphereSettings,
                        vec4{m_cameraWorldPos * 0.001f, 0.0f});
                    m_gpuProfiler->endZone(cmd);
                };
                m_renderGraph->addRasterPass(std::move(ppDesc));
            }

            // SMAA Pass 1: luma edge detection — LDR → edge texture.
            {
                gfx::RenderGraph::RasterPassDesc edgeDesc{};
                edgeDesc.name          = "SMAAEdgePass";
                edgeDesc.colorTargets  = {smaaEdgeHandle};
                edgeDesc.sampledInputs = {ldrHandle};
                edgeDesc.clearColor    = {{0.0f, 0.0f, 0.0f, 0.0f}};
                edgeDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                    m_gpuProfiler->beginZone(cmd, "SMAAEdgePass");
                    // Linear sampler for all three SMAA passes: the reference
                    // algorithm relies on sub-texel bilinear interpolation
                    // (edge search taps, area/search tex lookups, neighbourhood
                    // 0.5-texel sampling). Nearest here would break the math.
                    m_smaaPass->recordEdge(cmd, m_descriptorAllocator->globalSet(), ext,
                                            m_smaaLdrSampledSlot, m_linearSamplerSlot);
                    m_gpuProfiler->endZone(cmd);
                };
                m_renderGraph->addRasterPass(std::move(edgeDesc));
            }

            // SMAA Pass 2: blending weights — edge → weight texture.
            {
                gfx::RenderGraph::RasterPassDesc blendDesc{};
                blendDesc.name          = "SMAABlendPass";
                blendDesc.colorTargets  = {smaaWeightHandle};
                blendDesc.sampledInputs = {smaaEdgeHandle};
                blendDesc.clearColor    = {{0.0f, 0.0f, 0.0f, 0.0f}};
                blendDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                    m_gpuProfiler->beginZone(cmd, "SMAABlendPass");
                    m_smaaPass->recordBlend(cmd, m_descriptorAllocator->globalSet(), ext,
                                             m_linearSamplerSlot);
                    m_gpuProfiler->endZone(cmd);
                };
                m_renderGraph->addRasterPass(std::move(blendDesc));
            }

            // SMAA Pass 3: neighbourhood blend — LDR + weights → swapchain.
            {
                gfx::RenderGraph::RasterPassDesc nbrDesc{};
                nbrDesc.name          = "SMAANeighborPass";
                nbrDesc.colorTargets  = {colorHandle};
                nbrDesc.sampledInputs = {ldrHandle, smaaWeightHandle};
                nbrDesc.clearColor    = {{0.0f, 0.0f, 0.0f, 1.0f}};
                nbrDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                    m_gpuProfiler->beginZone(cmd, "SMAANeighborPass");
                    // Neighborhood pass MUST use the linear sampler: the shader samples at
                    // uv ± 0.5*texel and relies on bilinear interpolation to produce the
                    // sub-pixel blend between the current pixel and its neighbour. A
                    // nearest sampler snaps back to one of the two pixels, collapsing the
                    // AA effect to a no-op (SMAA appears disabled).
                    m_smaaPass->recordNeighborhood(cmd, m_descriptorAllocator->globalSet(), ext,
                                                    m_smaaLdrSampledSlot, m_linearSamplerSlot);
                    m_gpuProfiler->endZone(cmd);
                };
                m_renderGraph->addRasterPass(std::move(nbrDesc));
            }
        } else {
            // PostProcessPass — final output directly to swapchain.
            gfx::RenderGraph::RasterPassDesc ppDesc{};
            ppDesc.name          = "PostProcessPass";
            ppDesc.colorTargets  = {colorHandle};
            ppDesc.sampledInputs = {hdrColorHandle, gbufDepth};
            ppDesc.clearColor    = {{0.0f, 0.0f, 0.0f, 1.0f}};
            ppDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "PostProcessPass");
                m_postProcessPass->record(cmd,
                    m_descriptorAllocator->globalSet(),
                    m_atmospherePass->aerialPerspectiveReadSet(),
                    ext,
                    m_hdrColorSampledSlot,
                    m_gbufDepthSlot,
                    cameraSlot,
                    m_gbufferSamplerSlot,
                    m_linearSamplerSlot,
                    m_atmosphereSettings,
                    vec4{m_cameraWorldPos * 0.001f, 0.0f});
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(ppDesc));
        }

        } else if (m_debugMode == DebugMode::Unlit) {
            // === UNLIT — read albedo G-buffer, skip lighting ===
            gfx::RenderGraph::RasterPassDesc dbgDesc{};
            dbgDesc.name          = "DebugUnlitPass";
            dbgDesc.colorTargets  = {colorHandle};
            dbgDesc.sampledInputs = {gbufAlbedo, gbufDepth};
            dbgDesc.clearColor    = {{0.02f, 0.02f, 0.05f, 1.0f}};
            dbgDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "DebugUnlitPass");
                m_debugVisPass->recordUnlit(cmd, m_descriptorAllocator->globalSet(), ext,
                                            m_gbufAlbedoSlot, m_gbufDepthSlot,
                                            m_gbufferSamplerSlot);
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(dbgDesc));

        } else if (m_debugMode == DebugMode::Wireframe) {
            // === WIREFRAME — hardware line rasterization on black background ===
            gfx::RenderGraph::RasterPassDesc wireDesc{};
            wireDesc.name         = "DebugWireframePass";
            wireDesc.colorTargets = {colorHandle};
            wireDesc.clearColor   = {{0.0f, 0.0f, 0.0f, 1.0f}};
            wireDesc.execute      = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "DebugWireframePass");
                if (m_terrain && m_terrain->isEnabled() && m_terrainWireIndirectBuffer
                    && m_visibilityPass->hasTerrainWireframePipeline()) {
                    const u32 tmc  = m_gpuMeshlets->total_meshlet_count() - m_gpuMeshlets->scene_meshlet_count();
                    const auto topo = m_terrain->sharedTopologyHandle();
                    m_visibilityPass->recordTerrainWireframe(cmd, m_descriptorAllocator->globalSet(), ext,
                                                              *m_gpuScene, *m_gpuMeshlets, *m_terrainWireIndirectBuffer,
                                                              cameraSlot,
                                                              topo.topologyVerticesSlot,
                                                              topo.topologyTrianglesSlot,
                                                              tmc, m_wireframeColor);
                }
                m_visibilityPass->recordWireframe(cmd, m_descriptorAllocator->globalSet(), ext,
                                                  *m_gpuScene, *m_gpuMeshlets, *m_indirectBuffer,
                                                  cameraSlot, m_wireframeColor);
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(wireDesc));

        } else if (m_debugMode == DebugMode::LitWireframe) {
            // === LIT WIREFRAME — lit scene (no post-proc) + wireframe overlay ===
            // Phase 1: lighting -> hdrColorHandle
            gfx::RenderGraph::RasterPassDesc lightPassDesc{};
            lightPassDesc.name          = "LightingPass";
            lightPassDesc.colorTargets  = {hdrColorHandle};
            lightPassDesc.sampledInputs = {gbufAlbedo, gbufNormal, gbufMetalRough, gbufDepth};
            lightPassDesc.clearColor    = {{0.02f, 0.02f, 0.05f, 1.0f}};
            lightPassDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "LightingPass");
                m_lightingPass->record(cmd, m_descriptorAllocator->globalSet(), ext,
                                       m_gbufAlbedoSlot, m_gbufNormalSlot,
                                       m_gbufMetalRoughSlot, m_gbufDepthSlot,
                                       cameraSlot, m_gbufferSamplerSlot,
                                       vec4{m_sunWorldDir, m_atmosphereSettings.sunIntensity},
                                       vec4{m_light.color, 0.0f});
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(lightPassDesc));

            // Phase 1b: SkyBackgroundPass — writes the atmospheric sky into
            // HDR at pixels where depth==0.  Without this the LitWire frame
            // shows black void where the sky should be, which also defeats
            // the visual cue for aerial perspective (fog reads as a gradient
            // *into* the sky). Mirror of the pass that runs in Lit mode.
            {
                gfx::RenderGraph::RasterPassDesc skyDesc{};
                skyDesc.name          = "SkyBackgroundPass";
                skyDesc.colorTargets  = {hdrColorHandle};
                skyDesc.colorLoadOp   = VK_ATTACHMENT_LOAD_OP_LOAD;
                skyDesc.sampledInputs = {gbufDepth};
                skyDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                    m_gpuProfiler->beginZone(cmd, "SkyBackgroundPass");
                    m_skyPass->record(cmd,
                        m_descriptorAllocator->globalSet(), ext,
                        cameraSlot,
                        m_gbufDepthSlot,
                        m_atmospherePass->skyViewLutSlot(),
                        m_atmospherePass->transmittanceLutSlot(),
                        m_gbufferSamplerSlot,
                        vec4{m_sunWorldDir, m_atmosphereSettings.sunIntensity},
                        vec4{m_cameraWorldPos * 0.001f, 0.0f});
                    m_gpuProfiler->endZone(cmd);
                };
                m_renderGraph->addRasterPass(std::move(skyDesc));
            }

            // Phase 2: PostProcessPass — identical setup to Lit mode so the
            // wireframe overlay sits on top of a fully-composited beauty
            // frame (aerial perspective, bloom, tonemap, exposure).  The
            // prior DebugBlitPass only ran a Reinhard tonemap and skipped
            // atmosphere, which made LitWire look stylistically detached
            // from the pure Lit mode.
            gfx::RenderGraph::RasterPassDesc ppDesc{};
            ppDesc.name          = "PostProcessPass";
            ppDesc.colorTargets  = {colorHandle};
            ppDesc.sampledInputs = {hdrColorHandle, gbufDepth};
            ppDesc.clearColor    = {{0.0f, 0.0f, 0.0f, 1.0f}};
            ppDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "PostProcessPass");
                m_postProcessPass->record(cmd,
                    m_descriptorAllocator->globalSet(),
                    m_atmospherePass->aerialPerspectiveReadSet(),
                    ext,
                    m_hdrColorSampledSlot,
                    m_gbufDepthSlot,
                    cameraSlot,
                    m_gbufferSamplerSlot,
                    m_linearSamplerSlot,
                    m_atmosphereSettings,
                    vec4{m_cameraWorldPos * 0.001f, 0.0f});
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(ppDesc));

            // Phase 3: wireframe overlay -> swapchain (LOAD preserves lit base)
            gfx::RenderGraph::RasterPassDesc wireOverlayDesc{};
            wireOverlayDesc.name         = "DebugWireframeOverlayPass";
            wireOverlayDesc.colorTargets = {colorHandle};
            wireOverlayDesc.colorLoadOp  = VK_ATTACHMENT_LOAD_OP_LOAD;
            wireOverlayDesc.execute      = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "DebugWireframeOverlayPass");
                if (m_terrain && m_terrain->isEnabled() && m_terrainWireIndirectBuffer
                    && m_visibilityPass->hasTerrainWireframePipeline()) {
                    const u32 tmc  = m_gpuMeshlets->total_meshlet_count() - m_gpuMeshlets->scene_meshlet_count();
                    const auto topo = m_terrain->sharedTopologyHandle();
                    m_visibilityPass->recordTerrainWireframe(cmd, m_descriptorAllocator->globalSet(), ext,
                                                              *m_gpuScene, *m_gpuMeshlets, *m_terrainWireIndirectBuffer,
                                                              cameraSlot,
                                                              topo.topologyVerticesSlot,
                                                              topo.topologyTrianglesSlot,
                                                              tmc, m_wireframeColor);
                }
                m_visibilityPass->recordWireframe(cmd, m_descriptorAllocator->globalSet(), ext,
                                                  *m_gpuScene, *m_gpuMeshlets, *m_indirectBuffer,
                                                  cameraSlot, m_wireframeColor);
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(wireOverlayDesc));

        } else if (m_debugMode == DebugMode::DetailLighting) {
            // === DETAIL LIGHTING — Cook-Torrance on white material ===
            gfx::RenderGraph::RasterPassDesc dbgDesc{};
            dbgDesc.name          = "DebugDetailLightingPass";
            dbgDesc.colorTargets  = {colorHandle};
            dbgDesc.sampledInputs = {gbufAlbedo, gbufNormal, gbufMetalRough, gbufDepth};
            dbgDesc.clearColor    = {{0.02f, 0.02f, 0.05f, 1.0f}};
            dbgDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "DebugDetailLightingPass");
                m_debugVisPass->recordDetailLighting(cmd, m_descriptorAllocator->globalSet(), ext,
                                                     m_gbufAlbedoSlot, m_gbufNormalSlot,
                                                     m_gbufMetalRoughSlot, m_gbufDepthSlot,
                                                     cameraSlot, m_gbufferSamplerSlot,
                                                     vec4{m_sunWorldDir, m_atmosphereSettings.sunIntensity},
                                                     vec4{m_light.color, 0.0f});
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(dbgDesc));

        } else if (m_debugMode == DebugMode::Clusters) {
            // === CLUSTERS — meshlet color visualization ===
            gfx::RenderGraph::RasterPassDesc dbgDesc{};
            dbgDesc.name          = "DebugClustersPass";
            dbgDesc.colorTargets  = {colorHandle};
            dbgDesc.clearColor    = {{0.02f, 0.02f, 0.05f, 1.0f}};
            dbgDesc.execute       = [&](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "DebugClustersPass");
                if (m_visibilityPass)
                    m_debugVisPass->recordClusters(cmd, m_descriptorAllocator->globalSet(), ext,
                                                   m_visibilityPass->vis_buffer_slot(),
                                                   m_gbufferSamplerSlot);
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(dbgDesc));

        } else if (m_debugMode == DebugMode::MicropolyRasterClass) {
            // === M4.6: MICROPOLY RASTER CLASS — decoded rasterClassBits overlay ===
            // Samples the 64-bit Micropoly vis image and writes per-pixel
            // red (HW) / green (SW) / black (empty). The vis image stays in
            // GENERAL layout (see MicropolyPass::createVisImage) so no layout
            // transition is needed — but the upstream raster passes terminate
            // their barrier chains at COMPUTE_SHADER (HW raster FRAGMENT→COMPUTE,
            // SW raster COMPUTE→COMPUTE). This debug read is in FRAGMENT_SHADER,
            // so we emit an explicit COMPUTE→FRAGMENT memory barrier inside the
            // execute lambda to make the prior compute writes visible.
            const u32 vis64Slot =
                m_micropolyPass ? m_micropolyPass->visBindlessSlot() : UINT32_MAX;
            gfx::RenderGraph::RasterPassDesc dbgDesc{};
            // Emit vis-image COMPUTE->FRAGMENT barrier as a preceding
            // no-attachment pass. Must run OUTSIDE the dynamic-rendering
            // instance the debug raster pass opens.
            addMicropolyVisBarrierPass(*m_renderGraph);

            dbgDesc.name          = "DebugMicropolyRasterClassPass";
            dbgDesc.colorTargets  = {colorHandle};
            dbgDesc.clearColor    = {{0.0f, 0.0f, 0.0f, 1.0f}};
            dbgDesc.execute       = [this, vis64Slot](VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "DebugMicropolyRasterClassPass");
                if (vis64Slot != UINT32_MAX) {
                    m_debugVisPass->recordMicropolyRasterClass(
                        cmd, m_descriptorAllocator->globalSet(), ext, vis64Slot);
                }
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(dbgDesc));

        } else if (m_debugMode == DebugMode::MicropolyLodHeatmap) {
            // === M6.1: MICROPOLY LOD HEATMAP — cluster → DAG → lodLevel ===
            // Same vis-image barrier pattern as MicropolyRasterClass (M4.6
            // Phase 4 fix). Additional SSBOs read: the DAG buffer (not yet
            // wired; UINT32_MAX falls through to magenta on the shader side).
            const u32 vis64Slot =
                m_micropolyPass ? m_micropolyPass->visBindlessSlot() : UINT32_MAX;
            // M3.3-deferred: DAG buffer bindless slot now wired through
            // MicropolyStreaming's DEVICE_LOCAL SSBO. UINT32_MAX falls back
            // to the shader's defensive magenta paint when micropoly is
            // disabled or asset-load failed.
            const u32 dagSlot       = m_micropolyStreaming
                ? m_micropolyStreaming->dagNodeBufferBindless() : UINT32_MAX;
            const u32 dagNodeCount  =
                m_micropolyReader ? m_micropolyReader->header().dagNodeCount : 0u;
            gfx::RenderGraph::RasterPassDesc dbgDesc{};
            addMicropolyVisBarrierPass(*m_renderGraph);
            dbgDesc.name          = "DebugMicropolyLodHeatmapPass";
            dbgDesc.colorTargets  = {colorHandle};
            dbgDesc.clearColor    = {{0.0f, 0.0f, 0.0f, 1.0f}};
            dbgDesc.execute       = [this, vis64Slot, dagSlot, dagNodeCount]
                                    (VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "DebugMicropolyLodHeatmapPass");
                if (vis64Slot != UINT32_MAX) {

                    m_debugVisPass->recordMicropolyLodHeatmap(
                        cmd, m_descriptorAllocator->globalSet(), ext,
                        vis64Slot, dagSlot, dagNodeCount);
                }
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(dbgDesc));

        } else if (m_debugMode == DebugMode::MicropolyResidencyHeatmap) {
            // === M6.1: MICROPOLY RESIDENCY HEATMAP — cluster → pageId → slot ===
            // Reads vis + DAG + pageToSlot. Under steady-state operation any
            // non-resident fragment should be rare (cull-pass gates geometry
            // on residency); magenta pixels indicate eviction racing raster
            // or a DAG/pageCount bound mismatch.
            const u32 vis64Slot =
                m_micropolyPass ? m_micropolyPass->visBindlessSlot() : UINT32_MAX;
            // M3.3-deferred: DAG buffer bindless slot wired from streaming.
            const u32 dagSlot        = m_micropolyStreaming
                ? m_micropolyStreaming->dagNodeBufferBindless() : UINT32_MAX;
            const u32 pageToSlotSlot = m_micropolyStreaming
                ? m_micropolyStreaming->pageToSlotBufferBindless() : UINT32_MAX;
            const u32 dagNodeCount   =
                m_micropolyReader ? m_micropolyReader->header().dagNodeCount : 0u;
            const u32 pageCount      = m_micropolyStreaming
                ? m_micropolyStreaming->pageToSlotPageCount() : 0u;
            gfx::RenderGraph::RasterPassDesc dbgDesc{};
            addMicropolyVisBarrierPass(*m_renderGraph);
            dbgDesc.name          = "DebugMicropolyResidencyHeatmapPass";
            dbgDesc.colorTargets  = {colorHandle};
            dbgDesc.clearColor    = {{0.0f, 0.0f, 0.0f, 1.0f}};
            dbgDesc.execute       = [this, vis64Slot, dagSlot, pageToSlotSlot,
                                     dagNodeCount, pageCount]
                                    (VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "DebugMicropolyResidencyHeatmapPass");
                if (vis64Slot != UINT32_MAX) {

                    m_debugVisPass->recordMicropolyResidencyHeatmap(
                        cmd, m_descriptorAllocator->globalSet(), ext,
                        vis64Slot, dagSlot, pageToSlotSlot,
                        dagNodeCount, pageCount);
                }
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(dbgDesc));

        } else if (m_debugMode == DebugMode::MicropolyBounds) {
            // === M6.2b: MICROPOLY BOUNDS — per-cluster bounding-sphere wireframe ===
            // Iterates the DAG and projects each cluster's bounding sphere
            // to screen space, tinting pixels near the projected outline.
            // Reads only the DAG SSBO (written once at asset load) + the
            // camera buffer (uploaded earlier in drawFrame); no COMPUTE→
            // FRAGMENT barrier is needed — the DAG is not touched by any
            // earlier compute pass this frame.
            //
            // Cost: O(pixels × clusters), capped at 4096 clusters on the
            // shader side. Acceptable for debug visualisation; not
            // production-viable.
            const u32 dagSlot       = m_micropolyStreaming
                ? m_micropolyStreaming->dagNodeBufferBindless() : UINT32_MAX;
            const u32 dagNodeCount  =
                m_micropolyReader ? m_micropolyReader->header().dagNodeCount : 0u;
            const u32 screenW       = extent.width;
            const u32 screenH       = extent.height;
            gfx::RenderGraph::RasterPassDesc dbgDesc{};
            dbgDesc.name          = "DebugMicropolyBoundsPass";
            dbgDesc.colorTargets  = {colorHandle};
            dbgDesc.clearColor    = {{0.0f, 0.0f, 0.0f, 1.0f}};
            dbgDesc.execute       = [this, dagSlot, dagNodeCount,
                                     cameraSlot, screenW, screenH]
                                    (VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "DebugMicropolyBoundsPass");
                if (dagSlot != UINT32_MAX && dagNodeCount != 0u) {
                    m_debugVisPass->recordMicropolyBounds(
                        cmd, m_descriptorAllocator->globalSet(), ext,
                        dagSlot, dagNodeCount, cameraSlot,
                        screenW, screenH);
                }
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(dbgDesc));

        } else if (m_debugMode == DebugMode::MicropolyBinOverflowHeat) {
            // === M6 plan §3.M6: MICROPOLY BIN OVERFLOW HEAT ===
            // Per-pixel colour by the SW raster tile bin fill level. Reads
            // the tileBinCount + spillBuffer SSBOs that MicropolySwRasterPass
            // writes during the SW binning compute dispatch. Availability
            // gate is the pass itself being constructed (mpBinOverflowAvail)
            // — HW-only builds and non-micropoly capability rows skip this
            // branch entirely. Does NOT read the 64-bit vis image, so only
            // the SSBO barrier is needed.
            //
            // The upstream SW raster pass terminates its barrier chain at
            // COMPUTE_SHADER (its own COMPUTE->COMPUTE barrier on the bin
            // SSBOs). This debug read is in FRAGMENT_SHADER, so we reuse
            // the existing emitMicropolyVisComputeToFragmentBarrier helper —
            // despite its name it emits a generic memory barrier on all
            // shader-storage reads/writes, which is what we need here.
            const u32 tileBinCountSlot = m_micropolySwRasterPass
                ? m_micropolySwRasterPass->tileBinCountBindlessSlot() : UINT32_MAX;
            const u32 spillBufferSlot  = m_micropolySwRasterPass
                ? m_micropolySwRasterPass->spillBufferBindlessSlot()  : UINT32_MAX;
            const u32 tilesX           = m_micropolySwRasterPass
                ? m_micropolySwRasterPass->tilesX() : 0u;
            const u32 tilesY           = m_micropolySwRasterPass
                ? m_micropolySwRasterPass->tilesY() : 0u;
            const u32 screenW          = extent.width;
            const u32 screenH          = extent.height;
            gfx::RenderGraph::RasterPassDesc dbgDesc{};
            addMicropolyVisBarrierPass(*m_renderGraph);
            dbgDesc.name          = "DebugMicropolyBinOverflowPass";
            dbgDesc.colorTargets  = {colorHandle};
            dbgDesc.clearColor    = {{0.0f, 0.0f, 0.0f, 1.0f}};
            dbgDesc.execute       = [this, tileBinCountSlot, spillBufferSlot,
                                     tilesX, tilesY, screenW, screenH]
                                    (VkCommandBuffer cmd, VkExtent2D ext) {
                m_gpuProfiler->beginZone(cmd, "DebugMicropolyBinOverflowPass");
                if (tileBinCountSlot != UINT32_MAX
                    && spillBufferSlot != UINT32_MAX
                    && tilesX != 0u && tilesY != 0u) {
                    // COMPUTE->FRAGMENT barrier on the SW bin SSBOs (generic
                    // memory barrier; the helper name is vis-centric for
                    // historical reasons but the barrier it emits is
                    // storage-buffer-wide).

                    m_debugVisPass->recordMicropolyBinOverflow(
                        cmd, m_descriptorAllocator->globalSet(), ext,
                        tileBinCountSlot, spillBufferSlot,
                        tilesX, tilesY, screenW, screenH);
                }
                m_gpuProfiler->endZone(cmd);
            };
            m_renderGraph->addRasterPass(std::move(dbgDesc));
        }

    } else {
        // ---- Forward fallback: TrianglePass ----
        const auto depthHandle = m_renderGraph->importImage(
            "swapchain_depth",
            depthImage, depthView, m_swapchain->depthFormat(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_ASPECT_DEPTH_BIT);

        gfx::RenderGraph::RasterPassDesc triPassDesc{};
        triPassDesc.name         = "TrianglePass";
        triPassDesc.colorTargets = {colorHandle};
        triPassDesc.depthTarget  = depthHandle;
        triPassDesc.clearColor   = {{0.02f, 0.02f, 0.05f, 1.0f}};
        triPassDesc.clearDepth   = {0.0f, 0};
        triPassDesc.execute      = [&](VkCommandBuffer cmd, VkExtent2D ext) {
            m_gpuProfiler->beginZone(cmd, "TrianglePass");
            m_trianglePass->record(cmd, m_descriptorAllocator->globalSet(), ext);
            m_gpuProfiler->endZone(cmd);
        };
        m_renderGraph->addRasterPass(std::move(triPassDesc));
    }

    m_renderGraph->execute(frame.commandBuffer, extent);

    // Upscaler evaluate — after lighting pass, before present.
    // Compute Halton(2,3) jitter for the current frame.
    ++m_jitterIndex;
    const f32 jx = haltonJitter(m_jitterIndex, 2) - 0.5f;
    const f32 jy = haltonJitter(m_jitterIndex, 3) - 0.5f;

    // Track the swapchain image layout so ImGuiLayer can emit the correct
    // pre-barrier. The render graph leaves it in PRESENT_SRC_KHR; the
    // upscaler (when real) writes to it with outputLayout COLOR_ATTACHMENT_OPTIMAL.
    VkImageLayout swapchainLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    // PostProcessPass writes the final tonemapped image directly to the swapchain,
    // so the upscaler is skipped when it is active. The upscaler can be
    // re-integrated in a future pass once the HDR → TAA → tonemap ordering is
    // designed with a proper intermediate render target.
    if (m_upscaler && m_scene != nullptr && !m_postProcessPass) {
        m_upscaler->evaluate(frame.commandBuffer,
                             m_hdrColorView,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             depthView,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             m_gbufMotionVec.view,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             m_swapchain->view(imageIndex),
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             extent, extent,
                             jx, jy,
                             m_upscalerSettings.sharpness,
                             false);
    }

    // ImGui render -- after scene, before end command buffer.
    if (m_imguiLayer) {
        m_imguiLayer->render(frame.commandBuffer,
                             targetImage,
                             targetView,
                             extent,
                             swapchainLayout);
    }

    ENIGMA_VK_CHECK(vkEndCommandBuffer(frame.commandBuffer));

    // Submit.
    const u64 signalValue = frame.frameValue + 1;
    const VkSemaphore imageRenderFinished = m_swapchain->renderFinished(imageIndex);

    // M2.4a: when micropoly streaming submitted a transfer upload this frame,
    // the graphics queue must wait on the timeline-semaphore value returned
    // by uploadCounter() before dispatching any shader that reads the page
    // cache. Because the current wait dstStageMask only covers color-output
    // (imageAvailable), extending the wait list to include the streaming
    // upload at COMPUTE_SHADER/TASK_SHADER/MESH_SHADER is the safe choice.
    //
    // When streaming is off OR when no upload happened this frame
    // (uploadCounter==0), we don't add the wait — the semaphore value would
    // still be 0 and waiting on it is a no-op, but skipping keeps the
    // pre-M2.4a submit shape on the bit-identical-when-disabled path.
    VkSemaphore uploadWaitSem = VK_NULL_HANDLE;
    u64         uploadWaitVal = 0ull;
    if (m_micropolyStreaming &&
        m_micropolyStreaming->uploadDoneSemaphore() != VK_NULL_HANDLE &&
        m_micropolyStreaming->uploadCounter() > 0ull) {
        uploadWaitSem = m_micropolyStreaming->uploadDoneSemaphore();
        uploadWaitVal = m_micropolyStreaming->uploadCounter();
    }

    // Array size [3] (forward-compat for M3's future semaphore; today
    // only indices [0..1] are used — Phase-4 CR MINOR / Security LOW fix).
    VkSemaphore                waitSemsA[3];
    VkPipelineStageFlags       waitStagesA[3];
    u64                        waitValuesA[3] = {0ull, 0ull, 0ull};
    u32                        waitCount = 0u;
    waitSemsA[waitCount]    = frame.imageAvailable;
    waitStagesA[waitCount]  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    waitValuesA[waitCount]  = 0ull;
    ++waitCount;
    if (uploadWaitSem != VK_NULL_HANDLE) {
        // Phase-4 CR MEDIUM: TASK/MESH stage bits require VK_EXT_mesh_shader
        // to have been enabled. On devices that lack mesh-shader support,
        // waiting on those stages is a validation error — fall back to
        // COMPUTE_SHADER only (the uploaded pages still feed cull/material
        // eval via compute).
        VkPipelineStageFlags uploadStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        if (m_device->supportsMeshShaders()) {
            uploadStage |= VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT
                        |  VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT;
        }
        waitSemsA[waitCount]   = uploadWaitSem;
        waitStagesA[waitCount] = uploadStage;
        waitValuesA[waitCount] = uploadWaitVal;
        ++waitCount;
    }

    const VkSemaphore        signalSems[]  = { imageRenderFinished, frame.inFlight };
    const u64                signalValues[] = { 0, signalValue };

    VkTimelineSemaphoreSubmitInfo timelineInfo{};
    timelineInfo.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo.waitSemaphoreValueCount   = waitCount;
    timelineInfo.pWaitSemaphoreValues      = waitValuesA;
    timelineInfo.signalSemaphoreValueCount = 2;
    timelineInfo.pSignalSemaphoreValues    = signalValues;

    VkSubmitInfo submitInfo{};
    submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext                = &timelineInfo;
    submitInfo.waitSemaphoreCount   = waitCount;
    submitInfo.pWaitSemaphores      = waitSemsA;
    submitInfo.pWaitDstStageMask    = waitStagesA;
    submitInfo.commandBufferCount   = 1;
    submitInfo.pCommandBuffers      = &frame.commandBuffer;
    submitInfo.signalSemaphoreCount = 2;
    submitInfo.pSignalSemaphores    = signalSems;

    ENIGMA_VK_CHECK(vkQueueSubmit(m_device->graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE));
    frame.frameValue = signalValue;

    // Present.
    VkSwapchainKHR swapchain = m_swapchain->handle();

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores    = &imageRenderFinished;
    presentInfo.swapchainCount     = 1;
    presentInfo.pSwapchains        = &swapchain;
    presentInfo.pImageIndices      = &imageIndex;

    const VkResult presentResult = vkQueuePresentKHR(m_device->graphicsQueue(), &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
        const auto [w, h] = m_window.framebufferSize();
        m_swapchain->recreate(w, h);
        resizeGBuffer(m_swapchain->extent());
    } else if (presentResult != VK_SUCCESS) {
        ENIGMA_VK_CHECK(presentResult);
    }

    m_frameIndex = (m_frameIndex + 1) % gfx::MAX_FRAMES_IN_FLIGHT;
}

void Renderer::setScene(Scene* scene) {
    m_scene = scene;
    if (m_scene != nullptr && m_device->gpuTier() >= gfx::GpuTier::Recommended) {
        buildAccelerationStructures();
    }
    if (m_scene != nullptr && m_gpuMeshlets) {
        uploadMeshlets();
    }
}

void Renderer::buildAccelerationStructures() {
    ENIGMA_ASSERT(m_scene != nullptr);

    // Vertex stride: packed as 3 x float4 per vertex (see packVertices in GltfLoader).
    constexpr VkDeviceSize kVertexStride = 3 * sizeof(vec4); // 48 bytes

    for (auto& prim : m_scene->primitives) {
        if (prim.blas.has_value()) continue;
        if (prim.vertexBuffer == VK_NULL_HANDLE) continue;

        prim.blas = gfx::BLAS::build(*m_device, *m_allocator,
                                     prim.vertexBuffer, prim.vertexCount, kVertexStride,
                                     prim.indexBuffer, prim.indexCount);
    }

    // Build TLAS from all scene nodes.
    m_scene->tlas.emplace(*m_device, *m_allocator, 4096);

    for (const auto& node : m_scene->nodes) {
        for (u32 primIdx : node.primitiveIndices) {
            auto& prim = m_scene->primitives[primIdx];
            if (!prim.blas.has_value()) continue;

            const u32 slot = m_scene->tlas->allocateInstanceSlot();

            VkAccelerationStructureInstanceKHR inst{};
            // Copy the 3x4 transform matrix (row-major, column-major GLM needs transpose).
            const mat4& m = node.worldTransform;
            inst.transform.matrix[0][0] = m[0][0]; inst.transform.matrix[0][1] = m[1][0];
            inst.transform.matrix[0][2] = m[2][0]; inst.transform.matrix[0][3] = m[3][0];
            inst.transform.matrix[1][0] = m[0][1]; inst.transform.matrix[1][1] = m[1][1];
            inst.transform.matrix[1][2] = m[2][1]; inst.transform.matrix[1][3] = m[3][1];
            inst.transform.matrix[2][0] = m[0][2]; inst.transform.matrix[2][1] = m[1][2];
            inst.transform.matrix[2][2] = m[2][2]; inst.transform.matrix[2][3] = m[3][2];

            inst.instanceCustomIndex                    = primIdx;
            inst.mask                                   = 0xFF;
            inst.instanceShaderBindingTableRecordOffset  = 0;
            inst.flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            inst.accelerationStructureReference         = prim.blas->deviceAddress();

            m_scene->tlas->setInstance(slot, inst);
        }
    }

    // M5a: merge micropoly per-asset-proxy BLAS instances into the TLAS
    // build input. Non-null manager implies RT-capable device + a loaded
    // .mpa; instances() is empty when buildForAsset either wasn't called
    // or returned NoLevel3Clusters. shadow.rgen.hlsl traces the TLAS
    // as-is and picks up these instances automatically — no shader
    // changes required for M5a (micropoly content is opaque in v1).
    if (m_micropolyBlasManager != nullptr) {
        const auto mpInstances = m_micropolyBlasManager->instances();
        for (const auto& mpi : mpInstances) {
            const u32 slot = m_scene->tlas->allocateInstanceSlot();

            VkAccelerationStructureInstanceKHR inst{};
            // 3x4 transpose from GLM column-major to Vulkan row-major,
            // same pattern as the non-micropoly path above.
            const mat4& m = mpi.transform;
            inst.transform.matrix[0][0] = m[0][0]; inst.transform.matrix[0][1] = m[1][0];
            inst.transform.matrix[0][2] = m[2][0]; inst.transform.matrix[0][3] = m[3][0];
            inst.transform.matrix[1][0] = m[0][1]; inst.transform.matrix[1][1] = m[1][1];
            inst.transform.matrix[1][2] = m[2][1]; inst.transform.matrix[1][3] = m[3][1];
            inst.transform.matrix[2][0] = m[0][2]; inst.transform.matrix[2][1] = m[1][2];
            inst.transform.matrix[2][2] = m[2][2]; inst.transform.matrix[2][3] = m[3][2];

            inst.instanceCustomIndex                    = mpi.customIndex;
            inst.mask                                   = mpi.mask;
            inst.instanceShaderBindingTableRecordOffset  = 0;
            // v1: opaque content only (plan §3.M5a line 509 — alpha-tested
            // any-hit deferred). Keep triangle-facing cull ON so back faces
            // are skipped; matches the meshlet-path rasterizer semantics.
            inst.flags                                  = 0;
            inst.accelerationStructureReference         = mpi.blasAddress;

            m_scene->tlas->setInstance(slot, inst);
        }
        if (!mpInstances.empty()) {
            ENIGMA_LOG_INFO(
                "[renderer] merged {} micropoly BLAS instance(s) into TLAS",
                mpInstances.size());
        }
    }

    // Build TLAS on the GPU via immediate command buffer.
    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolCI.queueFamilyIndex = m_device->graphicsQueueFamily();

    VkCommandPool cmdPool = VK_NULL_HANDLE;
    ENIGMA_VK_CHECK(vkCreateCommandPool(m_device->logical(), &poolCI, nullptr, &cmdPool));

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool        = cmdPool;
    cmdAllocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    ENIGMA_VK_CHECK(vkAllocateCommandBuffers(m_device->logical(), &cmdAllocInfo, &cmd));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ENIGMA_VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    m_scene->tlas->build(cmd, *m_device, *m_allocator);

    ENIGMA_VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;
    ENIGMA_VK_CHECK(vkQueueSubmit(m_device->graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE));
    ENIGMA_VK_CHECK(vkQueueWaitIdle(m_device->graphicsQueue()));

    vkDestroyCommandPool(m_device->logical(), cmdPool, nullptr);

    // Register TLAS in the bindless descriptor set.
    m_tlasSlot = m_descriptorAllocator->registerAccelerationStructure(m_scene->tlas->handle());

    ENIGMA_LOG_INFO("[renderer] acceleration structures built ({} BLASes, TLAS slot={})",
                    m_scene->primitives.size(), m_tlasSlot);
}

void Renderer::uploadMeshlets() {
    ENIGMA_ASSERT(m_scene != nullptr);
    ENIGMA_ASSERT(m_gpuMeshlets != nullptr);

    // Append scene meshlets into the CPU-side accumulation. These must be
    // committed BEFORE reserveCapacity() is called so the reserved GPU buffer
    // is sized large enough to cover scene meshlets + terrain activation
    // ceiling (see GpuMeshletBuffer::reserveCapacity contract).
    bool anyMeshlets = false;
    for (auto& prim : m_scene->primitives) {
        if (prim.meshlets.meshlets.empty()) continue;
        prim.meshletOffset = m_gpuMeshlets->append(prim.meshlets);
        anyMeshlets = true;
    }

    // Even with no mesh primitives we still need the one-shot command buffer
    // for heightmap upload + CDLOD initialize.
    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolCI.queueFamilyIndex = m_device->graphicsQueueFamily();

    VkCommandPool cmdPool = VK_NULL_HANDLE;
    ENIGMA_VK_CHECK(vkCreateCommandPool(m_device->logical(), &poolCI, nullptr, &cmdPool));

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool        = cmdPool;
    cmdAllocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    ENIGMA_VK_CHECK(vkAllocateCommandBuffers(m_device->logical(), &cmdAllocInfo, &cmd));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ENIGMA_VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    // Load heightmap (record staging->image copy + layout transitions).
    m_heightmapLoader = std::make_unique<HeightmapLoader>(
        *m_device, *m_allocator, *m_descriptorAllocator);
    HeightmapDesc hDesc{};
    hDesc.path        = "assets/terrain/height_4k.r32f";
    hDesc.sampleCount = 4097;
    hDesc.worldSize   = 4096.0f;
    hDesc.minHeight   = 0.0f;
    hDesc.maxHeight   = 512.0f;
    (void)m_heightmapLoader->load(hDesc, cmd); // tolerate missing file — zeros

    // Initialize CDLOD terrain. initialize() internally calls
    // GpuMeshletBuffer::reserveCapacity() which uploads the current
    // CPU-side meshlet array to the device-local buffer and pre-reserves
    // room for per-patch incremental appends. This replaces the old
    // m_gpuMeshlets->upload() call on the meshlet-only path.
    m_terrain = std::make_unique<CdlodTerrain>(
        *m_device, *m_allocator, *m_descriptorAllocator);
    CdlodConfig terrainConfig{};
    m_terrain->initialize(terrainConfig, *m_heightmapLoader,
                          *m_gpuMeshlets, *m_indirectBuffer, *m_scene,
                          cmd);

    // Upload the precomputed AreaTex + SearchTex lookup textures that the
    // reference SMAA blend pass consumes. Records on the same init cmd; the
    // waitIdle below guarantees staging buffers are safe to destroy.
    if (m_smaaPass) {
        m_smaaPass->uploadLookupTextures(cmd, *m_descriptorAllocator);
    }

    ENIGMA_VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;
    ENIGMA_VK_CHECK(vkQueueSubmit(m_device->graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE));
    ENIGMA_VK_CHECK(vkQueueWaitIdle(m_device->graphicsQueue()));

    // Release heightmap staging + meshlet staging + SMAA lookup-texture
    // staging now that the GPU has finished consuming them.
    m_heightmapLoader->releaseStaging();
    m_gpuMeshlets->flush_staging();
    if (m_smaaPass) m_smaaPass->releaseLookupUploadStaging();

    vkDestroyCommandPool(m_device->logical(), cmdPool, nullptr);

    // Rebuild material SSBO to include the terrain material appended by
    // CdlodTerrain::initialize() — the buffer built by loadGltf was sized
    // before terrain init, so the terrain slot was absent.
    if (m_terrain && m_terrain->isEnabled() && m_scene->materialBufferSlot != 0xFFFFFFFFu) {
        const VkDeviceSize matBufSize = m_scene->materials.size() * sizeof(Material);

        VkBufferCreateInfo bufCI{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufCI.size        = matBufSize;
        bufCI.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo devAllocCI{};
        devAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        VkBuffer      newBuf   = VK_NULL_HANDLE;
        VmaAllocation newAlloc = nullptr;
        ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &bufCI, &devAllocCI,
                                        &newBuf, &newAlloc, nullptr));

        VkBufferCreateInfo stagCI{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        stagCI.size        = matBufSize;
        stagCI.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        stagCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo stagAllocCI{};
        stagAllocCI.usage  = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        stagAllocCI.flags  = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                             VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        VkBuffer          stagBuf   = VK_NULL_HANDLE;
        VmaAllocation     stagAlloc = nullptr;
        VmaAllocationInfo stagInfo{};
        ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &stagCI, &stagAllocCI,
                                        &stagBuf, &stagAlloc, &stagInfo));
        std::memcpy(stagInfo.pMappedData, m_scene->materials.data(), matBufSize);
        vmaFlushAllocation(m_allocator->handle(), stagAlloc, 0, VK_WHOLE_SIZE);

        VkCommandPoolCreateInfo pool2CI{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool2CI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pool2CI.queueFamilyIndex = m_device->graphicsQueueFamily();
        VkCommandPool pool2 = VK_NULL_HANDLE;
        ENIGMA_VK_CHECK(vkCreateCommandPool(m_device->logical(), &pool2CI, nullptr, &pool2));
        VkCommandBufferAllocateInfo cb2AI{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cb2AI.commandPool        = pool2;
        cb2AI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cb2AI.commandBufferCount = 1;
        VkCommandBuffer cmd2 = VK_NULL_HANDLE;
        ENIGMA_VK_CHECK(vkAllocateCommandBuffers(m_device->logical(), &cb2AI, &cmd2));
        VkCommandBufferBeginInfo cb2BI{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        cb2BI.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd2, &cb2BI);
        VkBufferCopy region{0, 0, matBufSize};
        vkCmdCopyBuffer(cmd2, stagBuf, newBuf, 1, &region);
        vkEndCommandBuffer(cmd2);
        VkSubmitInfo sub2{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        sub2.commandBufferCount = 1;
        sub2.pCommandBuffers    = &cmd2;
        ENIGMA_VK_CHECK(vkQueueSubmit(m_device->graphicsQueue(), 1, &sub2, VK_NULL_HANDLE));
        ENIGMA_VK_CHECK(vkQueueWaitIdle(m_device->graphicsQueue()));
        vkDestroyCommandPool(m_device->logical(), pool2, nullptr);
        vmaDestroyBuffer(m_allocator->handle(), stagBuf, stagAlloc);

        vmaDestroyBuffer(m_allocator->handle(),
                         m_scene->materialBuffer.buffer,
                         m_scene->materialBuffer.allocation);
        m_scene->materialBuffer.buffer     = newBuf;
        m_scene->materialBuffer.allocation = newAlloc;
        m_scene->materialBufferSlot = m_descriptorAllocator->registerStorageBuffer(
            newBuf, matBufSize);
        ENIGMA_LOG_INFO("[renderer] material SSBO rebuilt with {} materials ({} B)",
                        m_scene->materials.size(), matBufSize);
    }

    ENIGMA_LOG_INFO("[renderer] meshlets reserved ({} scene meshlets), "
                    "cdlod terrain {} (anyMeshlets={})",
                    m_gpuMeshlets->total_meshlet_count(),
                    m_terrain->isEnabled() ? "enabled" : "disabled",
                    anyMeshlets);
}

void Renderer::applyImpact(const ImpactEvent& event) {
    m_deformationSystem.applyImpact(event);
    m_deformationPending = true;
}

void Renderer::setUpscalerSettings(const UpscalerSettings& s) {
    const auto oldBackend = m_upscalerSettings.effectiveBackend(
        m_device->properties(), m_device->gpuTier());
    m_upscalerSettings = s;
    const auto newBackend = m_upscalerSettings.effectiveBackend(
        m_device->properties(), m_device->gpuTier());

    if (newBackend != oldBackend) {
        // Backend changed: destroy old, create new.
        vkDeviceWaitIdle(m_device->logical());
        m_upscaler->shutdown();
        m_upscaler = UpscalerFactory::create(
            m_device->properties(), m_device->gpuTier(), newBackend);
        m_upscaler->init(m_device->logical(), m_device->physical(),
                         m_swapchain->extent(), m_upscalerSettings.quality);
    } else if (m_upscalerSettings.quality != s.quality) {
        // Quality only changed: reinit in place.
        vkDeviceWaitIdle(m_device->logical());
        m_upscaler->reinit(m_swapchain->extent(), m_upscalerSettings.quality);
    }
}

void Renderer::resizeGBuffer(VkExtent2D extent) {
    // createGBufferImages calls vkDeviceWaitIdle before destroying in-use images.
    createGBufferImages(extent);

    // Re-write the bindless descriptor slots to point at the new image views.
    m_descriptorAllocator->updateSampledImage(
        m_gbufAlbedoSlot,     m_gbufAlbedo.view,     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_descriptorAllocator->updateSampledImage(
        m_gbufNormalSlot,     m_gbufNormal.view,     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_descriptorAllocator->updateSampledImage(
        m_gbufMetalRoughSlot, m_gbufMetalRough.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_descriptorAllocator->updateSampledImage(
        m_gbufMotionVecSlot,  m_gbufMotionVec.view,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_descriptorAllocator->updateSampledImage(
        m_gbufDepthSlot,      m_gbufDepth.view,      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Re-allocate RT pass images on resize.
    m_rtReflectionPass->allocate(extent);
    m_descriptorAllocator->updateStorageImage(m_reflectionSlot, m_rtReflectionPass->outputView());

    m_giPass->allocate(extent);
    m_descriptorAllocator->updateStorageImage(m_giSlot, m_giPass->outputView());

    m_shadowPass->allocate(extent);
    m_descriptorAllocator->updateStorageImage(m_shadowSlot, m_shadowPass->outputView());

    m_wetRoadPass->allocate(extent);
    m_descriptorAllocator->updateStorageImage(m_wetRoadSlot, m_wetRoadPass->outputView());

    m_giDenoiser->allocate(extent, RTGIPass::kOutputFormat);
    m_descriptorAllocator->updateStorageImage(m_giDenoiseSlot, m_giDenoiser->outputView());

    m_shadowDenoiser->allocate(extent, RTShadowPass::kOutputFormat);
    m_descriptorAllocator->updateStorageImage(m_shadowDenoiseSlot, m_shadowDenoiser->outputView());

    m_reflectionDenoiser->allocate(extent, RTReflectionPass::kOutputFormat);
    m_descriptorAllocator->updateStorageImage(m_reflDenoiseSlot, m_reflectionDenoiser->outputView());

    // Update storage image slots to point at the new G-buffer views.
    m_descriptorAllocator->updateStorageImage(m_gbufAlbedoStorageSlot,     m_gbufAlbedo.view);
    m_descriptorAllocator->updateStorageImage(m_gbufNormalStorageSlot,     m_gbufNormal.view);
    m_descriptorAllocator->updateStorageImage(m_gbufMetalRoughStorageSlot, m_gbufMetalRough.view);
    m_descriptorAllocator->updateStorageImage(m_gbufMotionVecStorageSlot,  m_gbufMotionVec.view);

    // VB pipeline resize.
    if (m_hizPass)        m_hizPass->allocate(extent);
    if (m_visibilityPass) m_visibilityPass->allocate(extent, gbuf::kDepthFormat);
    if (m_materialEvalPass) {
        m_materialEvalPass->prepare(
            extent,
            m_gbufAlbedo.image,     m_gbufNormal.image,
            m_gbufMetalRough.image, m_gbufMotionVec.image,
            m_gbufDepth.image,
            m_gbufAlbedoStorageSlot,     m_gbufNormalStorageSlot,
            m_gbufMetalRoughStorageSlot, m_gbufMotionVecStorageSlot,
            m_gbufDepthSlot);
    }

    // M3.1: resize the 64-bit micropoly vis image. createVisImage is a
    // no-op when MicropolyConfig::enabled=false, and internally tears down
    // the previous allocation (vkDeviceWaitIdle was already issued by the
    // swapchain recreate path that called resizeGBuffer). The bindless
    // slot is released + re-acquired so existing pipelines that have
    // baked-in slot indices must be refreshed — no such pipelines exist
    // yet in M3.1 (M3.3/M3.4 will be the first callers).
    if (m_micropolyPass) {
        m_micropolyPass->createVisImage(
            extent, *m_allocator, *m_descriptorAllocator);
    }

    // M4.2: resize the SW raster tile-bin SSBOs. Without this the bin
    // shader writes past the end of the old tile-count-sized buffers at
    // 1080p -> 4K. resize() internally waits-idle + tears down the
    // extent-dependent buffers and re-registers their bindless slots.
    // A failure here is non-fatal — log and leave the pass in a
    // crash-safe (non-functional) state until the next resize.
    if (m_micropolySwRasterPass) {
        auto r = m_micropolySwRasterPass->resize(extent);
        if (!r.has_value()) {
            ENIGMA_LOG_ERROR(
                "[renderer] micropoly SW raster resize failed: {} ({})",
                r.error().detail,
                renderer::micropoly::micropolySwRasterErrorKindString(
                    r.error().kind));
        }
    }

    // Recreate HDR intermediate at the new extent and patch bindless slots.
    createHdrColor(extent);

    // Recreate SMAA intermediates (LDR + edge/weight textures) at the new extent.
    createSmaaLdr(extent);
    if (m_smaaPass) {
        if (m_smaaPass->isAllocated()) {
            m_smaaPass->free(*m_descriptorAllocator);
        }
        m_smaaPass->allocate(extent, *m_descriptorAllocator);
    }
}

void Renderer::createGBufferImage(VkFormat format, VkImageUsageFlags usage,
                                   VkImageAspectFlags aspect, VkExtent2D extent, GBufferImage& out) {
    VkImageCreateInfo imageCI{};
    imageCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCI.imageType     = VK_IMAGE_TYPE_2D;
    imageCI.format        = format;
    imageCI.extent        = {extent.width, extent.height, 1};
    imageCI.mipLevels     = 1;
    imageCI.arrayLayers   = 1;
    imageCI.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageCI.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage         = usage;
    imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCI.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocCI.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    ENIGMA_VK_CHECK(vmaCreateImage(m_allocator->handle(), &imageCI, &allocCI,
                                   &out.image, &out.allocation, nullptr));

    VkImageViewCreateInfo viewCI{};
    viewCI.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image                           = out.image;
    viewCI.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format                          = format;
    viewCI.subresourceRange.aspectMask     = aspect;
    viewCI.subresourceRange.baseMipLevel   = 0;
    viewCI.subresourceRange.levelCount     = 1;
    viewCI.subresourceRange.baseArrayLayer = 0;
    viewCI.subresourceRange.layerCount     = 1;

    ENIGMA_VK_CHECK(vkCreateImageView(m_device->logical(), &viewCI, nullptr, &out.view));
}

void Renderer::destroyGBufferImage(GBufferImage& img) {
    if (img.view != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device->logical(), img.view, nullptr);
        img.view = VK_NULL_HANDLE;
    }
    if (img.image != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator->handle(), img.image, img.allocation);
        img.image      = VK_NULL_HANDLE;
        img.allocation = nullptr;
    }
}

void Renderer::createGBufferImages(VkExtent2D extent) {
    if (m_gbufAlbedo.image != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device->logical());
        destroyGBufferImages();
    }
    constexpr VkImageUsageFlags kColorUsage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    createGBufferImage(gbuf::kAlbedoFormat,     kColorUsage, VK_IMAGE_ASPECT_COLOR_BIT, extent, m_gbufAlbedo);
    createGBufferImage(gbuf::kNormalFormat,     kColorUsage, VK_IMAGE_ASPECT_COLOR_BIT, extent, m_gbufNormal);
    createGBufferImage(gbuf::kMetalRoughFormat, kColorUsage, VK_IMAGE_ASPECT_COLOR_BIT, extent, m_gbufMetalRough);
    createGBufferImage(gbuf::kMotionVecFormat,  kColorUsage, VK_IMAGE_ASPECT_COLOR_BIT, extent, m_gbufMotionVec);
    createGBufferImage(gbuf::kDepthFormat,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                       VK_IMAGE_ASPECT_DEPTH_BIT, extent, m_gbufDepth);
    ENIGMA_LOG_INFO("[renderer] G-buffer allocated {}x{}", extent.width, extent.height);
}

void Renderer::destroyGBufferImages() {
    destroyGBufferImage(m_gbufAlbedo);
    destroyGBufferImage(m_gbufNormal);
    destroyGBufferImage(m_gbufMetalRough);
    destroyGBufferImage(m_gbufMotionVec);
    destroyGBufferImage(m_gbufDepth);
}

void Renderer::createHdrColor(VkExtent2D extent) {
    if (m_hdrColor != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device->logical());
        destroyHdrColor();
    }

    VkImageCreateInfo imgCI{};
    imgCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgCI.imageType     = VK_IMAGE_TYPE_2D;
    imgCI.format        = VK_FORMAT_R16G16B16A16_SFLOAT;
    imgCI.extent        = {extent.width, extent.height, 1};
    imgCI.mipLevels     = 1;
    imgCI.arrayLayers   = 1;
    imgCI.samples       = VK_SAMPLE_COUNT_1_BIT;
    imgCI.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imgCI.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                        | VK_IMAGE_USAGE_SAMPLED_BIT
                        | VK_IMAGE_USAGE_STORAGE_BIT
                        | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imgCI.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    imgCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    ENIGMA_VK_CHECK(vmaCreateImage(m_allocator->handle(), &imgCI, &allocCI,
                                   &m_hdrColor, &m_hdrColorAlloc, nullptr));

    VkImageViewCreateInfo viewCI{};
    viewCI.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image            = m_hdrColor;
    viewCI.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format           = VK_FORMAT_R16G16B16A16_SFLOAT;
    viewCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    ENIGMA_VK_CHECK(vkCreateImageView(m_device->logical(), &viewCI, nullptr, &m_hdrColorView));

    if (m_hdrColorSampledSlot == UINT32_MAX) {
        m_hdrColorSampledSlot = m_descriptorAllocator->registerSampledImage(
            m_hdrColorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    } else {
        m_descriptorAllocator->updateSampledImage(
            m_hdrColorSampledSlot, m_hdrColorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    if (m_hdrColorStorageSlot == UINT32_MAX) {
        m_hdrColorStorageSlot = m_descriptorAllocator->registerStorageImage(m_hdrColorView);
    } else {
        m_descriptorAllocator->updateStorageImage(m_hdrColorStorageSlot, m_hdrColorView);
    }

    ENIGMA_LOG_INFO("[renderer] HDR intermediate {}x{} created (sampled={}, storage={})",
                    extent.width, extent.height,
                    m_hdrColorSampledSlot, m_hdrColorStorageSlot);
}

void Renderer::destroyHdrColor() {
    if (m_hdrColorView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device->logical(), m_hdrColorView, nullptr);
        m_hdrColorView = VK_NULL_HANDLE;
    }
    if (m_hdrColor != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator->handle(), m_hdrColor, m_hdrColorAlloc);
        m_hdrColor      = VK_NULL_HANDLE;
        m_hdrColorAlloc = nullptr;
    }
}

void Renderer::createSmaaLdr(VkExtent2D extent) {
    if (m_smaaLdrImage != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device->logical());
        destroySmaaLdr();
    }

    VkImageCreateInfo imgCI{};
    imgCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgCI.imageType     = VK_IMAGE_TYPE_2D;
    imgCI.format        = m_swapchain->format();
    imgCI.extent        = {extent.width, extent.height, 1};
    imgCI.mipLevels     = 1;
    imgCI.arrayLayers   = 1;
    imgCI.samples       = VK_SAMPLE_COUNT_1_BIT;
    imgCI.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imgCI.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                        | VK_IMAGE_USAGE_SAMPLED_BIT;
    imgCI.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    imgCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    ENIGMA_VK_CHECK(vmaCreateImage(m_allocator->handle(), &imgCI, &allocCI,
                                    &m_smaaLdrImage, &m_smaaLdrAlloc, nullptr));

    VkImageViewCreateInfo viewCI{};
    viewCI.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image            = m_smaaLdrImage;
    viewCI.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format           = m_swapchain->format();
    viewCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    ENIGMA_VK_CHECK(vkCreateImageView(m_device->logical(), &viewCI, nullptr, &m_smaaLdrView));

    if (m_smaaLdrSampledSlot == UINT32_MAX) {
        m_smaaLdrSampledSlot = m_descriptorAllocator->registerSampledImage(
            m_smaaLdrView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    } else {
        m_descriptorAllocator->updateSampledImage(
            m_smaaLdrSampledSlot, m_smaaLdrView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    ENIGMA_LOG_INFO("[renderer] SMAA LDR intermediate {}x{} created (slot={})",
                    extent.width, extent.height, m_smaaLdrSampledSlot);
}

void Renderer::destroySmaaLdr() {
    if (m_smaaLdrView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device->logical(), m_smaaLdrView, nullptr);
        m_smaaLdrView = VK_NULL_HANDLE;
    }
    if (m_smaaLdrImage != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator->handle(), m_smaaLdrImage, m_smaaLdrAlloc);
        m_smaaLdrImage = VK_NULL_HANDLE;
        m_smaaLdrAlloc = nullptr;
    }
}

void Renderer::applyTextureFilterSettings() {
    if (m_scene == nullptr ||
        m_scene->defaultMaterialSamplerSlot == 0xFFFFFFFFu) {
        return;
    }

    vkDeviceWaitIdle(m_device->logical());

    const f32  devMaxAniso   = m_device->properties().limits.maxSamplerAnisotropy;
    const bool anisoSupported = devMaxAniso > 1.0f;
    const u32  requestedAniso = std::clamp<u32>(
        m_textureFilterSettings.anisotropy,
        1u,
        static_cast<u32>(std::floor(devMaxAniso)));
    m_textureFilterSettings.anisotropy = requestedAniso;

    VkSamplerCreateInfo ci{};
    ci.sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    ci.magFilter        = VK_FILTER_LINEAR;
    ci.minFilter        = VK_FILTER_LINEAR;
    ci.mipmapMode       = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    ci.addressModeU     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    ci.addressModeV     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    ci.addressModeW     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    ci.anisotropyEnable = (anisoSupported && requestedAniso > 1u) ? VK_TRUE : VK_FALSE;
    ci.maxAnisotropy    = ci.anisotropyEnable ? static_cast<f32>(requestedAniso) : 1.0f;
    ci.borderColor      = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    ci.minLod           = 0.0f;
    ci.maxLod           = m_textureFilterSettings.mipmapsEnabled ? VK_LOD_CLAMP_NONE : 0.25f;

    VkSampler newSampler = VK_NULL_HANDLE;
    ENIGMA_VK_CHECK(vkCreateSampler(m_device->logical(), &ci, nullptr, &newSampler));

    m_descriptorAllocator->updateSampler(
        m_scene->defaultMaterialSamplerSlot, newSampler);

    // Destroy only our previously-owned sampler. The very first call (m_materialSampler==null)
    // leaves the GltfLoader-created default in scene.ownedSamplers; it's no longer
    // referenced by the bindless slot but will be freed by Scene::destroy during shutdown.
    // Not destroying it here avoids double-free paths if the scene is later swapped.
    if (m_materialSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device->logical(), m_materialSampler, nullptr);
    }
    m_materialSampler = newSampler;

    ENIGMA_LOG_INFO("[renderer] material sampler rebuilt: mipmaps={}, aniso={}x",
                    m_textureFilterSettings.mipmapsEnabled ? "on" : "off",
                    requestedAniso);
}

} // namespace enigma
