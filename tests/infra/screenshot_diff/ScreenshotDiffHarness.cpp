// ScreenshotDiffHarness implementation.
//
// Headless Vulkan bring-up (no surface / swapchain). Mirrors the
// subset of src/gfx/Instance.cpp + src/gfx/Device.cpp needed for a
// single color-attachment offscreen render pass. Deliberately
// duplicates (rather than links against) those classes because they
// require GLFW and a window surface at construction — neither is
// available on headless CI.

#include "ScreenshotDiffHarness.h"

// volk provides the Vulkan function loader; same as the main engine.
#include <volk.h>

// stb_image (read PNG) + stb_image_write (emit PNG). Both are header-
// only single-file libs; we stamp out the implementations here because
// this is a standalone translation unit not shared with src/asset/
// GltfLoader.cpp (which also stamps STB_IMAGE_IMPLEMENTATION).
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

namespace enigma::test_infra {

namespace {

// Built-in scene: opaque flat color. Chosen to have non-zero channels
// in every byte position so sign-extension and row-padding bugs in
// the readback path surface in the diff rather than silently clearing
// to 0. The alpha channel is 255 — the reference PNG is encoded with
// alpha for round-trip identity.
constexpr VkClearColorValue kBuiltinClearColor = {{ 32.0f / 255.0f,
                                                    96.0f / 255.0f,
                                                    160.0f / 255.0f,
                                                    1.0f }};

// We fix the color format to UNORM8 RGBA so the readback bytes match
// the PNG RGBA8 pixel layout byte-for-byte. Requires format support
// as a color attachment + transfer src — guaranteed on every Vulkan
// 1.0 implementation per core spec table (no feature query needed).
constexpr VkFormat kColorFormat = VK_FORMAT_R8G8B8A8_UNORM;

// --- error-handling helpers -------------------------------------------------

// Fatal guard: on any Vulkan failure we emit to stderr and return a
// sentinel Vulkan-less result. We never abort() — the harness must
// report a structured failure so the test runner can surface the
// message cleanly.
#define VK_CHECK_RETURN(expr, returnValue)                                    \
    do {                                                                       \
        const VkResult _vr = (expr);                                           \
        if (_vr != VK_SUCCESS) {                                               \
            std::fprintf(stderr,                                               \
                "[ScreenshotDiffHarness] %s failed: VkResult=%d\n",            \
                #expr, static_cast<int>(_vr));                                 \
            return returnValue;                                                \
        }                                                                      \
    } while (0)

// Common result helper: populate a failed ScreenshotDiffResult with a
// human-readable message, leaving diff counters zeroed.
ScreenshotDiffResult makeFailure(std::string message) {
    ScreenshotDiffResult r;
    r.passed  = false;
    r.message = std::move(message);
    return r;
}

// --- headless Vulkan device bundle ------------------------------------------

// RAII bundle: Vulkan instance, physical device, logical device, and
// a graphics queue. No surface, no swapchain, no validation layers
// (CI doesn't ship them; dev machines can flip VK_LAYER_PATH if
// needed). volk is initialized once lazily.
struct VulkanBundle {
    VkInstance       instance        = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice  = VK_NULL_HANDLE;
    VkDevice         device          = VK_NULL_HANDLE;
    VkQueue          graphicsQueue   = VK_NULL_HANDLE;
    std::uint32_t    graphicsFamily  = 0;
    VkCommandPool    commandPool     = VK_NULL_HANDLE;

    ~VulkanBundle() {
        if (device != VK_NULL_HANDLE) {
            if (commandPool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, commandPool, nullptr);
            }
            vkDestroyDevice(device, nullptr);
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
        }
    }
};

bool initVolk() {
    static bool initialized = false;
    if (initialized) return true;
    const VkResult r = volkInitialize();
    if (r != VK_SUCCESS) {
        std::fprintf(stderr,
            "[ScreenshotDiffHarness] volkInitialize failed: VkResult=%d "
            "(is the Vulkan loader installed?)\n", static_cast<int>(r));
        return false;
    }
    initialized = true;
    return true;
}

// Select any physical device that exposes a graphics-capable queue.
// We prefer discrete GPUs when present — matches the production
// Device.cpp heuristic — but fall back to integrated / CPU for CI.
VkPhysicalDevice pickPhysicalDevice(VkInstance instance, std::uint32_t& outGraphicsFamily) {
    // code-reviewer MAJOR-1 fix: vkEnumeratePhysicalDevices returns
    // VkResult. On VK_INCOMPLETE / VK_ERROR_OUT_OF_HOST_MEMORY the
    // returned count is unreliable; bail out and let the caller treat
    // "no physical device" as a hard failure.
    std::uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS) {
        std::fprintf(stderr,
            "[ScreenshotDiffHarness] vkEnumeratePhysicalDevices (count) failed\n");
        return VK_NULL_HANDLE;
    }
    if (count == 0) return VK_NULL_HANDLE;
    std::vector<VkPhysicalDevice> devices(count);
    if (vkEnumeratePhysicalDevices(instance, &count, devices.data()) != VK_SUCCESS) {
        std::fprintf(stderr,
            "[ScreenshotDiffHarness] vkEnumeratePhysicalDevices (enumerate) failed\n");
        return VK_NULL_HANDLE;
    }

    VkPhysicalDevice fallback = VK_NULL_HANDLE;
    std::uint32_t    fallbackFamily = 0;

    for (VkPhysicalDevice pd : devices) {
        std::uint32_t qfCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, qf.data());

        for (std::uint32_t i = 0; i < qfCount; ++i) {
            if ((qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) continue;

            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(pd, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                outGraphicsFamily = i;
                return pd;
            }
            if (fallback == VK_NULL_HANDLE) {
                fallback = pd;
                fallbackFamily = i;
            }
            break;
        }
    }
    outGraphicsFamily = fallbackFamily;
    return fallback;
}

bool createVulkanBundle(VulkanBundle& out) {
    if (!initVolk()) return false;

    // --- instance ---
    VkApplicationInfo appInfo{};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName   = "enigma-screenshot-diff-harness";
    appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    appInfo.pEngineName        = "Enigma";
    appInfo.engineVersion      = VK_MAKE_API_VERSION(0, 0, 1, 0);
    // Vulkan 1.3 matches the engine's target (find_package(Vulkan 1.3)
    // in CMakeLists.txt). Needed for core features like dynamic rendering,
    // sync2 — though the harness itself uses only 1.0 core.
    appInfo.apiVersion         = VK_API_VERSION_1_3;

    VkInstanceCreateInfo instCI{};
    instCI.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instCI.pApplicationInfo = &appInfo;
    VK_CHECK_RETURN(vkCreateInstance(&instCI, nullptr, &out.instance), false);
    volkLoadInstance(out.instance);

    // --- physical device ---
    out.physicalDevice = pickPhysicalDevice(out.instance, out.graphicsFamily);
    if (out.physicalDevice == VK_NULL_HANDLE) {
        std::fprintf(stderr,
            "[ScreenshotDiffHarness] no physical device with a "
            "graphics queue family found\n");
        return false;
    }

    // --- logical device + queue ---
    const float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo qCI{};
    qCI.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qCI.queueFamilyIndex = out.graphicsFamily;
    qCI.queueCount       = 1;
    qCI.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo devCI{};
    devCI.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devCI.queueCreateInfoCount = 1;
    devCI.pQueueCreateInfos    = &qCI;

    VK_CHECK_RETURN(vkCreateDevice(out.physicalDevice, &devCI, nullptr, &out.device), false);
    volkLoadDevice(out.device);
    vkGetDeviceQueue(out.device, out.graphicsFamily, 0, &out.graphicsQueue);

    // --- command pool ---
    VkCommandPoolCreateInfo cpCI{};
    cpCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpCI.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpCI.queueFamilyIndex = out.graphicsFamily;
    VK_CHECK_RETURN(vkCreateCommandPool(out.device, &cpCI, nullptr, &out.commandPool), false);

    return true;
}

// --- memory helper: find a host-visible+coherent memory type --------------

std::uint32_t findMemoryType(VkPhysicalDevice pd, std::uint32_t typeBits, VkMemoryPropertyFlags required) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(pd, &props);
    for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        const bool typeAllowed = (typeBits & (1u << i)) != 0;
        const bool hasRequired = (props.memoryTypes[i].propertyFlags & required) == required;
        if (typeAllowed && hasRequired) return i;
    }
    return UINT32_MAX;
}

// --- capture path: clear-color render + readback --------------------------

struct CapturedImage {
    std::vector<std::uint8_t> rgba8; // width * height * 4 bytes
    std::uint32_t             width  = 0;
    std::uint32_t             height = 0;
};

CapturedImage captureBuiltinScene() {
    CapturedImage img;
    img.width  = kScreenshotDiffWidth;
    img.height = kScreenshotDiffHeight;

    VulkanBundle bundle;
    if (!createVulkanBundle(bundle)) {
        return img; // empty; caller treats width/height mismatch as failure
    }

    // All Vulkan handles owned within this function. Declared up-front so
    // the cleanup lambda can reference them from any error-return path.
    // code-reviewer CRITICAL-1/2/3 fix groundwork: any VkResult failure
    // on the hot path now runs `cleanup()` before returning.
    VkImage         target       = VK_NULL_HANDLE;
    VkDeviceMemory  targetMem    = VK_NULL_HANDLE;
    VkBuffer        readback     = VK_NULL_HANDLE;
    VkDeviceMemory  readbackMem  = VK_NULL_HANDLE;
    VkCommandBuffer cmd          = VK_NULL_HANDLE;
    VkFence         fence        = VK_NULL_HANDLE;

    auto cleanup = [&]() {
        if (fence != VK_NULL_HANDLE) vkDestroyFence(bundle.device, fence, nullptr);
        if (cmd != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(bundle.device, bundle.commandPool, 1, &cmd);
        }
        if (readback != VK_NULL_HANDLE) vkDestroyBuffer(bundle.device, readback, nullptr);
        if (readbackMem != VK_NULL_HANDLE) vkFreeMemory(bundle.device, readbackMem, nullptr);
        if (target != VK_NULL_HANDLE) vkDestroyImage(bundle.device, target, nullptr);
        if (targetMem != VK_NULL_HANDLE) vkFreeMemory(bundle.device, targetMem, nullptr);
    };
    // Local VkResult check used throughout. Any failure cleans up all
    // in-scope handles and returns an empty CapturedImage.
    #define ENIGMA_HARNESS_VK_CHECK(expr)                                       \
        do {                                                                    \
            const VkResult _vr = (expr);                                        \
            if (_vr != VK_SUCCESS) {                                            \
                std::fprintf(stderr,                                            \
                    "[ScreenshotDiffHarness] %s failed: VkResult=%d\n",         \
                    #expr, static_cast<int>(_vr));                              \
                cleanup();                                                      \
                return {};                                                      \
            }                                                                   \
        } while (0)

    // --- target image (device-local, color attachment + transfer_src) ---
    {
        VkImageCreateInfo ci{};
        ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType     = VK_IMAGE_TYPE_2D;
        ci.format        = kColorFormat;
        ci.extent        = { kScreenshotDiffWidth, kScreenshotDiffHeight, 1 };
        ci.mipLevels     = 1;
        ci.arrayLayers   = 1;
        ci.samples       = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ci.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                         | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                         | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ci.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ENIGMA_HARNESS_VK_CHECK(vkCreateImage(bundle.device, &ci, nullptr, &target));

        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(bundle.device, target, &req);
        const std::uint32_t type = findMemoryType(bundle.physicalDevice, req.memoryTypeBits,
                                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type == UINT32_MAX) {
            std::fprintf(stderr, "[ScreenshotDiffHarness] no device-local memory type\n");
            cleanup();
            return {};
        }
        VkMemoryAllocateInfo ai{};
        ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = type;
        ENIGMA_HARNESS_VK_CHECK(vkAllocateMemory(bundle.device, &ai, nullptr, &targetMem));
        // code-reviewer CRITICAL-1 fix: was unchecked.
        ENIGMA_HARNESS_VK_CHECK(vkBindImageMemory(bundle.device, target, targetMem, 0));
    }

    // --- readback buffer (host-visible, coherent) ---
    const VkDeviceSize readbackSize = static_cast<VkDeviceSize>(kScreenshotDiffWidth)
                                    * kScreenshotDiffHeight * 4u;
    {
        VkBufferCreateInfo ci{};
        ci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size        = readbackSize;
        ci.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ENIGMA_HARNESS_VK_CHECK(vkCreateBuffer(bundle.device, &ci, nullptr, &readback));

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(bundle.device, readback, &req);
        const std::uint32_t type = findMemoryType(bundle.physicalDevice, req.memoryTypeBits,
                                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                 | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (type == UINT32_MAX) {
            std::fprintf(stderr, "[ScreenshotDiffHarness] no host-visible coherent memory type\n");
            cleanup();
            return {};
        }
        VkMemoryAllocateInfo ai{};
        ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = type;
        ENIGMA_HARNESS_VK_CHECK(vkAllocateMemory(bundle.device, &ai, nullptr, &readbackMem));
        // code-reviewer CRITICAL-2 fix: was unchecked.
        ENIGMA_HARNESS_VK_CHECK(vkBindBufferMemory(bundle.device, readback, readbackMem, 0));
    }

    // --- record + submit: transition, clear, transition, copy ---
    {
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = bundle.commandPool;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        // code-reviewer CRITICAL-3 fix: was unchecked.
        ENIGMA_HARNESS_VK_CHECK(vkAllocateCommandBuffers(bundle.device, &ai, &cmd));
    }
    {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        // code-reviewer CRITICAL-3 fix: was unchecked.
        ENIGMA_HARNESS_VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    }

    auto imageBarrier = [&](VkImageLayout oldLayout, VkImageLayout newLayout,
                             VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                             VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcAccessMask       = srcAccess;
        b.dstAccessMask       = dstAccess;
        b.oldLayout           = oldLayout;
        b.newLayout           = newLayout;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = target;
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
    };

    // UNDEFINED -> TRANSFER_DST for the clear. vkCmdClearColorImage is a
    // transfer op in core Vulkan (despite the "Clear" name), so it wants
    // transfer_dst layout, not color_attachment_optimal.
    imageBarrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkClearColorValue clear = kBuiltinClearColor;
    VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCmdClearColorImage(cmd, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);

    // TRANSFER_DST -> TRANSFER_SRC for the copy to readback buffer.
    imageBarrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy copy{};
    copy.bufferOffset      = 0;
    copy.bufferRowLength   = 0; // tightly packed — matches width
    copy.bufferImageHeight = 0;
    copy.imageSubresource  = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.imageOffset       = { 0, 0, 0 };
    copy.imageExtent       = { kScreenshotDiffWidth, kScreenshotDiffHeight, 1 };
    vkCmdCopyImageToBuffer(cmd, target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback, 1, &copy);

    // Make host reads see the copy. Coherent memory + fence-wait already
    // suffices in core Vulkan, but the explicit HOST barrier documents
    // intent and matches engine convention.
    VkBufferMemoryBarrier bb{};
    bb.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bb.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
    bb.dstAccessMask       = VK_ACCESS_HOST_READ_BIT;
    bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.buffer              = readback;
    bb.offset              = 0;
    bb.size                = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 0, nullptr, 1, &bb, 0, nullptr);

    // code-reviewer CRITICAL-3 fix: was unchecked.
    ENIGMA_HARNESS_VK_CHECK(vkEndCommandBuffer(cmd));

    // Submit with a fence and block on it — the harness is explicitly
    // synchronous. No async / frame-in-flight machinery.
    {
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        // code-reviewer CRITICAL-3 fix: was unchecked.
        ENIGMA_HARNESS_VK_CHECK(vkCreateFence(bundle.device, &fi, nullptr, &fence));
    }
    {
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cmd;
        // code-reviewer CRITICAL-3 fix: was unchecked.
        ENIGMA_HARNESS_VK_CHECK(vkQueueSubmit(bundle.graphicsQueue, 1, &si, fence));
    }
    ENIGMA_HARNESS_VK_CHECK(vkWaitForFences(bundle.device, 1, &fence, VK_TRUE, UINT64_MAX));

    // Map + copy into the output vector.
    // security LOW-1 / code-reviewer MINOR-1 fix: was unchecked.
    img.rgba8.resize(static_cast<std::size_t>(readbackSize));
    void* mapped = nullptr;
    ENIGMA_HARNESS_VK_CHECK(vkMapMemory(bundle.device, readbackMem, 0, readbackSize, 0, &mapped));
    std::memcpy(img.rgba8.data(), mapped, static_cast<std::size_t>(readbackSize));
    vkUnmapMemory(bundle.device, readbackMem);

    // Success path teardown — same ordering as cleanup() but inline so
    // we don't risk double-free via the lambda. Ordering: fence, cmd
    // buffer, buffer/image before their memory.
    vkDestroyFence(bundle.device, fence, nullptr);              fence       = VK_NULL_HANDLE;
    vkFreeCommandBuffers(bundle.device, bundle.commandPool, 1, &cmd);
                                                                 cmd         = VK_NULL_HANDLE;
    vkDestroyBuffer(bundle.device, readback, nullptr);          readback    = VK_NULL_HANDLE;
    vkFreeMemory(bundle.device, readbackMem, nullptr);          readbackMem = VK_NULL_HANDLE;
    vkDestroyImage(bundle.device, target, nullptr);             target      = VK_NULL_HANDLE;
    vkFreeMemory(bundle.device, targetMem, nullptr);            targetMem   = VK_NULL_HANDLE;
    // bundle destructor tears down device + instance.
    #undef ENIGMA_HARNESS_VK_CHECK
    return img;
}

// --- PNG I/O + diff ---------------------------------------------------------

bool writePng(const std::filesystem::path& outPng, const CapturedImage& img) {
    if (img.rgba8.empty()) return false;
    const std::string outStr = outPng.string();
    // stride_in_bytes = width * 4 for RGBA8 tightly packed.
    const int ok = stbi_write_png(outStr.c_str(),
                                   static_cast<int>(img.width),
                                   static_cast<int>(img.height),
                                   4, img.rgba8.data(),
                                   static_cast<int>(img.width * 4));
    return ok != 0;
}

// Returns (loaded, width, height). On failure `loaded.empty()`.
CapturedImage readPng(const std::filesystem::path& inPng) {
    CapturedImage img;
    const std::string inStr = inPng.string();
    // security LOW-2 fix: inspect PNG dimensions BEFORE stbi_load so a
    // hostile or corrupt reference cannot force an enormous allocation.
    // The harness's built-in scene is 128x128; clamp at 4096 as a
    // generous upper bound for future higher-resolution scenes.
    int pw = 0, ph = 0, pch = 0;
    if (!stbi_info(inStr.c_str(), &pw, &ph, &pch)
        || pw <= 0 || ph <= 0 || pw > 4096 || ph > 4096) {
        std::fprintf(stderr,
            "[ScreenshotDiffHarness] reference PNG dims suspicious (%dx%d) in %s\n",
            pw, ph, inStr.c_str());
        return img;
    }
    int w = 0, h = 0, ch = 0;
    // force 4 channels — we always compare in RGBA8.
    stbi_uc* data = stbi_load(inStr.c_str(), &w, &h, &ch, 4);
    if (data == nullptr) return img;
    img.width  = static_cast<std::uint32_t>(w);
    img.height = static_cast<std::uint32_t>(h);
    img.rgba8.assign(data, data + (static_cast<std::size_t>(w) * h * 4));
    stbi_image_free(data);
    return img;
}

} // namespace

ScreenshotDiffResult runScreenshotDiff(const std::string& testSceneName,
                                       const std::filesystem::path& referencePng,
                                       std::uint32_t pixelTolerance) {
    // Loud failure if the reference is missing — no silent "first run
    // pass" semantics. The captureBaseline() path is explicit.
    if (!std::filesystem::exists(referencePng)) {
        return makeFailure("reference PNG does not exist: " + referencePng.string()
                           + " (run with --capture-baseline to generate)");
    }

    const CapturedImage captured = captureBuiltinScene();
    if (captured.rgba8.empty()) {
        return makeFailure("capture failed for scene '" + testSceneName
                           + "' (Vulkan init or readback error — see stderr)");
    }

    const CapturedImage reference = readPng(referencePng);
    if (reference.rgba8.empty()) {
        return makeFailure("failed to load reference PNG: " + referencePng.string());
    }

    if (reference.width != captured.width || reference.height != captured.height) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "reference size (%ux%u) != captured size (%ux%u)",
            reference.width, reference.height, captured.width, captured.height);
        return makeFailure(buf);
    }

    // Per-pixel diff. A pixel is counted as "differing" if any of its
    // four channels deviates by more than `pixelTolerance`. maxDelta
    // tracks the largest channel delta seen, useful for tuning.
    ScreenshotDiffResult r;
    r.width      = captured.width;
    r.height     = captured.height;
    r.diffPixels = 0;
    r.maxDelta   = 0;

    const std::size_t pixelCount = static_cast<std::size_t>(captured.width) * captured.height;
    for (std::size_t i = 0; i < pixelCount; ++i) {
        bool differs = false;
        for (int c = 0; c < 4; ++c) {
            const int a = captured.rgba8[i * 4 + c];
            const int b = reference.rgba8[i * 4 + c];
            const int d = a > b ? a - b : b - a;
            if (static_cast<std::uint32_t>(d) > r.maxDelta) {
                r.maxDelta = static_cast<std::uint32_t>(d);
            }
            if (static_cast<std::uint32_t>(d) > pixelTolerance) {
                differs = true;
            }
        }
        if (differs) ++r.diffPixels;
    }

    r.passed = (r.diffPixels == 0);
    if (!r.passed) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "scene '%s' diff: %llu/%llu pixels differ (tolerance=%u, maxDelta=%u)",
            testSceneName.c_str(),
            static_cast<unsigned long long>(r.diffPixels),
            static_cast<unsigned long long>(pixelCount),
            pixelTolerance, r.maxDelta);
        r.message = buf;
    } else {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "scene '%s' OK (%ux%u, maxDelta=%u)",
            testSceneName.c_str(), r.width, r.height, r.maxDelta);
        r.message = buf;
    }
    return r;
}

bool captureBaseline(const std::string& testSceneName,
                     const std::filesystem::path& outPng) {
    const CapturedImage captured = captureBuiltinScene();
    if (captured.rgba8.empty()) {
        std::fprintf(stderr,
            "[ScreenshotDiffHarness] baseline capture failed for scene '%s'\n",
            testSceneName.c_str());
        return false;
    }
    // Ensure directory exists.
    const auto parent = outPng.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }
    if (!writePng(outPng, captured)) {
        std::fprintf(stderr,
            "[ScreenshotDiffHarness] stbi_write_png failed for '%s'\n",
            outPng.string().c_str());
        return false;
    }
    std::printf("[ScreenshotDiffHarness] wrote baseline: %s (%ux%u)\n",
                outPng.string().c_str(), captured.width, captured.height);
    return true;
}

} // namespace enigma::test_infra
