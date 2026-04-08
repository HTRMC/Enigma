#include "gfx/Instance.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Validation.h"

#include <GLFW/glfw3.h>

#include <cstring>
#include <vector>

namespace enigma::gfx {

namespace {

constexpr const char* kValidationLayerName = "VK_LAYER_KHRONOS_validation";

bool validationLayerAvailable() {
    u32 count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (const auto& l : layers) {
        if (std::strcmp(l.layerName, kValidationLayerName) == 0) {
            return true;
        }
    }
    return false;
}

} // namespace

Instance::Instance() {
    {
        const VkResult volkInit = volkInitialize();
        if (volkInit != VK_SUCCESS) {
            ENIGMA_LOG_ERROR("[gfx] volkInitialize failed: {}", static_cast<int>(volkInit));
            ENIGMA_ASSERT(false);
            return;
        }
    }

    VkApplicationInfo appInfo{};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName   = "Enigma";
    appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    appInfo.pEngineName        = "Enigma";
    appInfo.engineVersion      = VK_MAKE_API_VERSION(0, 0, 1, 0);
    appInfo.apiVersion         = VK_API_VERSION_1_3;

    // GLFW required extensions (surface + platform surface).
    u32 glfwExtCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    std::vector<const char*> enabledExtensions;
    enabledExtensions.reserve(glfwExtCount + 2);
    for (u32 i = 0; i < glfwExtCount; ++i) {
        enabledExtensions.push_back(glfwExts[i]);
    }

    // Debug builds: add validation layer + debug utils extension.
    const bool wantValidation = validationEnabled();
    const bool haveValidationLayer = wantValidation && validationLayerAvailable();
    if (wantValidation && !haveValidationLayer) {
        ENIGMA_LOG_WARN("[gfx] validation layer requested but not available");
    }
    if (haveValidationLayer) {
        enabledExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    std::vector<const char*> enabledLayers;
    if (haveValidationLayer) {
        enabledLayers.push_back(kValidationLayerName);
    }

    VkInstanceCreateInfo createInfo{};
    createInfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo        = &appInfo;
    createInfo.enabledExtensionCount   = static_cast<u32>(enabledExtensions.size());
    createInfo.ppEnabledExtensionNames = enabledExtensions.data();
    createInfo.enabledLayerCount       = static_cast<u32>(enabledLayers.size());
    createInfo.ppEnabledLayerNames     = enabledLayers.data();

    // Wire the debug messenger into the instance create pNext chain so
    // that validation messages produced during vkCreateInstance itself are
    // caught by our counter.
    VkDebugUtilsMessengerCreateInfoEXT preInfo{};
    if (haveValidationLayer) {
        populateDebugMessengerCreateInfo(preInfo);
        createInfo.pNext = &preInfo;
    }

    ENIGMA_VK_CHECK(vkCreateInstance(&createInfo, nullptr, &m_instance));
    ENIGMA_ASSERT(m_instance != VK_NULL_HANDLE);

    volkLoadInstance(m_instance);

    // Create the standalone messenger so we own a destroyable handle.
    if (haveValidationLayer) {
        ENIGMA_VK_CHECK(createDebugUtilsMessenger(m_instance, &m_messenger));
    }

    ENIGMA_LOG_INFO(
        "[gfx] VkInstance created (api = 1.3, ext = {}, validation = {})",
        enabledExtensions.size(),
        haveValidationLayer ? "on" : "off");
}

Instance::~Instance() {
    if (m_messenger != VK_NULL_HANDLE) {
        destroyDebugUtilsMessenger(m_instance, m_messenger);
        m_messenger = VK_NULL_HANDLE;
    }
    if (m_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(m_instance, nullptr);
        m_instance = VK_NULL_HANDLE;
    }
}

} // namespace enigma::gfx
