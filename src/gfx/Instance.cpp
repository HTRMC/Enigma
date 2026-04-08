#include "gfx/Instance.h"

#include "core/Assert.h"
#include "core/Log.h"

#include <GLFW/glfw3.h>

#include <vector>

namespace enigma::gfx {

Instance::Instance() {
    // volk MUST be initialized before any Vulkan entry points are called.
    // It loads vkGetInstanceProcAddr dynamically and populates the
    // global-level function pointers (vkCreateInstance, etc.).
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

    // GLFW tells us which instance extensions it needs for surface creation
    // (VK_KHR_surface + the OS-specific surface extension, e.g. win32).
    u32 glfwExtCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    std::vector<const char*> enabledExtensions;
    enabledExtensions.reserve(glfwExtCount);
    for (u32 i = 0; i < glfwExtCount; ++i) {
        enabledExtensions.push_back(glfwExts[i]);
    }

    VkInstanceCreateInfo createInfo{};
    createInfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo        = &appInfo;
    createInfo.enabledExtensionCount   = static_cast<u32>(enabledExtensions.size());
    createInfo.ppEnabledExtensionNames = enabledExtensions.data();
    createInfo.enabledLayerCount       = 0;
    createInfo.ppEnabledLayerNames     = nullptr;

    ENIGMA_VK_CHECK(vkCreateInstance(&createInfo, nullptr, &m_instance));
    ENIGMA_ASSERT(m_instance != VK_NULL_HANDLE);

    // Populate instance-level function pointers for volk.
    volkLoadInstance(m_instance);

    ENIGMA_LOG_INFO("[gfx] VkInstance created (api = 1.3, extensions = {})",
                    enabledExtensions.size());
}

Instance::~Instance() {
    if (m_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(m_instance, nullptr);
        m_instance = VK_NULL_HANDLE;
    }
}

} // namespace enigma::gfx
