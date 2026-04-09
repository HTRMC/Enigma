#include "gfx/Validation.h"

#include "core/Assert.h"
#include "core/Log.h"

#include <atomic>

namespace enigma::gfx {

namespace {

std::atomic<u32> g_validationCounter{0};

#if ENIGMA_DEBUG
constexpr bool kValidationEnabled = true;
#else
constexpr bool kValidationEnabled = false;
#endif

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT      severity,
    VkDebugUtilsMessageTypeFlagsEXT             type,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*                                       /*userData*/) {

    const char* msg = (data != nullptr && data->pMessage != nullptr)
                          ? data->pMessage
                          : "(null)";

    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        // Only count real API validation errors, not loader/layer general messages
        // (e.g. OBS duplicate-layer warnings are GENERAL_BIT, not VALIDATION_BIT).
        if (type & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) {
            g_validationCounter.fetch_add(1, std::memory_order_relaxed);
        }
        ENIGMA_LOG_ERROR("[validation] {}", msg);
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        if (type & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) {
            g_validationCounter.fetch_add(1, std::memory_order_relaxed);
        }
        ENIGMA_LOG_WARN("[validation] {}", msg);
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
        ENIGMA_LOG_INFO("[validation] {}", msg);
    } else {
        // Verbose. Drop on the floor to avoid spam.
    }

    // Never abort from the callback. The shutdown gate in Renderer::~Renderer
    // is the single point that fails the run if the counter is non-zero.
    return VK_FALSE;
}

} // namespace

u32 getValidationCounter() {
    return g_validationCounter.load(std::memory_order_relaxed);
}

void resetValidationCounter() {
    g_validationCounter.store(0, std::memory_order_relaxed);
}

bool validationEnabled() {
    return kValidationEnabled;
}

void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& info) {
    info       = {};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT     |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT  |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = &debugCallback;
    info.pUserData       = nullptr;
}

VkResult createDebugUtilsMessenger(
    VkInstance                instance,
    VkDebugUtilsMessengerEXT* pMessenger) {

    if (!kValidationEnabled) {
        *pMessenger = VK_NULL_HANDLE;
        return VK_SUCCESS;
    }
    if (vkCreateDebugUtilsMessengerEXT == nullptr) {
        ENIGMA_LOG_WARN("[validation] vkCreateDebugUtilsMessengerEXT unavailable");
        *pMessenger = VK_NULL_HANDLE;
        return VK_SUCCESS;
    }

    VkDebugUtilsMessengerCreateInfoEXT info{};
    populateDebugMessengerCreateInfo(info);
    return vkCreateDebugUtilsMessengerEXT(instance, &info, nullptr, pMessenger);
}

void destroyDebugUtilsMessenger(
    VkInstance               instance,
    VkDebugUtilsMessengerEXT messenger) {

    if (messenger == VK_NULL_HANDLE) {
        return;
    }
    if (vkDestroyDebugUtilsMessengerEXT == nullptr) {
        return;
    }
    vkDestroyDebugUtilsMessengerEXT(instance, messenger, nullptr);
}

} // namespace enigma::gfx
