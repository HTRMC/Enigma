#pragma once

#include "core/Types.h"

#include <volk.h>

namespace enigma::gfx {

// Global validation counter. The debug messenger installed by `Instance`
// increments this on any WARNING_BIT / ERROR_BIT message. The Renderer
// asserts it is zero at the end of its destructor (post-vkDeviceWaitIdle);
// see `Renderer::~Renderer()` for the single enforcement point.
u32  getValidationCounter();
void resetValidationCounter();

// Enable the debug-build validation layer at this build. Queried by Instance
// during enabledLayer population and by the messenger create step.
bool validationEnabled();

// Populate a `VkDebugUtilsMessengerCreateInfoEXT` that points at the counter
// callback. Used both in the VkInstance pNext chain (so instance-creation
// validation is captured) and as a standalone messenger after instance
// creation (so the returned VkDebugUtilsMessengerEXT handle can be
// destroyed cleanly at shutdown).
void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& info);

VkResult createDebugUtilsMessenger(
    VkInstance                   instance,
    VkDebugUtilsMessengerEXT*    pMessenger);

void destroyDebugUtilsMessenger(
    VkInstance               instance,
    VkDebugUtilsMessengerEXT messenger);

} // namespace enigma::gfx
