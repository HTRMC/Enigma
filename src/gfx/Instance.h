#pragma once

#include "core/Types.h"

#include <volk.h>

namespace enigma::gfx {

// Owns the VkInstance + volk loader state + (debug builds) the
// VkDebugUtilsMessengerEXT. Construction order is:
//   1. volkInitialize
//   2. vkCreateInstance (with validation layer + debug messenger in pNext
//      when available, so instance-creation messages are captured)
//   3. volkLoadInstance
//   4. vkCreateDebugUtilsMessengerEXT (standalone handle for clean shutdown)
class Instance {
public:
    Instance();
    ~Instance();

    Instance(const Instance&)            = delete;
    Instance& operator=(const Instance&) = delete;
    Instance(Instance&&)                 = delete;
    Instance& operator=(Instance&&)      = delete;

    VkInstance handle() const { return m_instance; }

private:
    VkInstance               m_instance  = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_messenger = VK_NULL_HANDLE;
};

} // namespace enigma::gfx
