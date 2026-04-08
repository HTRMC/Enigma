#pragma once

#include "core/Types.h"

#include <volk.h>

namespace enigma::gfx {

// Owns the VkInstance + volk loader state. Exactly one Instance should
// exist per process at this milestone. Construction calls `volkInitialize`,
// builds `VkInstanceCreateInfo`, creates the instance and loads the
// instance-level entry points via `volkLoadInstance`. Validation layers
// and the debug messenger are added in step 20.
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
    VkInstance m_instance = VK_NULL_HANDLE;
};

} // namespace enigma::gfx
