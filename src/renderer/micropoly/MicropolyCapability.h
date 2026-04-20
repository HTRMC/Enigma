#pragma once

#include "core/Types.h"

namespace enigma::gfx { class Device; }

namespace enigma {

// HwMatrixRow
// -----------
// Enumerates the rows of the HW Capability Matrix documented in
// src/renderer/micropoly/README.md (and .omc/plans/ralplan-micropolygon.md
// §3.M0a). Each row corresponds to one possible micropoly runtime mode
// dictated by the device's reported feature set.
enum class HwMatrixRow : u8 {
    // (a) mesh + atomic_int64 + image_int64 + sparse + RT
    FullFeatureSet          = 0,
    // (b) mesh + atomic_int64 + image_int64 + RT (no sparse)
    RtShadowOnly            = 1,
    // (c) mesh + atomic_int64 + image_int64 + sparse (no RT)
    VsmShadow               = 2,
    // (d) mesh + atomic_int64 + image_int64 (no sparse, no RT)
    GeometryNoShadows       = 3,
    // (e) mesh + atomic_int64 (image_int64 absent)
    HwOnly                  = 4,
    // (f) mesh shader OR atomic_int64 absent — hard gate off
    Disabled                = 5,
};

// Flat list of probed capabilities. Each member reflects the result of a
// single Vulkan feature or extension probe performed against the logical
// gfx::Device at construction time. MicropolyCaps is a pure value type —
// no device handles are retained.
struct MicropolyCaps {
    bool meshShader        = false; // VK_EXT_mesh_shader
    bool atomicInt64       = false; // VkPhysicalDeviceShaderAtomicInt64Features::shaderBufferInt64Atomics
    bool imageInt64        = false; // VK_EXT_shader_image_atomic_int64 (SPV_EXT_shader_image_int64)
    bool sparseResidency   = false; // VkPhysicalDeviceFeatures::sparseBinding + sparseResidencyImage2D
    bool rayTracing        = false; // VK_KHR_ray_tracing_pipeline

    HwMatrixRow row        = HwMatrixRow::Disabled;

    // Human-readable status string for the single "micropoly: <status>" log
    // line emitted at boot. Never empty.
    const char* statusString = "disabled";
};

// Classify a pre-populated capability bag into a HW matrix row. Pure function;
// exposed for unit testing (see tests/micropoly_capability_test.cpp). The
// device-facing overload below wraps this after driving the probes.
HwMatrixRow classifyMicropolyRow(bool meshShader, bool atomicInt64,
                                  bool imageInt64, bool sparseResidency,
                                  bool rayTracing);

// Translate a HW matrix row into the boot-log status string. Kept next to
// classify() so both are single source of truth, mirroring plan §3.M0a.
const char* statusStringFor(HwMatrixRow row);

// Probe the given Device and return a fully populated MicropolyCaps.
// Reads VkPhysicalDevice feature structs via vkGetPhysicalDeviceFeatures2
// and extension enumeration — does not modify the device.
MicropolyCaps micropolyCaps(gfx::Device& device);

} // namespace enigma
