#pragma once

#include "core/Types.h"

#include <volk.h>

#include <string>
#include <string_view>
#include <vector>

namespace enigma::gfx {

class Device;

// GpuProfiler
// ===========
// Lightweight GPU timestamp profiler backed by a VkQueryPool. Supports up
// to kMaxZones nested/sequential named zones per frame. Each zone occupies
// two timestamp slots (begin + end). Usage pattern per frame:
//
//   profiler.reset(cmd);               // vkCmdResetQueryPool at frame start
//   profiler.beginZone(cmd, "MeshPass");
//   // ... draw calls ...
//   profiler.endZone(cmd);
//   // ... end command buffer, submit ...
//   auto results = profiler.readback(); // after GPU finishes the frame
//
// Tracy integration is compiled in when ENIGMA_TRACY is defined.
class GpuProfiler {
public:
    static constexpr u32 kMaxZones = 32;

    struct ZoneResult {
        std::string name;
        f32         durationMs = 0.0f;
    };

    explicit GpuProfiler(Device& device);
    ~GpuProfiler();

    GpuProfiler(const GpuProfiler&)            = delete;
    GpuProfiler& operator=(const GpuProfiler&) = delete;
    GpuProfiler(GpuProfiler&&)                 = delete;
    GpuProfiler& operator=(GpuProfiler&&)      = delete;

    // Reset query pool slots. Call once per frame before any beginZone().
    void reset(VkCommandBuffer cmd);

    // Write a "begin" timestamp and push a zone name onto the stack.
    void beginZone(VkCommandBuffer cmd, std::string_view name);

    // Write an "end" timestamp for the innermost open zone.
    void endZone(VkCommandBuffer cmd);

    // Read back all completed zone results. Blocks until GPU work is done
    // (VK_QUERY_RESULT_WAIT_BIT). Returns results in zone-begin order.
    // Call after the frame's submission fence has signalled.
    std::vector<ZoneResult> readback();

private:
    VkDevice      m_device        = VK_NULL_HANDLE;
    VkQueryPool   m_queryPool     = VK_NULL_HANDLE;
    f32           m_nsPerTick     = 1.0f; // timestampPeriod in nanoseconds

    u32           m_nextSlot      = 0;    // next free query index (incremented by 2 per zone)

    struct PendingZone {
        std::string name;
        u32         beginSlot = 0;
    };
    std::vector<PendingZone> m_pendingZones; // zones recorded this frame, awaiting readback
};

} // namespace enigma::gfx
