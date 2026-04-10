#include "gfx/GpuProfiler.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Device.h"

#include <cstring>

namespace enigma::gfx {

GpuProfiler::GpuProfiler(Device& device)
    : m_device(device.logical()) {

    // timestampPeriod: nanoseconds per GPU tick. Used to convert raw
    // 64-bit timestamp deltas into milliseconds during readback.
    m_nsPerTick = device.properties().limits.timestampPeriod;

    VkQueryPoolCreateInfo info{};
    info.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    info.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = kMaxZones * 2; // begin + end per zone

    const VkResult vr = vkCreateQueryPool(m_device, &info, nullptr, &m_queryPool);
    ENIGMA_ASSERT(vr == VK_SUCCESS && "vkCreateQueryPool failed for GpuProfiler");
    ENIGMA_LOG_INFO("[gfx] GpuProfiler created ({} timestamp slots)", kMaxZones * 2);
}

GpuProfiler::~GpuProfiler() {
    if (m_queryPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(m_device, m_queryPool, nullptr);
        m_queryPool = VK_NULL_HANDLE;
    }
}

void GpuProfiler::reset(VkCommandBuffer cmd) {
    vkCmdResetQueryPool(cmd, m_queryPool, 0, kMaxZones * 2);
    m_nextSlot = 0;
    m_pendingZones.clear();
}

void GpuProfiler::beginZone(VkCommandBuffer cmd, std::string_view name) {
    if (m_nextSlot + 1 >= kMaxZones * 2) {
        ENIGMA_LOG_WARN("[gfx] GpuProfiler: zone limit reached, skipping '{}'", name);
        return;
    }
    const u32 slot = m_nextSlot;
    m_nextSlot += 2;

    // VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT gives a conservative timestamp
    // at the point all prior commands complete. For individual pass timing,
    // callers may use stage-specific bits — but ALL_COMMANDS is safe everywhere.
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, m_queryPool, slot);
    m_pendingZones.push_back({std::string(name), slot});
}

void GpuProfiler::endZone(VkCommandBuffer cmd) {
    if (m_pendingZones.empty()) {
        return;
    }
    const u32 endSlot = m_pendingZones.back().beginSlot + 1;
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, m_queryPool, endSlot);
}

std::vector<GpuProfiler::ZoneResult> GpuProfiler::readback() {
    if (m_pendingZones.empty()) {
        return {};
    }

    const u32 slotCount = static_cast<u32>(m_pendingZones.size()) * 2;
    std::vector<u64> timestamps(slotCount, 0);

    // VK_QUERY_RESULT_64_BIT: timestamps are 64-bit values.
    // VK_QUERY_RESULT_WAIT_BIT: block until all queries are available.
    const VkResult vr = vkGetQueryPoolResults(
        m_device,
        m_queryPool,
        0,
        slotCount,
        slotCount * sizeof(u64),
        timestamps.data(),
        sizeof(u64),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    if (vr != VK_SUCCESS) {
        ENIGMA_LOG_WARN("[gfx] GpuProfiler::readback() vkGetQueryPoolResults failed: {}",
                        static_cast<int>(vr));
        return {};
    }

    std::vector<ZoneResult> results;
    results.reserve(m_pendingZones.size());
    for (const auto& zone : m_pendingZones) {
        const u64 begin = timestamps[zone.beginSlot];
        const u64 end   = timestamps[zone.beginSlot + 1];
        const f32 ms    = static_cast<f32>(end - begin) * m_nsPerTick * 1e-6f;
        results.push_back({zone.name, ms});
    }
    return results;
}

} // namespace enigma::gfx
