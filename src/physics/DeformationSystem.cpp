#include "physics/DeformationSystem.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Allocator.h"
#include "gfx/Device.h"
#include "scene/Scene.h"

#include <volk.h>
#include <vk_mem_alloc.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace enigma {

void DeformationSystem::registerPrimitive(u32 primitiveIndex,
                                           std::vector<vec3> originalPositions,
                                           CrumpleZone zone) {
    ENIGMA_ASSERT(originalPositions.size() == zone.vertices.size());

    PrimitiveState state;
    state.index             = primitiveIndex;
    state.originalPositions = std::move(originalPositions);
    state.deformedPositions = state.originalPositions;
    state.zone              = std::move(zone);

    m_primitives.push_back(std::move(state));
}

f32 DeformationSystem::applyImpact(const ImpactEvent& event, f32 radius) {
    f32 maxDisp = 0.0f;

    for (auto& ps : m_primitives) {
        const u32 count = static_cast<u32>(ps.deformedPositions.size());

        for (u32 i = 0; i < count; ++i) {
            const vec3& pos = ps.deformedPositions[i];
            const f32 dist = glm::length(pos - event.worldPosition);
            if (dist >= radius) continue;

            const auto& vert = ps.zone.vertices[i];
            if (vert.weight <= 0.0f) continue;

            const f32 distanceAttenuation = 1.0f - (dist / radius);

            // Current total displacement from original.
            const f32 currentTotalDisp = glm::length(ps.deformedPositions[i] - ps.originalPositions[i]);

            f32 displacement = event.force * vert.weight / vert.hardness * distanceAttenuation;
            displacement = std::min(displacement, vert.maxDisplacement - currentTotalDisp);
            displacement = std::max(displacement, 0.0f);

            if (displacement > 0.0f) {
                ps.deformedPositions[i] += event.direction * displacement;
                maxDisp = std::max(maxDisp, displacement);
            }
        }

        // Accumulate damage.
        if (maxDisp > 0.0f) {
            ps.zone.currentDamage = std::min(ps.zone.currentDamage + maxDisp * 0.5f, 1.0f);
        }
    }

    return maxDisp;
}

void DeformationSystem::uploadDeformedPositions(u32 primitiveIndex,
                                                  VkBuffer vertexBuffer,
                                                  gfx::Device& device,
                                                  gfx::Allocator& allocator) {
    const PrimitiveState* ps = findPrimitive(primitiveIndex);
    if (ps == nullptr) return;

    const u32 vertexCount = static_cast<u32>(ps->deformedPositions.size());
    if (vertexCount == 0) return;

    // Vertex layout: position(vec3) + normal(vec3) + uv(vec2) + tangent(vec4) = 48 bytes.
    constexpr VkDeviceSize kVertexStride = 48;
    constexpr VkDeviceSize kPositionSize = sizeof(vec3); // 12 bytes

    const VkDeviceSize stagingSize = static_cast<VkDeviceSize>(vertexCount) * kPositionSize;

    // Allocate host-visible staging buffer.
    VkBufferCreateInfo stagingCI{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    stagingCI.size  = stagingSize;
    stagingCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocCI{};
    stagingAllocCI.usage = VMA_MEMORY_USAGE_AUTO;
    stagingAllocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                           VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer      stagingBuf{};
    VmaAllocation stagingAlloc{};
    VmaAllocationInfo stagingInfo{};
    ENIGMA_VK_CHECK(vmaCreateBuffer(allocator.handle(), &stagingCI, &stagingAllocCI,
                                    &stagingBuf, &stagingAlloc, &stagingInfo));

    // Copy deformed positions into staging (packed, one vec3 per vertex).
    auto* dst = static_cast<vec3*>(stagingInfo.pMappedData);
    for (u32 i = 0; i < vertexCount; ++i) {
        dst[i] = ps->deformedPositions[i];
    }
    vmaFlushAllocation(allocator.handle(), stagingAlloc, 0, VK_WHOLE_SIZE);

    // Build one copy region per vertex: staging[i*12] -> vertexBuffer[i*48], 12 bytes.
    std::vector<VkBufferCopy> regions;
    regions.reserve(vertexCount);
    for (u32 i = 0; i < vertexCount; ++i) {
        VkBufferCopy r{};
        r.srcOffset = static_cast<VkDeviceSize>(i) * kPositionSize;
        r.dstOffset = static_cast<VkDeviceSize>(i) * kVertexStride;
        r.size      = kPositionSize;
        regions.push_back(r);
    }

    // Immediate submit: single vkCmdCopyBuffer with N regions instead of N vkCmdUpdateBuffer calls.
    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolCI.queueFamilyIndex = device.graphicsQueueFamily();

    VkCommandPool cmdPool = VK_NULL_HANDLE;
    ENIGMA_VK_CHECK(vkCreateCommandPool(device.logical(), &poolCI, nullptr, &cmdPool));

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool        = cmdPool;
    cmdAllocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    ENIGMA_VK_CHECK(vkAllocateCommandBuffers(device.logical(), &cmdAllocInfo, &cmd));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ENIGMA_VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    vkCmdCopyBuffer(cmd, stagingBuf, vertexBuffer,
                    static_cast<u32>(regions.size()), regions.data());

    ENIGMA_VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;
    ENIGMA_VK_CHECK(vkQueueSubmit(device.graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE));
    ENIGMA_VK_CHECK(vkQueueWaitIdle(device.graphicsQueue()));

    vkDestroyCommandPool(device.logical(), cmdPool, nullptr);
    vmaDestroyBuffer(allocator.handle(), stagingBuf, stagingAlloc);
}

bool DeformationSystem::requiresBlasRebuild(u32 primitiveIndex) const {
    const PrimitiveState* ps = findPrimitive(primitiveIndex);
    if (ps == nullptr) return false;

    const u32 vertexCount = static_cast<u32>(ps->deformedPositions.size());
    if (vertexCount == 0) return false;

    // Count vertices whose total displacement from original exceeds 30% of their max.
    u32 largeCount = 0;
    for (u32 i = 0; i < vertexCount; ++i) {
        const f32 totalDisp = glm::length(ps->deformedPositions[i] - ps->originalPositions[i]);
        const f32 threshold = ps->zone.vertices[i].maxDisplacement * 0.3f;
        if (totalDisp > threshold) {
            ++largeCount;
        }
    }

    const f32 ratio = static_cast<f32>(largeCount) / static_cast<f32>(vertexCount);
    return ratio > 0.3f;
}

void DeformationSystem::reset(u32 primitiveIndex) {
    PrimitiveState* ps = findPrimitive(primitiveIndex);
    if (ps == nullptr) return;

    ps->deformedPositions  = ps->originalPositions;
    ps->zone.currentDamage = 0.0f;
}

const std::vector<vec3>& DeformationSystem::deformedPositions(u32 primitiveIndex) const {
    const PrimitiveState* ps = findPrimitive(primitiveIndex);
    ENIGMA_ASSERT(ps != nullptr);
    return ps->deformedPositions;
}

DeformationSystem::PrimitiveState* DeformationSystem::findPrimitive(u32 index) {
    for (auto& ps : m_primitives) {
        if (ps.index == index) return &ps;
    }
    return nullptr;
}

const DeformationSystem::PrimitiveState* DeformationSystem::findPrimitive(u32 index) const {
    for (const auto& ps : m_primitives) {
        if (ps.index == index) return &ps;
    }
    return nullptr;
}

} // namespace enigma
