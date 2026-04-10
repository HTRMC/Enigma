#include "physics/DeformationSystem.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Device.h"
#include "scene/Scene.h"

#include <volk.h>

#include <algorithm>
#include <cstring>

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
    state.largeDisplacementCount = 0;

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

                // Track large displacements for rebuild decision.
                if (displacement > vert.maxDisplacement * 0.3f) {
                    ++ps.largeDisplacementCount;
                }
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
                                                  gfx::Device& device) {
    const PrimitiveState* ps = findPrimitive(primitiveIndex);
    if (ps == nullptr) return;

    // Vertex layout: position(vec3) + normal(vec3) + uv(vec2) + tangent(vec4) = 48 bytes.
    constexpr VkDeviceSize kVertexStride = 48;
    constexpr VkDeviceSize kPositionSize = sizeof(vec3); // 12 bytes

    // Get buffer memory via vkMapMemory on the device memory bound to vertexBuffer.
    VkMemoryRequirements memReqs{};
    vkGetBufferMemoryRequirements(device.logical(), vertexBuffer, &memReqs);

    // Use an immediate command buffer to copy position data via staging.
    // For simplicity, we use vkCmdUpdateBuffer which supports small updates
    // inline in the command buffer (max 65536 bytes per call).
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

    const u32 vertexCount = static_cast<u32>(ps->deformedPositions.size());
    for (u32 i = 0; i < vertexCount; ++i) {
        const VkDeviceSize offset = static_cast<VkDeviceSize>(i) * kVertexStride;
        vkCmdUpdateBuffer(cmd, vertexBuffer, offset, kPositionSize, &ps->deformedPositions[i]);
    }

    ENIGMA_VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;
    ENIGMA_VK_CHECK(vkQueueSubmit(device.graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE));
    ENIGMA_VK_CHECK(vkQueueWaitIdle(device.graphicsQueue()));

    vkDestroyCommandPool(device.logical(), cmdPool, nullptr);
}

bool DeformationSystem::requiresBlasRebuild(u32 primitiveIndex) const {
    const PrimitiveState* ps = findPrimitive(primitiveIndex);
    if (ps == nullptr) return false;

    const u32 totalVertices = static_cast<u32>(ps->deformedPositions.size());
    if (totalVertices == 0) return false;

    // Rebuild if > 30% of vertices have large displacements.
    const f32 ratio = static_cast<f32>(ps->largeDisplacementCount) / static_cast<f32>(totalVertices);
    return ratio > 0.3f;
}

void DeformationSystem::reset(u32 primitiveIndex) {
    PrimitiveState* ps = findPrimitive(primitiveIndex);
    if (ps == nullptr) return;

    ps->deformedPositions = ps->originalPositions;
    ps->largeDisplacementCount = 0;
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
