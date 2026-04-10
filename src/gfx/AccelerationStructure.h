#pragma once

#include "core/Types.h"

#include <volk.h>

#include <vector>

struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

namespace enigma::gfx {

class Allocator;
class Device;

// BLAS -- one per mesh primitive. Built from vertex/index buffers using BDA.
class BLAS {
public:
    static BLAS build(Device& device, Allocator& allocator,
                      VkBuffer vertexBuffer, u32 vertexCount, VkDeviceSize vertexStride,
                      VkBuffer indexBuffer, u32 indexCount,
                      bool allowCompaction = true);
    void destroy(Device& device, Allocator& allocator);

    VkAccelerationStructureKHR handle()        const { return m_as; }
    VkDeviceAddress            deviceAddress() const { return m_address; }

    // Refit: update BLAS geometry in-place (cheap, ~10% of rebuild cost).
    // Valid when vertex positions change but topology (triangle count) is unchanged.
    // AS quality degrades slightly; rebuild after major deformation.
    // Uses an immediate command buffer internally (synchronous, blocks until done).
    void refit(Device& device, Allocator& allocator,
               VkBuffer vertexBuffer, u32 vertexCount, VkDeviceSize vertexStride,
               VkBuffer indexBuffer, u32 indexCount);

    // Rebuild BLAS from scratch (full quality, higher cost).
    // Uses an immediate command buffer internally (synchronous, blocks until done).
    void rebuild(Device& device, Allocator& allocator,
                 VkBuffer vertexBuffer, u32 vertexCount, VkDeviceSize vertexStride,
                 VkBuffer indexBuffer, u32 indexCount);

private:
    VkAccelerationStructureKHR m_as         = VK_NULL_HANDLE;
    VkBuffer                   m_buffer     = VK_NULL_HANDLE;
    VmaAllocation              m_allocation = nullptr;
    VkDeviceAddress            m_address    = 0;
};

// TLAS -- rebuilt per-frame. Fixed max capacity (4096 instances).
class TLAS {
public:
    explicit TLAS(Device& device, Allocator& allocator, u32 maxInstances = 4096);
    void destroy(Device& device, Allocator& allocator);

    // Slot allocator: returns index into the instance array.
    u32  allocateInstanceSlot();
    void releaseInstanceSlot(u32 slot);

    // Set the transform + BLAS for a slot.
    void setInstance(u32 slot, const VkAccelerationStructureInstanceKHR& inst);

    // Rebuild the TLAS from all active instances. Call once per frame.
    void build(VkCommandBuffer cmd, Device& device, Allocator& allocator);

    VkAccelerationStructureKHR handle()        const { return m_as; }
    VkDeviceAddress            deviceAddress() const { return m_address; }

private:
    VkAccelerationStructureKHR           m_as           = VK_NULL_HANDLE;
    VkBuffer                             m_buffer       = VK_NULL_HANDLE;
    VmaAllocation                        m_allocation   = nullptr;
    VkDeviceAddress                      m_address      = 0;

    // Instance staging buffer (host-visible, persistently mapped).
    VkBuffer                             m_instanceBuf   = VK_NULL_HANDLE;
    VmaAllocation                        m_instanceAlloc = nullptr;
    VkAccelerationStructureInstanceKHR*  m_instances     = nullptr;

    // Scratch buffer for build.
    VkBuffer                             m_scratchBuf    = VK_NULL_HANDLE;
    VmaAllocation                        m_scratchAlloc  = nullptr;

    u32                      m_maxInstances = 0;
    std::vector<u32>         m_freeSlots;
    u32                      m_nextSlot = 0;
};

} // namespace enigma::gfx
