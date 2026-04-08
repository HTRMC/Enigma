#include "gfx/Allocator.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Device.h"
#include "gfx/Instance.h"

// VMA is driven via volk function pointers. These three macros must be
// set BEFORE vk_mem_alloc.h is included:
//   - VMA_STATIC_VULKAN_FUNCTIONS  = 0  -> do not assume statically linked
//   - VMA_DYNAMIC_VULKAN_FUNCTIONS = 0  -> do not vkGetInstanceProcAddr
//   - VMA_IMPLEMENTATION               -> emit the single TU implementation
#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#define VMA_IMPLEMENTATION
#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable: 4100 4127 4189 4324 4505)
#endif
#include <vk_mem_alloc.h>
#if defined(_MSC_VER)
    #pragma warning(pop)
#endif

namespace enigma::gfx {

Allocator::Allocator(Instance& instance, Device& device) {
    // Hand VMA a function table populated from the volk globals. This is
    // the canonical "volk + VMA" integration — neither the static nor the
    // dynamic VMA path applies.
    VmaVulkanFunctions fns{};
    fns.vkGetInstanceProcAddr                    = vkGetInstanceProcAddr;
    fns.vkGetDeviceProcAddr                      = vkGetDeviceProcAddr;
    fns.vkGetPhysicalDeviceProperties            = vkGetPhysicalDeviceProperties;
    fns.vkGetPhysicalDeviceMemoryProperties      = vkGetPhysicalDeviceMemoryProperties;
    fns.vkAllocateMemory                         = vkAllocateMemory;
    fns.vkFreeMemory                             = vkFreeMemory;
    fns.vkMapMemory                              = vkMapMemory;
    fns.vkUnmapMemory                            = vkUnmapMemory;
    fns.vkFlushMappedMemoryRanges                = vkFlushMappedMemoryRanges;
    fns.vkInvalidateMappedMemoryRanges           = vkInvalidateMappedMemoryRanges;
    fns.vkBindBufferMemory                       = vkBindBufferMemory;
    fns.vkBindImageMemory                        = vkBindImageMemory;
    fns.vkGetBufferMemoryRequirements            = vkGetBufferMemoryRequirements;
    fns.vkGetImageMemoryRequirements             = vkGetImageMemoryRequirements;
    fns.vkCreateBuffer                           = vkCreateBuffer;
    fns.vkDestroyBuffer                          = vkDestroyBuffer;
    fns.vkCreateImage                            = vkCreateImage;
    fns.vkDestroyImage                           = vkDestroyImage;
    fns.vkCmdCopyBuffer                          = vkCmdCopyBuffer;
    fns.vkGetBufferMemoryRequirements2KHR        = vkGetBufferMemoryRequirements2;
    fns.vkGetImageMemoryRequirements2KHR         = vkGetImageMemoryRequirements2;
    fns.vkBindBufferMemory2KHR                   = vkBindBufferMemory2;
    fns.vkBindImageMemory2KHR                    = vkBindImageMemory2;
    fns.vkGetPhysicalDeviceMemoryProperties2KHR  = vkGetPhysicalDeviceMemoryProperties2;
    fns.vkGetDeviceBufferMemoryRequirements      = vkGetDeviceBufferMemoryRequirements;
    fns.vkGetDeviceImageMemoryRequirements       = vkGetDeviceImageMemoryRequirements;

    VmaAllocatorCreateInfo info{};
    info.instance         = instance.handle();
    info.physicalDevice   = device.physical();
    info.device           = device.logical();
    info.pVulkanFunctions = &fns;
    info.vulkanApiVersion = VK_API_VERSION_1_3;
    // NOTE: VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT intentionally NOT
    // set — BDA is dropped at this milestone per ADR.

    ENIGMA_VK_CHECK(vmaCreateAllocator(&info, &m_allocator));
    ENIGMA_ASSERT(m_allocator != nullptr);
    ENIGMA_LOG_INFO("[gfx] VMA allocator created");
}

Allocator::~Allocator() {
    if (m_allocator == nullptr) {
        return;
    }

    // Leak assertion arrives at step 43 — this destructor only tears down
    // the allocator here. When step 43 lands it inserts the
    // vmaCalculateStatistics check immediately above the destroy call.
    vmaDestroyAllocator(m_allocator);
    m_allocator = nullptr;
}

} // namespace enigma::gfx
