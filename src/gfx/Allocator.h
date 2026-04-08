#pragma once

#include "core/Types.h"

#include <volk.h>

// Forward declare the VMA allocator opaque handle — we only include the
// full header in Allocator.cpp so the rest of the codebase does not pay
// the VMA header cost.
struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T*;

namespace enigma::gfx {

class Instance;
class Device;

// RAII wrapper around VMA. VMA is driven via volk-loaded function pointers
// (VMA_STATIC_VULKAN_FUNCTIONS / VMA_DYNAMIC_VULKAN_FUNCTIONS are both 0
// in Allocator.cpp; the function table is populated manually from the
// volk globals).
//
// BDA flag is NOT set: `bufferDeviceAddress` is dropped at this milestone.
class Allocator {
public:
    Allocator(Instance& instance, Device& device);
    ~Allocator();

    Allocator(const Allocator&)            = delete;
    Allocator& operator=(const Allocator&) = delete;
    Allocator(Allocator&&)                 = delete;
    Allocator& operator=(Allocator&&)      = delete;

    VmaAllocator handle() const { return m_allocator; }

private:
    VmaAllocator m_allocator = nullptr;
};

} // namespace enigma::gfx
