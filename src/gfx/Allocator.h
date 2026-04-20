#pragma once

#include "core/Types.h"

#include <volk.h>

#include <memory>

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
// BDA flag is set: VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT enables
// device-address-capable allocations required by RT acceleration structures.
class Allocator {
public:
    Allocator(Instance& instance, Device& device);
    ~Allocator();

    Allocator(const Allocator&)            = delete;
    Allocator& operator=(const Allocator&) = delete;
    Allocator(Allocator&&)                 = delete;
    Allocator& operator=(Allocator&&)      = delete;

    // --- Test-only adopt path (M2.4a micropoly streaming tests) ---
    //
    // Wraps an already-created VmaAllocator in a gfx::Allocator so
    // test TUs that bring up their own headless Vulkan bundle (and
    // their own VMA allocator) can still pass a real `gfx::Allocator&`
    // reference into code that expects one (e.g. MicropolyStreaming).
    //
    // Ownership: the adopted Allocator does NOT destroy the VmaAllocator
    // — the caller retains full ownership. Mirrors Device::adopt(). This
    // replaces the `GfxAllocatorLayoutProbe` reinterpret_cast hack used
    // by pre-Phase-4 tests (Phase-4 Security MEDIUM fix).
    static std::unique_ptr<Allocator> adopt(VmaAllocator raw);

    VmaAllocator handle() const { return m_allocator; }

private:
    // Private ctor used by adopt() — wraps an externally-owned VMA.
    explicit Allocator(VmaAllocator externalVma);

    VmaAllocator m_allocator = nullptr;

    // When true, the destructor does NOT call vmaDestroyAllocator —
    // the handle was adopted from a caller who retains ownership.
    bool m_externallyOwnedVma = false;
};

} // namespace enigma::gfx
