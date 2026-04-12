#pragma once

// Jolt debug-renderer headers must appear before any namespace blocks.
// They are guarded so Release builds (JPH_NO_DEBUG → no JPH_DEBUG_RENDERER)
// compile the lightweight no-op stub instead.
#ifdef JPH_DEBUG_RENDERER
#include <Jolt/Jolt.h>
#include <Jolt/Renderer/DebugRendererSimple.h>
#include <Jolt/Physics/Body/BodyFilter.h>
#endif

#include "core/Types.h"

#include <volk.h>

#include <cstdint>
#include <vector>

// Forward-declare VMA handle (matches the typedef in vk_mem_alloc.h).
struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

// Forward-declare so gather() can appear in both the real class and the no-op
// stub without pulling all of Jolt into every translation unit.
namespace JPH { class PhysicsSystem; }

namespace enigma {

namespace gfx {
class Device;
class Allocator;
class DescriptorAllocator;
class ShaderManager;
} // namespace gfx

struct PhysicsDebugInitInfo {
    gfx::Device*              device              = nullptr;
    gfx::Allocator*           allocator           = nullptr;
    gfx::DescriptorAllocator* descriptorAllocator = nullptr;
    gfx::ShaderManager*       shaderManager       = nullptr;
    VkDescriptorSetLayout     globalSetLayout      = VK_NULL_HANDLE;
    VkFormat                  colorFormat         = VK_FORMAT_UNDEFINED;
    VkFormat                  depthFormat         = VK_FORMAT_UNDEFINED;
};

// ─────────────────────────────────────────────────────────────────────────────
#ifdef JPH_DEBUG_RENDERER

// Skips PhysicsLayer::Static (layer 0) so the 512×512 heightfield's
// ~500k triangles never enter the draw path.
class DynamicBodyFilter final : public JPH::BodyDrawFilter {
public:
    bool ShouldDraw(const JPH::Body& inBody) const override;
};

// Collects wireframe line geometry from dynamic physics bodies each frame
// and renders them as a Vulkan LINE_LIST overlay via the render graph.
//
// Per-frame contract:
//   1. gather(physicsSystem)  — populate CPU staging via Jolt's DrawBodies()
//   2. upload()               — memcpy staging → persistently-mapped GPU SSBO
//   3. drawFrame(cmd, ...)    — record draw inside render graph pass lambda
class PhysicsDebugRenderer final : public JPH::DebugRendererSimple {
public:
    PhysicsDebugRenderer()           = default;
    ~PhysicsDebugRenderer() override = default;

    void init(const PhysicsDebugInitInfo& info);
    void destroy();

    void gather(JPH::PhysicsSystem& physicsSystem);
    void upload();
    void drawFrame(VkCommandBuffer cmd, VkDescriptorSet globalSet,
                   VkExtent2D ext, u32 cameraSlot);

    bool enabled          = false;
    bool depthTestEnabled = true;

protected:
    // JPH::DebugRendererSimple required overrides.
    // DrawText3D: no-op — text rendering not needed for wireframe bodies.
    // DrawTriangle: auto-derived by base class as three DrawLine() calls.
    void DrawLine(JPH::RVec3Arg inFrom, JPH::RVec3Arg inTo,
                  JPH::ColorArg inColor) override;
    void DrawText3D(JPH::RVec3Arg, const std::string_view&,
                    JPH::ColorArg, float) override {}

private:
    // 16-byte vertex matching StructuredBuffer<float4> in physics_debug.hlsl.
    // xyz = world-space position, w-bits = RGBA8 colour (asuint in shader).
    struct LineVertex {
        float    x, y, z;
        uint32_t color; // r | (g<<8) | (b<<16) | (a<<24)
    };
    static_assert(sizeof(LineVertex) == 16);

    std::vector<LineVertex> m_lineVertices;
    u32                     m_uploadedCount = 0;

    // Persistently-mapped GPU SSBO: 64k lines × 2 verts × 16 B = 2 MB.
    VkBuffer       m_ssbo       = VK_NULL_HANDLE;
    VmaAllocation  m_allocation = nullptr;
    void*          m_mapped     = nullptr;
    u32            m_ssboSlot   = 0;

    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline       m_depthPipeline  = VK_NULL_HANDLE; // depth-tested
    VkPipeline       m_xrayPipeline   = VK_NULL_HANDLE; // always-on-top

    gfx::Device*              m_device              = nullptr;
    gfx::Allocator*           m_allocator           = nullptr;
    gfx::DescriptorAllocator* m_descriptorAllocator = nullptr;

    DynamicBodyFilter m_bodyFilter;

    static constexpr u32 kMaxLineVertices = 131072u; // 64k lines × 2

    VkPipeline buildPipeline(VkShaderModule vs, VkShaderModule fs,
                              bool depthTest, VkFormat colorFmt,
                              VkFormat depthFmt, VkDevice dev) const;
};

// ─────────────────────────────────────────────────────────────────────────────
#else // !JPH_DEBUG_RENDERER — Release stub, zero overhead

class PhysicsDebugRenderer {
public:
    void init(const PhysicsDebugInitInfo&) {}
    void destroy() {}
    void gather(JPH::PhysicsSystem&) {}
    void upload() {}
    void drawFrame(VkCommandBuffer, VkDescriptorSet, VkExtent2D, u32) {}
    bool enabled          = false;
    bool depthTestEnabled = true;
};

#endif // JPH_DEBUG_RENDERER

} // namespace enigma
