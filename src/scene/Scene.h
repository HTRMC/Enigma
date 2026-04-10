#pragma once

#include "core/Math.h"
#include "core/Types.h"
#include "gfx/AccelerationStructure.h"

#include <volk.h>

#include <optional>
#include <vector>

struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

namespace enigma::gfx {
class Allocator;
class Device;
} // namespace enigma::gfx

namespace enigma {

// CPU-side vertex layout for mesh loading. Packed into StructuredBuffer<float4>
// on GPU (3 float4s per vertex) to avoid std430 alignment issues with float3.
struct Vertex {
    vec3 position;
    vec3 normal;
    vec2 uv;
    vec4 tangent; // xyz = tangent direction, w = bitangent sign
};

struct Material {
    vec4 baseColorFactor{1.0f, 1.0f, 1.0f, 1.0f};
    vec4 emissiveFactor{0.0f, 0.0f, 0.0f, 0.0f}; // .w = alphaCutoff
    f32  metallicFactor     = 1.0f;
    f32  roughnessFactor    = 1.0f;
    f32  normalScale        = 1.0f;
    f32  occlusionStrength  = 1.0f;
    u32  baseColorTexIdx    = 0xFFFFFFFFu; // 0xFFFFFFFF = no texture
    u32  metalRoughTexIdx   = 0xFFFFFFFFu;
    u32  normalTexIdx       = 0xFFFFFFFFu;
    u32  emissiveTexIdx     = 0xFFFFFFFFu;
    u32  occlusionTexIdx    = 0xFFFFFFFFu;
    u32  flags              = 0u;          // bit0=BLEND, bit1=MASK
    u32  samplerSlot        = 0u;
    u32  _pad               = 0u;
};
static_assert(sizeof(Material) == 80, "Material must be 80 bytes for std430 SSBO layout");

struct MeshPrimitive {
    u32       vertexBufferSlot = 0;  // bindless SSBO slot (binding 2)
    u32       indexCount       = 0;
    VkBuffer  indexBuffer      = VK_NULL_HANDLE;
    i32       materialIndex    = -1; // -1 = default material
    // RT acceleration structure — built by Renderer after scene load.
    std::optional<gfx::BLAS> blas;
    // Vertex buffer handle + count needed for BLAS building.
    VkBuffer  vertexBuffer     = VK_NULL_HANDLE;
    u32       vertexCount      = 0;
};

struct MeshNode {
    mat4               worldTransform{1.0f};
    std::vector<u32>   primitiveIndices;
};

// Owns all GPU resources for a loaded scene. Must be destroyed before
// the Device and Allocator that created it.
struct Scene {
    std::vector<MeshPrimitive> primitives;
    std::vector<MeshNode>      nodes;
    std::vector<Material>      materials;

    // GPU resource ownership for cleanup.
    struct GpuBuffer {
        VkBuffer      buffer     = VK_NULL_HANDLE;
        VmaAllocation allocation = nullptr;
    };
    struct GpuImage {
        VkImage       image      = VK_NULL_HANDLE;
        VmaAllocation allocation = nullptr;
        VkImageView   view       = VK_NULL_HANDLE;
    };

    // Bindless SSBO slot for the packed material array (one entry per material).
    u32       materialBufferSlot = 0xFFFFFFFFu;
    GpuBuffer materialBuffer{};

    std::vector<GpuBuffer>  ownedBuffers;
    std::vector<GpuImage>   ownedImages;
    std::vector<VkSampler>  ownedSamplers;

    // Top-level acceleration structure for RT — built by Renderer.
    std::optional<gfx::TLAS> tlas;

    void destroy(gfx::Device& device, gfx::Allocator& allocator);
};

} // namespace enigma
