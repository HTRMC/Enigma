#pragma once

#include "core/Math.h"
#include "core/Types.h"

#include <volk.h>

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
    u32  baseColorTextureSlot = 0; // bindless sampled image (binding 0)
    u32  samplerSlot          = 0; // bindless sampler (binding 3)
    vec4 baseColorFactor{1.0f, 1.0f, 1.0f, 1.0f};
};

struct MeshPrimitive {
    u32       vertexBufferSlot = 0;  // bindless SSBO slot (binding 2)
    u32       indexCount       = 0;
    VkBuffer  indexBuffer      = VK_NULL_HANDLE;
    i32       materialIndex    = -1; // -1 = default material
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

    std::vector<GpuBuffer>  ownedBuffers;
    std::vector<GpuImage>   ownedImages;
    std::vector<VkSampler>  ownedSamplers;

    void destroy(gfx::Device& device, gfx::Allocator& allocator);
};

} // namespace enigma
