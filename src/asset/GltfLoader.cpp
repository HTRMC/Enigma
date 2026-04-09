#include "asset/GltfLoader.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "core/Math.h"
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"
#include "gfx/UploadContext.h"
#include "scene/Scene.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vk_mem_alloc.h>

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace enigma {

namespace {

// Pack vertices into float4 triplets for GPU (matches mesh.hlsl layout).
std::vector<vec4> packVertices(const std::vector<Vertex>& verts) {
    std::vector<vec4> packed;
    packed.reserve(verts.size() * 3);
    for (const auto& v : verts) {
        packed.emplace_back(v.position.x, v.position.y, v.position.z, v.normal.x);
        packed.emplace_back(v.normal.y, v.normal.z, v.uv.x, v.uv.y);
        packed.emplace_back(v.tangent);
    }
    return packed;
}

Scene::GpuBuffer createAndUploadSSBO(gfx::Device& device, gfx::Allocator& allocator,
                                      const void* data, VkDeviceSize size) {
    VkBufferCreateInfo bufInfo{};
    bufInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size        = size;
    bufInfo.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    Scene::GpuBuffer result{};
    ENIGMA_VK_CHECK(vmaCreateBuffer(allocator.handle(), &bufInfo, &allocInfo,
                                    &result.buffer, &result.allocation, nullptr));

    gfx::UploadContext ctx(device, allocator);
    ctx.uploadBuffer(result.buffer, data, size);
    ctx.submitAndWait();
    return result;
}

Scene::GpuBuffer createAndUploadIndexBuffer(gfx::Device& device, gfx::Allocator& allocator,
                                             const void* data, VkDeviceSize size) {
    VkBufferCreateInfo bufInfo{};
    bufInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size        = size;
    bufInfo.usage       = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    Scene::GpuBuffer result{};
    ENIGMA_VK_CHECK(vmaCreateBuffer(allocator.handle(), &bufInfo, &allocInfo,
                                    &result.buffer, &result.allocation, nullptr));

    gfx::UploadContext ctx(device, allocator);
    ctx.uploadBuffer(result.buffer, data, size);
    ctx.submitAndWait();
    return result;
}

u32 uploadTexture(gfx::Device& device, gfx::Allocator& allocator,
                  gfx::DescriptorAllocator& descriptorAllocator,
                  const u8* pixels, u32 width, u32 height,
                  Scene& scene) {
    const VkDeviceSize texSize = static_cast<VkDeviceSize>(width) * height * 4;

    VkImageCreateInfo imgInfo{};
    imgInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType     = VK_IMAGE_TYPE_2D;
    imgInfo.format        = VK_FORMAT_R8G8B8A8_SRGB;
    imgInfo.extent        = {width, height, 1};
    imgInfo.mipLevels     = 1;
    imgInfo.arrayLayers   = 1;
    imgInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imgInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo imgAllocInfo{};
    imgAllocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    Scene::GpuImage gpuImg{};
    ENIGMA_VK_CHECK(vmaCreateImage(allocator.handle(), &imgInfo, &imgAllocInfo,
                                   &gpuImg.image, &gpuImg.allocation, nullptr));

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image            = gpuImg.image;
    viewInfo.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format           = VK_FORMAT_R8G8B8A8_SRGB;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    ENIGMA_VK_CHECK(vkCreateImageView(device.logical(), &viewInfo, nullptr, &gpuImg.view));

    {
        gfx::UploadContext ctx(device, allocator);
        ctx.uploadImage(gpuImg.image, {width, height, 1}, VK_FORMAT_R8G8B8A8_SRGB,
                        pixels, texSize);
        ctx.submitAndWait();
    }

    u32 slot = descriptorAllocator.registerSampledImage(
        gpuImg.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    scene.ownedImages.push_back(gpuImg);
    return slot;
}

u32 createDefaultWhiteTexture(gfx::Device& device, gfx::Allocator& allocator,
                              gfx::DescriptorAllocator& descriptorAllocator,
                              Scene& scene) {
    const u32 white = 0xFFFFFFFFu;
    return uploadTexture(device, allocator, descriptorAllocator,
                         reinterpret_cast<const u8*>(&white), 1, 1, scene);
}

u32 createDefaultSampler(gfx::Device& device,
                         gfx::DescriptorAllocator& descriptorAllocator,
                         Scene& scene) {
    const f32 maxAniso = device.properties().limits.maxSamplerAnisotropy;

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter               = VK_FILTER_LINEAR;
    samplerInfo.minFilter               = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable        = VK_TRUE;
    samplerInfo.maxAnisotropy           = std::min(maxAniso, 16.0f);
    samplerInfo.borderColor             = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;

    VkSampler sampler = VK_NULL_HANDLE;
    ENIGMA_VK_CHECK(vkCreateSampler(device.logical(), &samplerInfo, nullptr, &sampler));

    u32 slot = descriptorAllocator.registerSampler(sampler);
    scene.ownedSamplers.push_back(sampler);
    return slot;
}

// Compute the world transform for a node by walking up the hierarchy.
mat4 computeWorldTransform(const cgltf_node* node) {
    f32 m[16];
    cgltf_node_transform_world(node, m);
    // cgltf outputs column-major, same as glm.
    mat4 result;
    std::memcpy(&result, m, sizeof(mat4));
    return result;
}

// Load an image referenced by a cgltf_image. Returns pixel data (RGBA8) via stb_image.
u8* loadImage(const cgltf_image* image, const std::filesystem::path& baseDir,
              int* outW, int* outH) {
    // Embedded buffer view.
    if (image->buffer_view != nullptr) {
        const cgltf_buffer_view* bv = image->buffer_view;
        // Guard against integer truncation: stbi takes int length.
        if (bv->size > static_cast<cgltf_size>(INT_MAX)) {
            ENIGMA_LOG_WARN("[gltf] image buffer view too large ({} bytes), skipping", bv->size);
            return nullptr;
        }
        const u8* ptr = static_cast<const u8*>(bv->buffer->data) + bv->offset;
        int ch = 0;
        return stbi_load_from_memory(ptr, static_cast<int>(bv->size), outW, outH, &ch, 4);
    }
    // External URI — canonicalize and verify path stays within baseDir
    // to prevent path traversal via crafted glTF files.
    if (image->uri != nullptr) {
        const auto imgPath = std::filesystem::weakly_canonical(baseDir / image->uri);
        const auto safeBase = std::filesystem::weakly_canonical(baseDir);
        const auto imgStr = imgPath.string();
        const auto baseStr = safeBase.string();
        if (imgStr.size() < baseStr.size() ||
            imgStr.compare(0, baseStr.size(), baseStr) != 0) {
            ENIGMA_LOG_WARN("[gltf] image URI escapes base directory: {}", image->uri);
            return nullptr;
        }
        int ch = 0;
        return stbi_load(imgPath.string().c_str(), outW, outH, &ch, 4);
    }
    return nullptr;
}

} // namespace

std::optional<Scene> loadGltf(const std::filesystem::path& path,
                              gfx::Device& device,
                              gfx::Allocator& allocator,
                              gfx::DescriptorAllocator& descriptorAllocator) {
    ENIGMA_LOG_INFO("[gltf] loading: {}", path.string());

    cgltf_options options{};
    cgltf_data* data = nullptr;
    cgltf_result parseResult = cgltf_parse_file(&options, path.string().c_str(), &data);
    if (parseResult != cgltf_result_success) {
        ENIGMA_LOG_ERROR("[gltf] failed to parse: {}", path.string());
        return std::nullopt;
    }

    cgltf_result loadResult = cgltf_load_buffers(&options, data, path.string().c_str());
    if (loadResult != cgltf_result_success) {
        ENIGMA_LOG_ERROR("[gltf] failed to load buffers: {}", path.string());
        cgltf_free(data);
        return std::nullopt;
    }

    const auto baseDir = path.parent_path();
    Scene scene{};

    // Default resources.
    const u32 defaultTexSlot     = createDefaultWhiteTexture(device, allocator, descriptorAllocator, scene);
    const u32 defaultSamplerSlot = createDefaultSampler(device, descriptorAllocator, scene);

    // --- Load textures ---
    std::vector<u32> imageSlots(data->images_count, defaultTexSlot);
    for (cgltf_size i = 0; i < data->images_count; ++i) {
        int w = 0, h = 0;
        u8* pixels = loadImage(&data->images[i], baseDir, &w, &h);
        if (pixels != nullptr) {
            imageSlots[i] = uploadTexture(device, allocator, descriptorAllocator,
                                          pixels, static_cast<u32>(w), static_cast<u32>(h), scene);
            stbi_image_free(pixels);
            ENIGMA_LOG_INFO("[gltf] loaded texture {}: {}x{}", i, w, h);
        }
    }

    // --- Load materials ---
    for (cgltf_size i = 0; i < data->materials_count; ++i) {
        const cgltf_material* mat = &data->materials[i];
        Material material{};
        material.baseColorTextureSlot = defaultTexSlot;
        material.samplerSlot          = defaultSamplerSlot;

        if (mat->has_pbr_metallic_roughness) {
            const auto& pbr = mat->pbr_metallic_roughness;
            const auto* f = pbr.base_color_factor;
            material.baseColorFactor = vec4{f[0], f[1], f[2], f[3]};

            if (pbr.base_color_texture.texture != nullptr) {
                const cgltf_texture* tex = pbr.base_color_texture.texture;
                if (tex->image != nullptr) {
                    const cgltf_size imgIdx = static_cast<cgltf_size>(tex->image - data->images);
                    if (imgIdx < data->images_count) {
                        material.baseColorTextureSlot = imageSlots[imgIdx];
                    }
                }
            }
        }

        scene.materials.push_back(material);
    }

    // --- Load mesh primitives ---
    // Track base primitive index per mesh for node lookups.
    std::vector<usize> meshBasePrimitive;
    for (cgltf_size mi = 0; mi < data->meshes_count; ++mi) {
        meshBasePrimitive.push_back(scene.primitives.size());
        const cgltf_mesh* mesh = &data->meshes[mi];

        for (cgltf_size pi = 0; pi < mesh->primitives_count; ++pi) {
            const cgltf_primitive* prim = &mesh->primitives[pi];
            if (prim->type != cgltf_primitive_type_triangles) {
                ENIGMA_LOG_WARN("[gltf] skipping non-triangle primitive");
                continue;
            }

            // Find accessors.
            const cgltf_accessor* posAcc  = nullptr;
            const cgltf_accessor* normAcc = nullptr;
            const cgltf_accessor* uvAcc   = nullptr;
            const cgltf_accessor* tanAcc  = nullptr;
            for (cgltf_size ai = 0; ai < prim->attributes_count; ++ai) {
                const cgltf_attribute* attr = &prim->attributes[ai];
                switch (attr->type) {
                    case cgltf_attribute_type_position: posAcc  = attr->data; break;
                    case cgltf_attribute_type_normal:   normAcc = attr->data; break;
                    case cgltf_attribute_type_texcoord: uvAcc   = attr->data; break;
                    case cgltf_attribute_type_tangent:  tanAcc  = attr->data; break;
                    default: break;
                }
            }
            if (posAcc == nullptr) {
                ENIGMA_LOG_WARN("[gltf] primitive missing POSITION, skipping");
                continue;
            }

            const usize vertCount = posAcc->count;
            std::vector<Vertex> vertices(vertCount);

            // Positions.
            for (usize v = 0; v < vertCount; ++v) {
                f32 buf[3] = {};
                cgltf_accessor_read_float(posAcc, v, buf, 3);
                vertices[v].position = vec3{buf[0], buf[1], buf[2]};
            }
            // Normals.
            if (normAcc != nullptr) {
                for (usize v = 0; v < vertCount; ++v) {
                    f32 buf[3] = {};
                    cgltf_accessor_read_float(normAcc, v, buf, 3);
                    vertices[v].normal = vec3{buf[0], buf[1], buf[2]};
                }
            } else {
                for (auto& v : vertices) v.normal = vec3{0.0f, 1.0f, 0.0f};
            }
            // UVs.
            if (uvAcc != nullptr) {
                for (usize v = 0; v < vertCount; ++v) {
                    f32 buf[2] = {};
                    cgltf_accessor_read_float(uvAcc, v, buf, 2);
                    vertices[v].uv = vec2{buf[0], buf[1]};
                }
            }
            // Tangents.
            if (tanAcc != nullptr) {
                for (usize v = 0; v < vertCount; ++v) {
                    f32 buf[4] = {};
                    cgltf_accessor_read_float(tanAcc, v, buf, 4);
                    vertices[v].tangent = vec4{buf[0], buf[1], buf[2], buf[3]};
                }
            } else {
                for (auto& v : vertices) v.tangent = vec4{1.0f, 0.0f, 0.0f, 1.0f};
            }

            // Pack and upload vertex SSBO.
            const auto packed = packVertices(vertices);
            const VkDeviceSize ssboSize = packed.size() * sizeof(vec4);
            auto gpuBuf = createAndUploadSSBO(device, allocator, packed.data(), ssboSize);
            u32 ssboSlot = descriptorAllocator.registerStorageBuffer(gpuBuf.buffer, ssboSize);
            scene.ownedBuffers.push_back(gpuBuf);

            // Indices.
            std::vector<u32> indices;
            if (prim->indices != nullptr) {
                indices.resize(prim->indices->count);
                for (usize idx = 0; idx < prim->indices->count; ++idx) {
                    indices[idx] = static_cast<u32>(cgltf_accessor_read_index(prim->indices, idx));
                }
            } else {
                indices.resize(vertCount);
                for (usize idx = 0; idx < vertCount; ++idx) indices[idx] = static_cast<u32>(idx);
            }

            const VkDeviceSize idxSize = indices.size() * sizeof(u32);
            auto idxBuf = createAndUploadIndexBuffer(device, allocator, indices.data(), idxSize);
            scene.ownedBuffers.push_back(idxBuf);

            MeshPrimitive meshPrim{};
            meshPrim.vertexBufferSlot = ssboSlot;
            meshPrim.indexCount       = static_cast<u32>(indices.size());
            meshPrim.indexBuffer      = idxBuf.buffer;
            meshPrim.materialIndex    = prim->material != nullptr
                                            ? static_cast<i32>(prim->material - data->materials)
                                            : -1;
            scene.primitives.push_back(meshPrim);
        }
    }

    // --- Flatten node hierarchy ---
    for (cgltf_size ni = 0; ni < data->nodes_count; ++ni) {
        const cgltf_node* node = &data->nodes[ni];
        if (node->mesh == nullptr) continue;

        const cgltf_size meshIdx = static_cast<cgltf_size>(node->mesh - data->meshes);
        if (meshIdx >= data->meshes_count) continue;

        MeshNode meshNode{};
        meshNode.worldTransform = computeWorldTransform(node);

        const usize basePrim = meshBasePrimitive[meshIdx];
        const usize primCount = node->mesh->primitives_count;
        for (usize p = 0; p < primCount; ++p) {
            if (basePrim + p < scene.primitives.size()) {
                meshNode.primitiveIndices.push_back(static_cast<u32>(basePrim + p));
            }
        }
        scene.nodes.push_back(std::move(meshNode));
    }

    cgltf_free(data);

    ENIGMA_LOG_INFO("[gltf] loaded: {} primitives, {} nodes, {} materials, {} images",
                    scene.primitives.size(), scene.nodes.size(),
                    scene.materials.size(), imageSlots.size());
    return scene;
}

} // namespace enigma
