#include "asset/GltfLoader.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "core/Math.h"
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"
#include "gfx/UploadContext.h"
#include "renderer/MeshletBuilder.h"
#include "scene/Scene.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vk_mem_alloc.h>

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <fastgltf/glm_element_traits.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <climits>
#include <cstring>
#include <unordered_set>
#include <vector>

namespace enigma {

namespace {

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
    bufInfo.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                        | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                        | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
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
    bufInfo.usage       = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                        | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                        | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
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
                  const u8* pixels, u32 width, u32 height, Scene& scene,
                  bool srgb = true) {
    const VkDeviceSize texSize = static_cast<VkDeviceSize>(width) * height * 4;
    const VkFormat texFormat   = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;

    VkImageCreateInfo imgInfo{};
    imgInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType     = VK_IMAGE_TYPE_2D;
    imgInfo.format        = texFormat;
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
    viewInfo.format           = texFormat;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    ENIGMA_VK_CHECK(vkCreateImageView(device.logical(), &viewInfo, nullptr, &gpuImg.view));

    {
        gfx::UploadContext ctx(device, allocator);
        ctx.uploadImage(gpuImg.image, {width, height, 1}, texFormat,
                        pixels, texSize);
        ctx.submitAndWait();
    }

    u32 slot = descriptorAllocator.registerSampledImage(
        gpuImg.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    scene.ownedImages.push_back(gpuImg);
    return slot;
}

u32 createDefaultWhiteTexture(gfx::Device& device, gfx::Allocator& allocator,
                              gfx::DescriptorAllocator& descriptorAllocator, Scene& scene) {
    const u32 white = 0xFFFFFFFFu;
    return uploadTexture(device, allocator, descriptorAllocator,
                         reinterpret_cast<const u8*>(&white), 1, 1, scene);
}

u32 createDefaultSampler(gfx::Device& device,
                         gfx::DescriptorAllocator& descriptorAllocator, Scene& scene) {
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

// Load image pixel data via stb_image from a fastgltf image source.
u8* loadImagePixels(const fastgltf::Asset& asset, const fastgltf::Image& image,
                    const std::filesystem::path& baseDir, int* outW, int* outH) {
    u8* pixels = nullptr;
    int ch = 0;

    // Helper: decode from raw byte span.
    auto decodeFromMemory = [&](const u8* ptr, usize len) {
        if (len > static_cast<usize>(INT_MAX)) return;
        pixels = stbi_load_from_memory(ptr, static_cast<int>(len), outW, outH, &ch, 4);
    };

    // Helper: extract bytes from a buffer's DataSource at a given offset+length.
    auto readFromBuffer = [&](const fastgltf::Buffer& buffer, usize offset, usize length) {
        std::visit(fastgltf::visitor{
            [](auto&) {},
            [&](const fastgltf::sources::Vector& v) {
                decodeFromMemory(reinterpret_cast<const u8*>(v.bytes.data()) + offset, length);
            },
            [&](const fastgltf::sources::ByteView& v) {
                decodeFromMemory(reinterpret_cast<const u8*>(v.bytes.data()) + offset, length);
            },
            [&](const fastgltf::sources::Array& v) {
                decodeFromMemory(reinterpret_cast<const u8*>(v.bytes.data()) + offset, length);
            },
        }, buffer.data);
    };

    std::visit(fastgltf::visitor{
        [](auto&) {},
        [&](const fastgltf::sources::URI& uri) {
            const auto imgPath = std::filesystem::weakly_canonical(baseDir / uri.uri.fspath());
            const auto safeBase = std::filesystem::weakly_canonical(baseDir);
            if (imgPath.string().compare(0, safeBase.string().size(), safeBase.string()) != 0) {
                ENIGMA_LOG_WARN("[gltf] image URI escapes base directory: {}", uri.uri.string());
                return;
            }
            pixels = stbi_load(imgPath.string().c_str(), outW, outH, &ch, 4);
        },
        [&](const fastgltf::sources::Vector& vec) {
            decodeFromMemory(reinterpret_cast<const u8*>(vec.bytes.data()), vec.bytes.size());
        },
        [&](const fastgltf::sources::Array& arr) {
            decodeFromMemory(reinterpret_cast<const u8*>(arr.bytes.data()), arr.bytes.size());
        },
        [&](const fastgltf::sources::BufferView& bv) {
            const auto& view = asset.bufferViews[bv.bufferViewIndex];
            readFromBuffer(asset.buffers[view.bufferIndex], view.byteOffset, view.byteLength);
        },
    }, image.data);

    return pixels;
}

// Compute world transform for a node by walking up the tree.
mat4 computeWorldTransform(const fastgltf::Asset& asset, usize nodeIdx) {
    const auto& node = asset.nodes[nodeIdx];

    mat4 local{1.0f};
    if (const auto* trs = std::get_if<fastgltf::TRS>(&node.transform)) {
        const mat4 T = glm::translate(mat4{1.0f},
            vec3{trs->translation[0], trs->translation[1], trs->translation[2]});
        const quat Q{static_cast<f32>(trs->rotation[3]),
                     static_cast<f32>(trs->rotation[0]),
                     static_cast<f32>(trs->rotation[1]),
                     static_cast<f32>(trs->rotation[2])};
        const mat4 R = glm::mat4_cast(Q);
        const mat4 S = glm::scale(mat4{1.0f},
            vec3{trs->scale[0], trs->scale[1], trs->scale[2]});
        local = T * R * S;
    } else if (const auto* m = std::get_if<fastgltf::Node::TransformMatrix>(&node.transform)) {
        std::memcpy(&local, m->data(), sizeof(mat4));
    }

    return local;
}

void flattenNodes(const fastgltf::Asset& asset, usize nodeIdx, const mat4& parentTransform,
                  const std::vector<usize>& meshBasePrimitive, Scene& scene) {
    const auto& node = asset.nodes[nodeIdx];
    const mat4 world = parentTransform * computeWorldTransform(asset, nodeIdx);

    if (node.meshIndex.has_value()) {
        const usize meshIdx = node.meshIndex.value();
        MeshNode meshNode{};
        meshNode.worldTransform = world;

        const auto& mesh = asset.meshes[meshIdx];
        const usize basePrim = meshBasePrimitive[meshIdx];
        for (usize p = 0; p < mesh.primitives.size(); ++p) {
            if (basePrim + p < scene.primitives.size()) {
                meshNode.primitiveIndices.push_back(static_cast<u32>(basePrim + p));
            }
        }
        scene.nodes.push_back(std::move(meshNode));
    }

    for (usize child : node.children) {
        flattenNodes(asset, child, world, meshBasePrimitive, scene);
    }
}

} // namespace

std::optional<Scene> loadGltf(const std::filesystem::path& path,
                              gfx::Device& device,
                              gfx::Allocator& allocator,
                              gfx::DescriptorAllocator& descriptorAllocator) {
    ENIGMA_LOG_INFO("[gltf] loading: {}", path.string());

    fastgltf::Parser parser;
    fastgltf::GltfDataBuffer data;
    if (!data.loadFromFile(path)) {
        ENIGMA_LOG_ERROR("[gltf] failed to read file: {}", path.string());
        return std::nullopt;
    }

    const auto parentDir = path.parent_path();
    constexpr auto options = fastgltf::Options::LoadExternalBuffers
                           | fastgltf::Options::LoadGLBBuffers
                           | fastgltf::Options::GenerateMeshIndices;

    auto result = parser.loadGltf(&data, parentDir, options);
    if (result.error() != fastgltf::Error::None) {
        ENIGMA_LOG_ERROR("[gltf] parse error: {}", fastgltf::getErrorMessage(result.error()));
        return std::nullopt;
    }

    auto& asset = result.get();
    Scene scene{};
    const auto baseDir = path.parent_path();

    // Default resources.
    const u32 defaultTexSlot     = createDefaultWhiteTexture(device, allocator, descriptorAllocator, scene);
    const u32 defaultSamplerSlot = createDefaultSampler(device, descriptorAllocator, scene);

    // --- Identify linear (non-color) images before uploading ---
    // metalRoughness, normal, and occlusion textures must be uploaded as UNORM,
    // not SRGB, to avoid double gamma-correction in the shader.
    std::unordered_set<usize> linearImageIndices;
    for (const auto& mat : asset.materials) {
        auto markLinear = [&](const auto& texOpt) {
            if (texOpt.has_value()) {
                const auto& tex = asset.textures[texOpt->textureIndex];
                if (tex.imageIndex.has_value())
                    linearImageIndices.insert(tex.imageIndex.value());
            }
        };
        markLinear(mat.pbrData.metallicRoughnessTexture);
        markLinear(mat.normalTexture);
        markLinear(mat.occlusionTexture);
    }

    // --- Load textures ---
    std::vector<u32> imageSlots(asset.images.size(), defaultTexSlot);
    for (usize i = 0; i < asset.images.size(); ++i) {
        int w = 0, h = 0;
        u8* pixels = loadImagePixels(asset, asset.images[i], baseDir, &w, &h);
        if (pixels != nullptr) {
            const bool srgb = (linearImageIndices.find(i) == linearImageIndices.end());
            imageSlots[i] = uploadTexture(device, allocator, descriptorAllocator,
                                          pixels, static_cast<u32>(w), static_cast<u32>(h), scene, srgb);
            stbi_image_free(pixels);
            ENIGMA_LOG_INFO("[gltf] loaded texture {}: {}x{} ({})", i, w, h, srgb ? "srgb" : "linear");
        }
    }

    // --- Load materials ---
    // Helper: resolve a texture reference to a bindless image slot.
    auto resolveTexSlot = [&](const auto& texOpt) -> u32 {
        if (!texOpt.has_value()) return 0xFFFFFFFFu;
        const auto texIdx = texOpt->textureIndex;
        if (texIdx >= asset.textures.size()) return 0xFFFFFFFFu;
        const auto& tex = asset.textures[texIdx];
        if (!tex.imageIndex.has_value() || tex.imageIndex.value() >= imageSlots.size())
            return 0xFFFFFFFFu;
        return imageSlots[tex.imageIndex.value()];
    };

    for (const auto& mat : asset.materials) {
        Material material{};
        material.samplerSlot = defaultSamplerSlot;

        // Base color
        material.baseColorFactor = vec4{
            mat.pbrData.baseColorFactor[0], mat.pbrData.baseColorFactor[1],
            mat.pbrData.baseColorFactor[2], mat.pbrData.baseColorFactor[3]};
        material.baseColorTexIdx = resolveTexSlot(mat.pbrData.baseColorTexture);

        // Metallic-roughness (G=roughness, B=metallic per glTF spec)
        material.metallicFactor    = mat.pbrData.metallicFactor;
        material.roughnessFactor   = mat.pbrData.roughnessFactor;
        material.metalRoughTexIdx  = resolveTexSlot(mat.pbrData.metallicRoughnessTexture);

        // Normal map
        if (mat.normalTexture.has_value()) {
            material.normalScale   = mat.normalTexture->scale;
            material.normalTexIdx  = resolveTexSlot(mat.normalTexture);
        }

        // Occlusion
        if (mat.occlusionTexture.has_value()) {
            material.occlusionStrength = mat.occlusionTexture->strength;
            material.occlusionTexIdx   = resolveTexSlot(mat.occlusionTexture);
        }

        // Emissive
        material.emissiveFactor = vec4{
            mat.emissiveFactor[0], mat.emissiveFactor[1], mat.emissiveFactor[2],
            mat.alphaCutoff}; // store alphaCutoff in .w
        material.emissiveTexIdx = resolveTexSlot(mat.emissiveTexture);

        // Alpha mode → flags
        if (mat.alphaMode == fastgltf::AlphaMode::Blend) material.flags |= 1u;
        if (mat.alphaMode == fastgltf::AlphaMode::Mask)  material.flags |= 2u;

        scene.materials.push_back(material);
    }

    // Ensure at least one default material so shaders always have something to index.
    if (scene.materials.empty()) {
        Material def{};
        def.samplerSlot = defaultSamplerSlot;
        scene.materials.push_back(def);
    }

    // --- Upload material SSBO ---
    {
        const VkDeviceSize matBufSize = scene.materials.size() * sizeof(Material);
        scene.materialBuffer = createAndUploadSSBO(device, allocator,
                                                   scene.materials.data(), matBufSize);
        scene.materialBufferSlot = descriptorAllocator.registerStorageBuffer(
            scene.materialBuffer.buffer, matBufSize);
        ENIGMA_LOG_INFO("[gltf] uploaded material SSBO: {} materials ({} bytes)",
                        scene.materials.size(), matBufSize);
    }

    // --- Load mesh primitives ---
    std::vector<usize> meshBasePrimitive;
    for (const auto& mesh : asset.meshes) {
        meshBasePrimitive.push_back(scene.primitives.size());

        for (const auto& prim : mesh.primitives) {
            if (prim.type != fastgltf::PrimitiveType::Triangles) {
                ENIGMA_LOG_WARN("[gltf] skipping non-triangle primitive");
                continue;
            }

            auto posIt = prim.findAttribute("POSITION");
            if (posIt == prim.attributes.end()) {
                ENIGMA_LOG_WARN("[gltf] primitive missing POSITION, skipping");
                continue;
            }

            const auto& posAccessor = asset.accessors[posIt->second];
            const usize vertCount = posAccessor.count;
            std::vector<Vertex> vertices(vertCount);

            // Positions.
            fastgltf::iterateAccessorWithIndex<vec3>(asset, posAccessor,
                [&](vec3 p, usize i) { vertices[i].position = p; });

            // Normals.
            if (auto it = prim.findAttribute("NORMAL"); it != prim.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<vec3>(asset, asset.accessors[it->second],
                    [&](vec3 n, usize i) { vertices[i].normal = n; });
            } else {
                for (auto& v : vertices) v.normal = vec3{0.0f, 1.0f, 0.0f};
            }

            // UVs.
            if (auto it = prim.findAttribute("TEXCOORD_0"); it != prim.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<vec2>(asset, asset.accessors[it->second],
                    [&](vec2 uv, usize i) { vertices[i].uv = uv; });
            }

            // Tangents.
            if (auto it = prim.findAttribute("TANGENT"); it != prim.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<vec4>(asset, asset.accessors[it->second],
                    [&](vec4 t, usize i) { vertices[i].tangent = t; });
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
            if (prim.indicesAccessor.has_value()) {
                const auto& idxAccessor = asset.accessors[prim.indicesAccessor.value()];
                indices.resize(idxAccessor.count);
                fastgltf::iterateAccessorWithIndex<u32>(asset, idxAccessor,
                    [&](u32 idx, usize i) { indices[i] = idx; });
            } else {
                indices.resize(vertCount);
                for (usize i = 0; i < vertCount; ++i) indices[i] = static_cast<u32>(i);
            }

            const VkDeviceSize idxSize = indices.size() * sizeof(u32);
            auto idxBuf = createAndUploadIndexBuffer(device, allocator, indices.data(), idxSize);
            scene.ownedBuffers.push_back(idxBuf);

            // Build meshlets for visibility-buffer pipeline.
            std::vector<float> positions;
            positions.reserve(vertCount * 3);
            for (const auto& v : vertices) {
                positions.push_back(v.position.x);
                positions.push_back(v.position.y);
                positions.push_back(v.position.z);
            }
            MeshletData meshlets = MeshletBuilder::build(
                positions.data(), vertCount,
                indices.data(), indices.size());

            MeshPrimitive meshPrim{};
            meshPrim.vertexBufferSlot = ssboSlot;
            meshPrim.indexCount       = static_cast<u32>(indices.size());
            meshPrim.indexBuffer      = idxBuf.buffer;
            meshPrim.materialIndex    = prim.materialIndex.has_value()
                                            ? static_cast<i32>(prim.materialIndex.value())
                                            : -1;
            meshPrim.vertexBuffer     = gpuBuf.buffer;
            meshPrim.vertexCount      = static_cast<u32>(vertCount);
            meshPrim.meshlets         = std::move(meshlets);
            scene.primitives.push_back(meshPrim);
        }
    }

    // --- Flatten node hierarchy ---
    if (!asset.scenes.empty()) {
        const auto& defaultScene = asset.scenes[asset.defaultScene.value_or(0)];
        for (usize nodeIdx : defaultScene.nodeIndices) {
            flattenNodes(asset, nodeIdx, mat4{1.0f}, meshBasePrimitive, scene);
        }
    }

    ENIGMA_LOG_INFO("[gltf] loaded: {} primitives, {} nodes, {} materials, {} textures",
                    scene.primitives.size(), scene.nodes.size(),
                    scene.materials.size(), imageSlots.size());
    return scene;
}

} // namespace enigma
