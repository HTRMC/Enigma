#pragma once

#include <volk.h>

namespace enigma::gbuf {

static constexpr VkFormat kAlbedoFormat     = VK_FORMAT_R8G8B8A8_UNORM;
static constexpr VkFormat kNormalFormat     = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
static constexpr VkFormat kMetalRoughFormat = VK_FORMAT_R8G8_UNORM;
static constexpr VkFormat kMotionVecFormat  = VK_FORMAT_R16G16_SFLOAT;
static constexpr VkFormat kDepthFormat      = VK_FORMAT_D32_SFLOAT;

} // namespace enigma::gbuf
