#include "physics/CrumpleZone.h"

#include <algorithm>

namespace enigma {

CrumpleZone CrumpleZone::makeDefault(std::string_view name, u32 vertexCount) {
    CrumpleZone zone;
    zone.name = std::string(name);
    zone.vertices.resize(vertexCount);

    // Determine zone type from name prefix.
    const bool isFront = name.starts_with("front");
    const bool isDoor  = name.starts_with("door");
    const bool isRear  = name.starts_with("rear");

    if (isFront) {
        for (u32 i = 0; i < vertexCount; ++i) {
            zone.vertices[i] = {
                .weight          = 0.9f,
                .maxDisplacement = 0.2f,
                .hardness        = 0.5f
            };
        }
    } else if (isDoor) {
        for (u32 i = 0; i < vertexCount; ++i) {
            zone.vertices[i] = {
                .weight          = 0.7f,
                .maxDisplacement = 0.12f,
                .hardness        = 0.6f
            };
        }
    } else if (isRear) {
        for (u32 i = 0; i < vertexCount; ++i) {
            zone.vertices[i] = {
                .weight          = 0.8f,
                .maxDisplacement = 0.15f,
                .hardness        = 0.55f
            };
        }
    } else {
        // Default: uniform low deformability.
        for (u32 i = 0; i < vertexCount; ++i) {
            zone.vertices[i] = {
                .weight          = 0.1f,
                .maxDisplacement = 0.03f,
                .hardness        = 0.9f
            };
        }
    }

    return zone;
}

} // namespace enigma
