#pragma once

#include "core/Math.h"

namespace enigma {

struct Transform {
    vec3 position{0.0f, 0.0f, 0.0f};
    quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    vec3 scale{1.0f, 1.0f, 1.0f};

    mat4 toMatrix() const {
        const mat4 t = glm::translate(mat4{1.0f}, position);
        const mat4 r = glm::mat4_cast(rotation);
        const mat4 s = glm::scale(mat4{1.0f}, scale);
        return t * r * s;
    }
};

} // namespace enigma
