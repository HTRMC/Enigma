#pragma once

#include "core/Types.h"
#include "core/Math.h"

#include <vector>

namespace enigma {

class PhysicsWorld;

// Interpolates between two physics states (prevState, currState) using alpha.
// alpha = accumulator / kFixedDt, in [0, 1].
class PhysicsInterpolation {
public:
    struct BodyState {
        vec3 position{0.0f};
        quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    };

    // Snapshot the current physics state for interpolation.
    void snapshot(u32 bodyId, const PhysicsWorld& world);

    // Get interpolated transform given alpha in [0,1].
    mat4 interpolatedTransform(u32 bodyId, f32 alpha) const;

private:
    struct Entry {
        u32       bodyId = 0;
        BodyState prev;
        BodyState curr;
    };
    std::vector<Entry> m_entries;
};

} // namespace enigma
