#include "physics/PhysicsInterpolation.h"

#include "physics/PhysicsWorld.h"

namespace enigma {

void PhysicsInterpolation::snapshot(u32 bodyId, const PhysicsWorld& world) {
    // Find existing entry or create a new one.
    for (auto& entry : m_entries) {
        if (entry.bodyId == bodyId) {
            entry.prev = entry.curr;
            entry.curr.position = world.getPosition(bodyId);
            entry.curr.rotation = world.getRotation(bodyId);
            return;
        }
    }
    // New body: prev == curr (no interpolation on first frame).
    Entry e;
    e.bodyId        = bodyId;
    e.curr.position = world.getPosition(bodyId);
    e.curr.rotation = world.getRotation(bodyId);
    e.prev          = e.curr;
    m_entries.push_back(e);
}

mat4 PhysicsInterpolation::interpolatedTransform(u32 bodyId, f32 alpha) const {
    for (const auto& entry : m_entries) {
        if (entry.bodyId == bodyId) {
            const vec3 pos = glm::mix(entry.prev.position, entry.curr.position, alpha);
            const quat rot = glm::slerp(entry.prev.rotation, entry.curr.rotation, alpha);
            return glm::translate(mat4(1.0f), pos) * glm::mat4_cast(rot);
        }
    }
    // Unknown body — return identity.
    return mat4(1.0f);
}

} // namespace enigma
