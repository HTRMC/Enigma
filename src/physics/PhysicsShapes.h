#pragma once

#include "core/Types.h"
#include "core/Math.h"

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Core/Reference.h>

#include <span>

namespace JPH {
    class Shape;
}

namespace enigma::shapes {

// Convenience wrappers for Jolt shape creation.
// Returns JPH::Ref<JPH::Shape> (reference-counted shape handles).
JPH::Ref<JPH::Shape> makeBox(vec3 halfExtents);
JPH::Ref<JPH::Shape> makeSphere(f32 radius);
JPH::Ref<JPH::Shape> makeCapsule(f32 halfHeight, f32 radius);
JPH::Ref<JPH::Shape> makeConvexHull(std::span<const vec3> points);

} // namespace enigma::shapes
