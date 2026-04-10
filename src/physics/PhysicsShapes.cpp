#include "physics/PhysicsShapes.h"

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>

namespace enigma::shapes {

JPH::Ref<JPH::Shape> makeBox(vec3 halfExtents) {
    JPH::BoxShapeSettings settings(JPH::Vec3(halfExtents.x, halfExtents.y, halfExtents.z));
    return settings.Create().Get();
}

JPH::Ref<JPH::Shape> makeSphere(f32 radius) {
    JPH::SphereShapeSettings settings(radius);
    return settings.Create().Get();
}

JPH::Ref<JPH::Shape> makeCapsule(f32 halfHeight, f32 radius) {
    JPH::CapsuleShapeSettings settings(halfHeight, radius);
    return settings.Create().Get();
}

JPH::Ref<JPH::Shape> makeConvexHull(std::span<const vec3> points) {
    // JPH::ConvexHullShapeSettings expects an array of JPH::Vec3.
    std::vector<JPH::Vec3> jphPoints;
    jphPoints.reserve(points.size());
    for (const auto& p : points) {
        jphPoints.emplace_back(p.x, p.y, p.z);
    }
    JPH::ConvexHullShapeSettings settings(jphPoints.data(), static_cast<int>(jphPoints.size()));
    return settings.Create().Get();
}

} // namespace enigma::shapes
