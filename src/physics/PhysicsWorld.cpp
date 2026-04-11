#include "physics/PhysicsWorld.h"

#include "core/Log.h"

// Jolt includes — order matters: Jolt.h must come first.
#include <Jolt/Jolt.h>

#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>  // TempAllocatorMalloc
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayerInterfaceTable.h>
#include <Jolt/Physics/Collision/BroadPhase/ObjectVsBroadPhaseLayerFilterTable.h>
#include <Jolt/Physics/Collision/ObjectLayerPairFilterTable.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/PlaneShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>

#include <cstdarg>

// Jolt requires a trace function and an assert-fail handler to be defined
// by the application. These are free functions in the JPH namespace.
static void joltTraceImpl(const char* fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    enigma::log::write(enigma::log::Level::Trace, "[jolt] {}", buf);
}

#ifdef JPH_ENABLE_ASSERTS
static bool joltAssertFailed(const char* expression, const char* message, const char* file, JPH::uint line) {
    enigma::log::write(enigma::log::Level::Error,
                       "[jolt] assert failed: {} — {} ({}:{})", expression, message ? message : "", file, line);
    return true; // break into debugger
}
#endif

namespace enigma {

namespace {

// One-time Jolt initialization guard.
struct JoltInit {
    JoltInit() {
        JPH::RegisterDefaultAllocator();
        JPH::Trace = joltTraceImpl;
#ifdef JPH_ENABLE_ASSERTS
        JPH::AssertFailed = joltAssertFailed;
#endif
        JPH::Factory::sInstance = new JPH::Factory();
        JPH::RegisterTypes();
    }
    ~JoltInit() {
        JPH::UnregisterTypes();
        delete JPH::Factory::sInstance;
        JPH::Factory::sInstance = nullptr;
    }
};

void ensureJoltInit() {
    static JoltInit s_init;
}

} // namespace

PhysicsWorld::PhysicsWorld() {
    ensureJoltInit();

    m_tempAllocator = std::make_unique<JPH::TempAllocatorMalloc>();
    m_jobSystem     = std::make_unique<JPH::JobSystemThreadPool>(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, 2);

    // Broad-phase layer interface: maps ObjectLayer → BroadPhaseLayer.
    m_bpLayerInterface = std::make_unique<JPH::BroadPhaseLayerInterfaceTable>(PhysicsLayer::Count, BPLayer::Count);
    m_bpLayerInterface->MapObjectToBroadPhaseLayer(PhysicsLayer::Static,  JPH::BroadPhaseLayer(BPLayer::Static));
    m_bpLayerInterface->MapObjectToBroadPhaseLayer(PhysicsLayer::Dynamic, JPH::BroadPhaseLayer(BPLayer::Dynamic));
    m_bpLayerInterface->MapObjectToBroadPhaseLayer(PhysicsLayer::Vehicle, JPH::BroadPhaseLayer(BPLayer::Dynamic));

    // Object-layer-pair filter: which object layers can collide with each other.
    m_objLayerFilter = std::make_unique<JPH::ObjectLayerPairFilterTable>(PhysicsLayer::Count);
    m_objLayerFilter->EnableCollision(PhysicsLayer::Static,  PhysicsLayer::Dynamic);
    m_objLayerFilter->EnableCollision(PhysicsLayer::Static,  PhysicsLayer::Vehicle);
    m_objLayerFilter->EnableCollision(PhysicsLayer::Dynamic, PhysicsLayer::Dynamic);
    m_objLayerFilter->EnableCollision(PhysicsLayer::Dynamic, PhysicsLayer::Vehicle);
    m_objLayerFilter->EnableCollision(PhysicsLayer::Vehicle, PhysicsLayer::Vehicle);

    // Object-vs-broadphase filter (must be created after objLayerFilter).
    m_objBpFilter = std::make_unique<JPH::ObjectVsBroadPhaseLayerFilterTable>(
        *m_bpLayerInterface, BPLayer::Count, *m_objLayerFilter, PhysicsLayer::Count);
    // Static ↔ Static: no collision (default).

    constexpr JPH::uint cMaxBodies          = 65536;
    constexpr JPH::uint cNumBodyMutexes     = 0; // auto
    constexpr JPH::uint cMaxBodyPairs       = 65536;
    constexpr JPH::uint cMaxContactConstraints = 65536;

    m_physicsSystem = std::make_unique<JPH::PhysicsSystem>();
    m_physicsSystem->Init(cMaxBodies, cNumBodyMutexes, cMaxBodyPairs, cMaxContactConstraints,
                          *m_bpLayerInterface, *m_objBpFilter, *m_objLayerFilter);

    // Default gravity.
    m_physicsSystem->SetGravity(JPH::Vec3(0.0f, -9.81f, 0.0f));

    ENIGMA_LOG_INFO("[physics] world created (max bodies={}, 120 Hz fixed step)", cMaxBodies);
}

PhysicsWorld::~PhysicsWorld() {
    ENIGMA_LOG_INFO("[physics] world destroyed");
}

void PhysicsWorld::step(f32 dt) {
    // Clamp to avoid a spiral of death when dt is large (e.g. first frame
    // after a long load). 5 physics steps per render frame is the max.
    constexpr f32 kMaxDt = kFixedDt * 5.0f;
    m_accumulator += (dt < kMaxDt ? dt : kMaxDt);
    while (m_accumulator >= kFixedDt) {
        m_physicsSystem->Update(kFixedDt, 1, m_tempAllocator.get(), m_jobSystem.get());
        m_accumulator -= kFixedDt;
    }
}

u32 PhysicsWorld::addStaticBox(vec3 position, vec3 halfExtents) {
    auto& bi = m_physicsSystem->GetBodyInterface();
    JPH::BoxShapeSettings shapeSettings(JPH::Vec3(halfExtents.x, halfExtents.y, halfExtents.z));
    JPH::BodyCreationSettings bodySettings(
        shapeSettings.Create().Get(),
        JPH::RVec3(position.x, position.y, position.z),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Static,
        PhysicsLayer::Static);
    JPH::Body* body = bi.CreateBody(bodySettings);
    bi.AddBody(body->GetID(), JPH::EActivation::DontActivate);
    return body->GetID().GetIndexAndSequenceNumber();
}

u32 PhysicsWorld::addStaticPlane(vec3 normal, f32 offset) {
    auto& bi = m_physicsSystem->GetBodyInterface();
    JPH::PlaneShapeSettings shapeSettings(
        JPH::Plane(JPH::Vec3(normal.x, normal.y, normal.z), offset));
    JPH::BodyCreationSettings bodySettings(
        shapeSettings.Create().Get(),
        JPH::RVec3::sZero(),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Static,
        PhysicsLayer::Static);
    JPH::Body* body = bi.CreateBody(bodySettings);
    bi.AddBody(body->GetID(), JPH::EActivation::DontActivate);
    return body->GetID().GetIndexAndSequenceNumber();
}

u32 PhysicsWorld::addDynamicBox(vec3 position, vec3 halfExtents, f32 mass) {
    auto& bi = m_physicsSystem->GetBodyInterface();
    JPH::BoxShapeSettings shapeSettings(JPH::Vec3(halfExtents.x, halfExtents.y, halfExtents.z));
    JPH::BodyCreationSettings bodySettings(
        shapeSettings.Create().Get(),
        JPH::RVec3(position.x, position.y, position.z),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        PhysicsLayer::Dynamic);
    bodySettings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
    bodySettings.mMassPropertiesOverride.mMass = mass;
    JPH::Body* body = bi.CreateBody(bodySettings);
    bi.AddBody(body->GetID(), JPH::EActivation::Activate);
    return body->GetID().GetIndexAndSequenceNumber();
}

u32 PhysicsWorld::addDynamicSphere(vec3 position, f32 radius, f32 mass) {
    auto& bi = m_physicsSystem->GetBodyInterface();
    JPH::SphereShapeSettings shapeSettings(radius);
    JPH::BodyCreationSettings bodySettings(
        shapeSettings.Create().Get(),
        JPH::RVec3(position.x, position.y, position.z),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        PhysicsLayer::Dynamic);
    bodySettings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
    bodySettings.mMassPropertiesOverride.mMass = mass;
    JPH::Body* body = bi.CreateBody(bodySettings);
    bi.AddBody(body->GetID(), JPH::EActivation::Activate);
    return body->GetID().GetIndexAndSequenceNumber();
}

void PhysicsWorld::removeBody(u32 bodyId) {
    auto& bi = m_physicsSystem->GetBodyInterface();
    JPH::BodyID id(bodyId);
    bi.RemoveBody(id);
    bi.DestroyBody(id);
}

vec3 PhysicsWorld::getPosition(u32 bodyId) const {
    auto& bi = m_physicsSystem->GetBodyInterface();
    JPH::RVec3 p = bi.GetPosition(JPH::BodyID(bodyId));
    return {static_cast<f32>(p.GetX()), static_cast<f32>(p.GetY()), static_cast<f32>(p.GetZ())};
}

quat PhysicsWorld::getRotation(u32 bodyId) const {
    auto& bi = m_physicsSystem->GetBodyInterface();
    JPH::Quat q = bi.GetRotation(JPH::BodyID(bodyId));
    return quat(q.GetW(), q.GetX(), q.GetY(), q.GetZ());
}

mat4 PhysicsWorld::getWorldTransform(u32 bodyId) const {
    auto& bi = m_physicsSystem->GetBodyInterface();
    JPH::RMat44 jm = bi.GetWorldTransform(JPH::BodyID(bodyId));
    mat4 result;
    // Jolt stores columns: GetColumn4(n) returns the n-th column.
    for (int col = 0; col < 4; ++col) {
        JPH::Vec4 c = jm.GetColumn4(col);
        result[col] = vec4(c.GetX(), c.GetY(), c.GetZ(), c.GetW());
    }
    return result;
}

void PhysicsWorld::applyImpulse(u32 bodyId, vec3 impulse) {
    auto& bi = m_physicsSystem->GetBodyInterface();
    bi.AddImpulse(JPH::BodyID(bodyId), JPH::Vec3(impulse.x, impulse.y, impulse.z));
}

void PhysicsWorld::applyAngularImpulse(u32 bodyId, vec3 angularImpulse) {
    auto& bi = m_physicsSystem->GetBodyInterface();
    bi.AddAngularImpulse(JPH::BodyID(bodyId),
                         JPH::Vec3(angularImpulse.x, angularImpulse.y, angularImpulse.z));
}

void PhysicsWorld::setLinearVelocity(u32 bodyId, vec3 velocity) {
    auto& bi = m_physicsSystem->GetBodyInterface();
    bi.SetLinearVelocity(JPH::BodyID(bodyId), JPH::Vec3(velocity.x, velocity.y, velocity.z));
}

vec3 PhysicsWorld::getLinearVelocity(u32 bodyId) const {
    auto& bi = m_physicsSystem->GetBodyInterface();
    JPH::Vec3 v = bi.GetLinearVelocity(JPH::BodyID(bodyId));
    return {v.GetX(), v.GetY(), v.GetZ()};
}

vec3 PhysicsWorld::getAngularVelocity(u32 bodyId) const {
    auto& bi = m_physicsSystem->GetBodyInterface();
    JPH::Vec3 v = bi.GetAngularVelocity(JPH::BodyID(bodyId));
    return {v.GetX(), v.GetY(), v.GetZ()};
}

} // namespace enigma
