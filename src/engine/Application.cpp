#include "engine/Application.h"

#include "asset/GltfLoader.h"
#include "core/Log.h"
#include "core/Paths.h"
#include "ecs/Ecs.h"
#include "ecs/systems/CameraSystem.h"
#include "ecs/systems/InputSystem.h"
#include "ecs/systems/PhysicsSystem.h"
#include "ecs/systems/TransformSystem.h"
#include "ecs/systems/VehicleSystem.h"
#include "engine/Engine.h"
#include "input/Input.h"
#include "physics/PhysicsWorld.h"
#include "physics/VehicleController.h"
#include "platform/Window.h"
#include "renderer/Renderer.h"
#include "scene/Camera.h"
#include "scene/FollowCamera.h"
#include "scene/Scene.h"
#include "world/CdlodTerrain.h"
#include "world/HeightmapLoader.h"

#include <volk.h>

#include <GLFW/glfw3.h>

#include <filesystem>
#include <optional>
#include <vector>

namespace enigma {

Application::Application()  = default;
Application::~Application() = default;

int Application::run(int argc, char** argv) {
    const char* argv0 = (argc > 0 && argv != nullptr) ? argv[0] : nullptr;
    Paths::init(argv0);
    ENIGMA_LOG_INFO("[app] starting, exe = {}", Paths::executablePath().string());

    Engine engine;
    auto& window   = engine.window();
    auto& renderer = engine.renderer();
    auto& clock    = engine.clock();
    auto& input    = engine.input();
    auto& world    = engine.world();

    // Camera — overwritten every PreRender tick by CameraSystem.
    Camera camera({0.0f, 3.0f, 8.0f}, 60.0f, 0.1f);
    renderer.setCamera(&camera);
    FollowCamera followCam(camera, /*armLength=*/8.0f, /*heightOffset=*/2.5f);

    // glTF scene loading: command-line path, BMW M4 GT3, or bundled default.
    std::optional<Scene> scene;
    std::filesystem::path modelPath;
    if (argc > 1 && argv[1] != nullptr) {
        modelPath = argv[1];
    } else {
        const auto bmwPath =
            Paths::executablePath().parent_path() / "assets" / "bmw_m4_gt3_nic.glb";
        modelPath = std::filesystem::exists(bmwPath)
                  ? bmwPath
                  : Paths::executablePath().parent_path() / "assets" / "DamagedHelmet.glb";
    }

    scene = loadGltf(modelPath,
                     renderer.device(),
                     renderer.allocator(),
                     renderer.descriptorAllocator());

    // Record per-node rest transforms with -90° Y correction (GLB +X → physics +Z).
    std::vector<mat4> nodeRestTransforms;
    if (scene.has_value()) {
        renderer.setScene(&scene.value());
        ENIGMA_LOG_INFO("[app] loaded scene from: {}", modelPath.string());
        const mat4 correction = glm::rotate(mat4(1.0f),
                                            glm::radians(-90.0f),
                                            vec3(0.0f, 1.0f, 0.0f));
        nodeRestTransforms.resize(scene->nodes.size());
        for (u32 i = 0; i < static_cast<u32>(scene->nodes.size()); ++i)
            nodeRestTransforms[i] = correction * scene->nodes[i].worldTransform;
        ENIGMA_LOG_INFO("[app] recorded {} node rest transforms (correction: -90° Y)",
                        nodeRestTransforms.size());
    } else {
        ENIGMA_LOG_WARN("[app] failed to load scene: {}, falling back to triangle",
                        modelPath.string());
    }

    // CDLOD terrain — Renderer::setScene() has already loaded the heightmap
    // and initialized CdlodTerrain; we just fetch the HeightmapLoader pointer
    // here for physics heightfield construction.
    const HeightmapLoader* hfHeightmap =
        renderer.cdlodTerrain() ? renderer.cdlodTerrain()->heightmap() : nullptr;

    // Build the physics heightfield directly from the loaded heightmap — the
    // same byte-for-byte samples that the mesh shader bakes into patch vertex
    // Y, so physics and visuals can never drift.
    u32 hfBodyId = ~0u;
    if (hfHeightmap != nullptr) {
        const u32  hfN      = hfHeightmap->sampleCount();
        const f32  hfWorld  = hfHeightmap->worldSize();
        const vec3 hfOrigin = hfHeightmap->origin();
        const auto& srcH    = hfHeightmap->heights(); // hfN*hfN floats
        std::vector<f32> heights(srcH.begin(), srcH.end());
        hfBodyId = engine.physics().addHeightField(hfOrigin, hfWorld, hfN, heights);
        ENIGMA_LOG_INFO(
            "[app] physics heightfield built from HeightmapLoader "
            "(origin=({:.1f},{:.1f},{:.1f}), size={:.1f}, samples={})",
            hfOrigin.x, hfOrigin.y, hfOrigin.z, hfWorld, hfN);
    } else {
        ENIGMA_LOG_WARN("[app] no heightmap available — physics heightfield skipped");
    }

    // ECS entity: the player vehicle.
    world.spawn(ecs::VehicleTag{1},
                ecs::VehicleControls{},
                ecs::PhysicsBody{engine.vehicle()->bodyId()});

    // ECS systems — registered in execution order within each schedule group.
    // PrePhysics: gather input.
    world.add_system(ecs::SystemSchedule::PrePhysics,
        ecs::systems::makeInputSystem(input, world));

    // Physics: apply vehicle input, then run the fixed-step loop (Sequential).
    world.add_system(ecs::SystemSchedule::Physics,
        ecs::systems::makeVehicleSystem(*engine.vehicle(), world));
    world.add_system(ecs::SystemSchedule::Physics,
        ecs::systems::makePhysicsSystem(engine.physics(),
                                        engine.interpolation(),
                                        *engine.vehicle()),
        ecs::ExecutionPolicy::Sequential);

    // PostPhysics: propagate interpolated transform to scene nodes + stream HF.
    world.add_system(ecs::SystemSchedule::PostPhysics,
        ecs::systems::makeTransformSystem(engine.physics(),
                                          engine.interpolation(),
                                          *engine.vehicle(),
                                          scene.has_value() ? &scene.value() : nullptr,
                                          nodeRestTransforms));
    // CDLOD terrain traversal is driven by Renderer::drawFrame() per frame —
    // no ECS system is required. Physics heightfield streaming is similarly
    // unnecessary with a single 4 km tile covering the playable area.
    (void)hfBodyId;

    // PreRender: update follow camera + physics debug overlay toggle.
    world.add_system(ecs::SystemSchedule::PreRender,
        ecs::systems::makeCameraSystem(followCam,
                                       engine.physics(),
                                       engine.interpolation(),
                                       *engine.vehicle()));

    bool f3PrevDown = false;
    world.add_system(ecs::SystemSchedule::PreRender, [&](float) {
        const bool f3Down = input.isKeyDown(GLFW_KEY_F3);
        if (f3Down && !f3PrevDown)
            renderer.physicsDebugRenderer().enabled =
                !renderer.physicsDebugRenderer().enabled;
        f3PrevDown = f3Down;
        if (renderer.physicsDebugRenderer().enabled)
            renderer.physicsDebugRenderer().gather(engine.physics().system());
    });

    // -------------------------------------------------------------------------
    // Frame loop (AC 3.2: body < 30 lines).
    // -------------------------------------------------------------------------
    while (!window.shouldClose()) {
        window.pollEvents();
        const f32 dt = static_cast<f32>(clock.tick());
        input.update();
        world.run_systems(dt);
        renderer.drawFrame();
    }

    // Clean up scene before renderer teardown. CDLOD terrain is owned by the
    // renderer and released inside its destructor.
    if (scene.has_value()) {
        renderer.setScene(nullptr);
        vkDeviceWaitIdle(renderer.device().logical());
        scene->destroy(renderer.device(), renderer.allocator());
    }

    ENIGMA_LOG_INFO("[app] shutdown");
    return 0;
}

} // namespace enigma
