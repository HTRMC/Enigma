#include "engine/Application.h"

#include "asset/GltfLoader.h"
#include "core/Log.h"
#include "core/Paths.h"
#include "ecs/Ecs.h"
#include "ecs/systems/CameraSystem.h"
#include "ecs/systems/InputSystem.h"
#include "ecs/systems/PhysicsSystem.h"
#include "ecs/systems/TerrainSystem.h"
#include "ecs/systems/TransformSystem.h"
#include "ecs/systems/VehicleSystem.h"
#include "engine/Engine.h"
#include "input/Input.h"
#include "physics/PhysicsWorld.h"
#include "physics/VehicleController.h"
#include "platform/Window.h"
#include "renderer/GBufferFormats.h"
#include "renderer/Renderer.h"
#include "scene/Camera.h"
#include "scene/FollowCamera.h"
#include "scene/Scene.h"
#include "world/Terrain.h"

#include <volk.h>

#include <GLFW/glfw3.h>

#include <cmath>
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

    // GPU-driven clipmap terrain.
    Terrain terrain(renderer.device(),
                    renderer.allocator(),
                    renderer.descriptorAllocator());
    terrain.buildPipeline(renderer.shaderManager(),
                          renderer.descriptorAllocator().layout(),
                          gbuf::kAlbedoFormat,
                          gbuf::kDepthFormat,
                          gbuf::kNormalFormat,
                          gbuf::kMetalRoughFormat,
                          gbuf::kMotionVecFormat);
    terrain.registerHotReload(renderer.shaderHotReload());
    renderer.setTerrain(&terrain);

    // Heightfield streaming parameters and mutable state.
    // terrainHeight() must match terrain_clipmap.hlsl terrainHeight() exactly.
    auto terrainHeight = [](float wx, float wz) -> float {
        return std::sin(wx * 0.05f) * std::cos(wz * 0.05f) * 2.0f
             + std::sin(wx * 0.13f + 1.1f) * std::sin(wz * 0.09f) * 0.8f;
    };
    constexpr u32 kHFN       = 512;
    constexpr f32 kHFSize    = 512.0f;
    constexpr f32 kHFRebuild = 150.0f;
    constexpr f32 kHFSnap    = 64.0f;
    u32  hfBodyId = ~0u;
    vec2 hfCenter = {0.0f, 0.0f};

    // Build the initial heightfield centred at the origin.
    {
        const f32 spacing = kHFSize / static_cast<f32>(kHFN - 1);
        const f32 oriX    = -kHFSize * 0.5f;
        const f32 oriZ    = -kHFSize * 0.5f;
        std::vector<f32> heights(static_cast<size_t>(kHFN) * kHFN);
        for (u32 row = 0; row < kHFN; ++row)
            for (u32 col = 0; col < kHFN; ++col)
                heights[row * kHFN + col] =
                    terrainHeight(oriX + col * spacing, oriZ + row * spacing);
        hfBodyId = engine.physics().addHeightField(
            vec3(oriX, 0.0f, oriZ), kHFSize, kHFN, heights);
        ENIGMA_LOG_INFO("[app] initial heightfield built at origin");
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
    world.add_system(ecs::SystemSchedule::PostPhysics,
        ecs::systems::makeTerrainSystem(terrain, camera,
                                         engine.physics(), engine.interpolation(),
                                         *engine.vehicle(),
                                         terrainHeight,
                                         hfBodyId, hfCenter,
                                         kHFSize, kHFN, kHFRebuild, kHFSnap));

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

    // Clean up scene and terrain before renderer teardown.
    renderer.setTerrain(nullptr);
    if (scene.has_value()) {
        renderer.setScene(nullptr);
        vkDeviceWaitIdle(renderer.device().logical());
        scene->destroy(renderer.device(), renderer.allocator());
    }

    ENIGMA_LOG_INFO("[app] shutdown");
    return 0;
}

} // namespace enigma
