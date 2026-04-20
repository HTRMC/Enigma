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
#include "renderer/micropoly/MicropolyConfig.h"
#include "scene/Camera.h"
#include "scene/FollowCamera.h"
#include "scene/Scene.h"
#include "world/CdlodTerrain.h"
#include "world/HeightmapLoader.h"

#include <volk.h>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace enigma {

Application::Application()  = default;
Application::~Application() = default;

int Application::run(int argc, char** argv) {
    const char* argv0 = (argc > 0 && argv != nullptr) ? argv[0] : nullptr;
    Paths::init(argv0);
    ENIGMA_LOG_INFO("[app] starting, exe = {}", Paths::executablePath().string());

    // Parse --micropoly <path> early so the config can be threaded into the
    // Renderer via Engine's ctor. MicropolyStreaming::create handles path
    // validation / non-existent files via std::expected; no pre-check here.
    MicropolyConfig mpCfg{};
    for (int i = 1; i < argc - 1; ++i) {
        if (argv[i] && std::string_view(argv[i]) == "--micropoly" && argv[i + 1]) {
            // asset::isSafeMpaPath rejects non-absolute paths as defense-in-depth
            // (see src/asset/MpPathUtils.h:40-46). The CLI typically passes a
            // relative path like "assets/bmw.mpa" — canonicalise here so the
            // streaming init path sees a clean absolute path. std::filesystem::
            // absolute does NOT require the file to exist; a nonexistent path
            // still canonicalises against the CWD, and open() failure surfaces
            // cleanly in the streaming init log.
            mpCfg.mpaFilePath = std::filesystem::absolute(argv[i + 1]);
            mpCfg.enabled     = true;
            ENIGMA_LOG_INFO("[app] micropoly enabled: {}", mpCfg.mpaFilePath.string());
        }
    }

    Engine engine(std::move(mpCfg));
    auto& window   = engine.window();
    auto& renderer = engine.renderer();
    auto& clock    = engine.clock();
    auto& input    = engine.input();
    auto& world    = engine.world();

    // Camera — overwritten every PreRender tick by CameraSystem.
    Camera camera({0.0f, 3.0f, 8.0f}, 60.0f, 0.1f);
    renderer.setCamera(&camera);
    FollowCamera followCam(camera, /*armLength=*/8.0f, /*heightOffset=*/2.5f);

    // Parse optional flags. --profile N: run N frames, dump GPU timing CSV, exit.
    u32 profileFrames = 0;
    for (int i = 1; i < argc - 1; ++i) {
        if (argv[i] && std::string_view(argv[i]) == "--profile" && argv[i + 1]) {
            profileFrames = static_cast<u32>(std::stoul(argv[i + 1]));
            ENIGMA_LOG_INFO("[app] profile mode: {} capture frames (30 warmup)", profileFrames);
        }
    }

    // glTF scene loading: command-line path, BMW M4 GT3, or bundled default.
    std::optional<Scene> scene;
    std::filesystem::path modelPath;
    if (argc > 1 && argv[1] != nullptr && std::string_view(argv[1]) != "--profile") {
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
    // Frame loop.
    // -------------------------------------------------------------------------
    constexpr u32 kProfileWarmup = 30; // frames to skip before collecting
    u32 totalFrames = 0;
    std::map<std::string, std::vector<f32>> profileData;
    std::vector<f32> cpuFrameTimes;

    while (!window.shouldClose()) {
        window.pollEvents();
        const f32 dt = static_cast<f32>(clock.tick());
        input.update();
        world.run_systems(dt);
        renderer.drawFrame();
        ++totalFrames;

        // GPU + CPU timing collection (profile mode only).
        if (profileFrames > 0 && totalFrames > kProfileWarmup) {
            cpuFrameTimes.push_back(dt * 1000.f); // dt is seconds → ms
            for (const auto& r : renderer.gpuTimings())
                profileData[r.name].push_back(r.durationMs);

            const u32 collected = totalFrames - kProfileWarmup;
            if (collected >= profileFrames) {
                // Print table to stdout.
                const std::string sep(60, '-');
                printf("\n%s\n", sep.c_str());
                printf("  GPU Profile  |  %u frames captured  |  30 warmup\n", profileFrames);
                printf("%s\n", sep.c_str());
                printf("  %-28s  %7s  %7s  %7s\n", "Pass", "Min ms", "Max ms", "Avg ms");
                printf("  %-28s  %7s  %7s  %7s\n",
                       std::string(28,'-').c_str(), "-------", "-------", "-------");

                const auto csvPath =
                    Paths::executablePath().parent_path() / "enigma_profile.csv";
                std::ofstream csv(csvPath);
                csv << "Pass,MinMs,MaxMs,AvgMs,P95Ms\n";

                // CPU frame time summary.
                if (!cpuFrameTimes.empty()) {
                    std::sort(cpuFrameTimes.begin(), cpuFrameTimes.end());
                    const f32 cpuMin = cpuFrameTimes.front();
                    const f32 cpuMax = cpuFrameTimes.back();
                    const f32 cpuAvg = std::accumulate(cpuFrameTimes.begin(), cpuFrameTimes.end(), 0.f)
                                     / static_cast<f32>(cpuFrameTimes.size());
                    const f32 cpuP95 = cpuFrameTimes[static_cast<size_t>(cpuFrameTimes.size() * 0.95f)];
                    printf("  %-28s  %7.3f  %7.3f  %7.3f   (p95=%.3fms, ~%.0f fps)\n",
                           "CPU FrameTime", cpuMin, cpuMax, cpuAvg, cpuP95, 1000.f / cpuAvg);
                    csv << "CPU FrameTime," << cpuMin << "," << cpuMax << "," << cpuAvg << "," << cpuP95 << "\n";
                }

                f32 gpuTotal = 0.f;
                for (auto& [name, samples] : profileData) {
                    std::sort(samples.begin(), samples.end());
                    const f32 minMs  = samples.front();
                    const f32 maxMs  = samples.back();
                    const f32 avgMs  = std::accumulate(samples.begin(), samples.end(), 0.f)
                                     / static_cast<f32>(samples.size());
                    const f32 p95Ms  = samples[static_cast<size_t>(samples.size() * 0.95f)];
                    printf("  %-28s  %7.3f  %7.3f  %7.3f\n",
                           name.c_str(), minMs, maxMs, avgMs);
                    csv << name << "," << minMs << "," << maxMs << ","
                        << avgMs << "," << p95Ms << "\n";
                    gpuTotal += avgMs;
                }

                printf("%s\n", sep.c_str());
                printf("  %-28s  %7s  %7s  %7.3f\n", "GPU TOTAL (avg)", "", "", gpuTotal);
                printf("%s\n\n", sep.c_str());
                printf("  CSV written to: %s\n\n", csvPath.string().c_str());
                break;
            }
        }
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
