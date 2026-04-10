#include "physics/VehicleConfig.h"

namespace enigma {

VehicleConfig VehicleConfig::makeDefault() {
    VehicleConfig cfg;
    cfg.name = "sedan";
    cfg.mass = 1500.0f;
    cfg.centerOfMassOffset = {0.0f, -0.2f, 0.0f};

    // Front-wheel drive: front 2 wheels are driven + steered.
    const f32 halfTrack  = 0.75f; // half the distance between left/right wheels
    const f32 frontAxleZ = 1.3f;
    const f32 rearAxleZ  = -1.2f;
    const f32 wheelY     = -0.3f; // below body center

    WheelConfig frontLeft;
    frontLeft.offset  = {-halfTrack, wheelY, frontAxleZ};
    frontLeft.driven  = true;
    frontLeft.steered = true;

    WheelConfig frontRight;
    frontRight.offset  = {halfTrack, wheelY, frontAxleZ};
    frontRight.driven  = true;
    frontRight.steered = true;

    WheelConfig rearLeft;
    rearLeft.offset  = {-halfTrack, wheelY, rearAxleZ};
    rearLeft.driven  = false;
    rearLeft.steered = false;

    WheelConfig rearRight;
    rearRight.offset  = {halfTrack, wheelY, rearAxleZ};
    rearRight.driven  = false;
    rearRight.steered = false;

    cfg.wheels = {frontLeft, frontRight, rearLeft, rearRight};

    // 6-speed automatic.
    cfg.gears = {{3.82f}, {2.20f}, {1.40f}, {1.00f}, {0.80f}, {0.62f}};
    cfg.finalDriveRatio = 3.73f;
    cfg.shiftUpRPM   = 5500.0f;
    cfg.shiftDownRPM = 2000.0f;

    return cfg;
}

} // namespace enigma
