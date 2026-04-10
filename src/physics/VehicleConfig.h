#pragma once

#include "core/Types.h"
#include "core/Math.h"

#include <string>
#include <vector>

namespace enigma {

struct TireConfig {
    f32 maxLateralFriction      = 1.5f;
    f32 maxLongitudinalFriction = 1.0f;
    f32 radius                  = 0.33f; // meters
    f32 width                   = 0.22f; // meters
};

struct SuspensionConfig {
    f32 restLength = 0.30f;    // meters
    f32 minLength  = 0.10f;
    f32 maxLength  = 0.50f;
    f32 stiffness  = 30000.0f; // N/m
    f32 damping    = 3000.0f;  // N*s/m
};

struct WheelConfig {
    vec3             offset{};      // position relative to body center
    TireConfig       tire{};
    SuspensionConfig suspension{};
    bool             driven  = false; // receives engine torque
    bool             steered = false; // responds to steering input
};

struct GearRatio {
    f32 ratio = 1.0f;
};

struct VehicleConfig {
    std::string name = "default_car";
    f32  mass        = 1500.0f; // kg
    vec3 centerOfMassOffset{0.0f, -0.2f, 0.0f}; // lower CoM for stability

    std::vector<WheelConfig> wheels;

    // Engine.
    f32 maxEngineTorque = 500.0f; // N*m at peak
    f32 maxRPM          = 7000.0f;
    f32 idleRPM         = 800.0f;

    // Transmission (automatic).
    std::vector<GearRatio> gears;
    f32 shiftUpRPM      = 5500.0f;
    f32 shiftDownRPM    = 2000.0f;
    f32 finalDriveRatio = 3.73f;

    // Default 4-wheel configuration.
    static VehicleConfig makeDefault();
};

} // namespace enigma
