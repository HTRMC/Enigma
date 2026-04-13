// P2996 Feasibility Gate — Milestone 0, Day 1-2
//
// This file probes whether MSVC's /std:c++latest implements enough of P2996
// (static reflection, std::meta) to support the Enigma comptime ECS design.
//
// PASS: compiles clean, all static_asserts hold
//   → proceed with P2996 path for Phase 1 ECS Core
// FAIL: any compile error
//   → commit to ENIGMA_COMPONENT macro path; do not revisit P2996 this phase
//
// Required: MSVC /std:c++latest /W4 /WX  (or -std=c++2c on Clang/GCC)

#if defined(__cpp_reflection) || defined(__cpp_static_reflection)
// P2996 experimental guard — some compilers define these feature-test macros

#include <meta>
#include <string_view>
#include <array>

// --- Test struct: 5 fields, trivially copyable ---
struct TestComponent {
    float x;
    float y;
    float z;
    float w;
    int   id;
};

// --- Probe 1: enumerate members via std::meta::members_of ---
consteval auto get_member_count() -> std::size_t {
    return std::meta::members_of(^TestComponent,
        std::meta::is_nonstatic_data_member).size();
}

static_assert(get_member_count() == 5,
    "P2996 probe failed: expected 5 data members in TestComponent");

// --- Probe 2: generate compile-time field names ---
consteval auto get_first_field_name() -> std::string_view {
    auto members = std::meta::members_of(^TestComponent,
        std::meta::is_nonstatic_data_member);
    return std::meta::name_of(members[0]);
}

static_assert(get_first_field_name() == "x",
    "P2996 probe failed: first field name should be 'x'");

// --- Probe 3: compile-time SOA metadata (the key ECS requirement) ---
// Simulate what Archetype<Cs...> needs: a compile-time array of field offsets
consteval auto get_field_offsets() {
    auto members = std::meta::members_of(^TestComponent,
        std::meta::is_nonstatic_data_member);
    std::array<std::size_t, 5> offsets{};
    for (std::size_t i = 0; i < members.size(); ++i) {
        offsets[i] = std::meta::offset_of(members[i]);
    }
    return offsets;
}

constexpr auto kOffsets = get_field_offsets();
static_assert(kOffsets[0] == offsetof(TestComponent, x), "P2996 probe: x offset mismatch");
static_assert(kOffsets[1] == offsetof(TestComponent, y), "P2996 probe: y offset mismatch");
static_assert(kOffsets[4] == offsetof(TestComponent, id), "P2996 probe: id offset mismatch");

// --- Probe 4: type-based compile-time ID (needed for Query<Cs...> matching) ---
consteval auto type_id_of(std::meta::info r) -> std::size_t {
    // Use the reflection token as a stable compile-time hash seed
    return std::meta::identifier_of(r).size(); // placeholder — real impl uses FNV hash
}

constexpr std::size_t kTestComponentId = type_id_of(^TestComponent);
static_assert(kTestComponentId > 0, "P2996 probe: type ID generation failed");

int main() {
    // If we got here, P2996 compiled successfully.
    // All static_asserts above validated at compile time.
    return 0;
}

#else
// P2996 / std::meta not available on this compiler/flags combination.
// This is the expected result on MSVC /std:c++latest as of April 2026.
// → FAIL path: commit to ENIGMA_COMPONENT macro ECS.
#error "P2996_PROBE_FAIL: std::meta not available. Commit to ENIGMA_COMPONENT macro path."
#endif
