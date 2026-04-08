#pragma once

#include <format>
#include <string_view>
#include <utility>

namespace enigma::log {

enum class Level {
    Trace,
    Info,
    Warn,
    Error,
};

// Emit a pre-formatted line to stdout/stderr depending on severity.
void emit(Level level, std::string_view message);

template <class... Args>
void write(Level level, std::format_string<Args...> fmt, Args&&... args) {
    emit(level, std::format(fmt, std::forward<Args>(args)...));
}

} // namespace enigma::log

#define ENIGMA_LOG_TRACE(...) ::enigma::log::write(::enigma::log::Level::Trace, __VA_ARGS__)
#define ENIGMA_LOG_INFO(...)  ::enigma::log::write(::enigma::log::Level::Info,  __VA_ARGS__)
#define ENIGMA_LOG_WARN(...)  ::enigma::log::write(::enigma::log::Level::Warn,  __VA_ARGS__)
#define ENIGMA_LOG_ERROR(...) ::enigma::log::write(::enigma::log::Level::Error, __VA_ARGS__)
