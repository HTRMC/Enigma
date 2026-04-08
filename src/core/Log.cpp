#include "core/Log.h"

#include <iostream>

namespace enigma::log {

static const char* levelTag(Level level) {
    switch (level) {
        case Level::Trace: return "[trace]";
        case Level::Info:  return "[info ]";
        case Level::Warn:  return "[warn ]";
        case Level::Error: return "[error]";
    }
    return "[?????]";
}

void emit(Level level, std::string_view message) {
    auto& sink = (level == Level::Warn || level == Level::Error)
                     ? std::cerr
                     : std::cout;
    sink << levelTag(level) << ' ' << message << '\n';
}

} // namespace enigma::log
