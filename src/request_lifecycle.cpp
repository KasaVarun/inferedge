#include "inferedge/request_lifecycle.hpp"

#include <string>
#include <string_view>
#include <utility>

namespace inferedge {
namespace {

std::string_view state_name(RequestState state) {
    switch (state) {
    case RequestState::kCreated:
        return "created";
    case RequestState::kQueued:
        return "queued";
    case RequestState::kRunning:
        return "running";
    case RequestState::kCompleted:
        return "completed";
    case RequestState::kFailed:
        return "failed";
    }
    return "unknown";
}

bool is_allowed(RequestState from, RequestState to) {
    switch (from) {
    case RequestState::kCreated:
        return to == RequestState::kQueued;
    case RequestState::kQueued:
        return to == RequestState::kRunning;
    case RequestState::kRunning:
        return to == RequestState::kCompleted || to == RequestState::kFailed;
    case RequestState::kCompleted:
    case RequestState::kFailed:
        return false;
    }
    return false;
}

} // namespace

RequestState RequestLifecycle::state() const {
    return state_;
}

LifecycleResult RequestLifecycle::transition_to(RequestState next) {
    if (!is_allowed(state_, next)) {
        std::string message = "invalid transition from ";
        message += state_name(state_);
        message += " to ";
        message += state_name(next);
        return LifecycleResult{LifecycleError::kInvalidTransition, std::move(message)};
    }
    state_ = next;
    return LifecycleResult{LifecycleError::kNone, {}};
}

} // namespace inferedge
