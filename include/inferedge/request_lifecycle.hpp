#pragma once

#include <string>

namespace inferedge {

enum class RequestState {
    kCreated,
    kQueued,
    kRunning,
    kCompleted,
    kFailed,
};

enum class LifecycleError {
    kNone,
    kInvalidTransition,
};

struct LifecycleResult {
    LifecycleError error{LifecycleError::kNone};
    std::string message;
};

// Owns the progress of one request. Callers can read the state and request a
// transition. They cannot assign the state directly.
// Legal edges are Created->Queued, Queued->Running, Running->Completed, and
// Running->Failed. Any other request, including an enumerator outside this
// list, returns kInvalidTransition and leaves the state unchanged. The failure
// message names unknown enumerator values as "unknown".
class RequestLifecycle {
public:
    RequestLifecycle() = default;

    RequestState state() const;

    LifecycleResult transition_to(RequestState next);

private:
    RequestState state_{RequestState::kCreated};
};

} // namespace inferedge
