#pragma once

#include "inferedge/cli.hpp"

namespace inferedge {

// Loads one model, runs it, and returns stdout or stderr text.
// parsed.inputs owns the floats. Their spans stay valid for this call.
// Inference itself is InferenceEngine::run.
CliResult execute_inference(const CliResult& parsed);

} // namespace inferedge
