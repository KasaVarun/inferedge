#pragma once

#include <cstdint>
#include <limits>
#include <vector>

namespace inferedge {

// Maximum per-worker sealed-batch event capacity when batch-decision timing
// is enabled. Valid capacities are 1 through this bound inclusive.
inline constexpr std::uint32_t kMaxBatchDecisionEventCapacity = 65536;

// Why a non-empty worker-local batch stopped accepting additional requests.
// Reasons are mutually exclusive and recorded at the winning RunBatch site.
enum class BatchSealReason : std::uint8_t {
    kCapacityReached = 0,
    kDeadlineExpired = 1,
    kG3MinimumTarget = 2,
    kG4ArrivalAware = 3,
    kQueueClosed = 4,
    kShutdownOrOther = 5,
};

// One sealed-batch decision. POD-like; no payloads, request IDs, or heap.
// Timestamps are steady_clock time_since_epoch nanoseconds.
struct BatchDecisionEvent {
    std::uint64_t batch_id{0};
    std::uint32_t worker_index{0};
    std::int64_t batch_open_ns{0};
    std::int64_t batch_seal_ns{0};
    std::uint32_t batch_size{0};
    BatchSealReason seal_reason{BatchSealReason::kShutdownOrOther};
};

// Post-join read-only snapshot of batch-decision timing. Independent of
// SchedulerMetrics and completion timing. timing_data_valid is true only when
// the feature was enabled, every worker joined successfully, no overflow or
// drop occurred, timestamps are ordered, counts reconcile, and batch IDs are
// unique. Callers must not destroy a scheduler concurrently with
// batch_decision_timing().
struct BatchDecisionTimingSnapshot {
    bool enabled{false};
    bool timing_data_valid{false};
    bool trace_overflow{false};
    std::uint32_t buffer_capacity_per_worker{0};
    std::uint64_t sealed_batches_total{0};
    std::uint64_t events_recorded{0};
    std::uint64_t events_dropped{0};
    std::vector<BatchDecisionEvent> events;
};

// Packs a worker-local sequence into a process-wide batch_id.
// local_sequence must be non-zero and at most UINT32_MAX.
[[nodiscard]] inline std::uint64_t PackBatchDecisionId(std::uint32_t worker_index,
                                                       std::uint32_t local_sequence) noexcept {
    return (static_cast<std::uint64_t>(worker_index) << 32) |
           static_cast<std::uint64_t>(local_sequence);
}

// True when worker_count * capacity * sizeof(BatchDecisionEvent) fits in size_t.
[[nodiscard]] inline bool BatchDecisionAllocationFits(std::uint32_t worker_count,
                                                      std::uint32_t capacity) noexcept {
    if (worker_count == 0 || capacity == 0) {
        return false;
    }
    const std::size_t per_event = sizeof(BatchDecisionEvent);
    const std::size_t max_size = std::numeric_limits<std::size_t>::max();
    if (per_event == 0 || static_cast<std::size_t>(capacity) > max_size / per_event) {
        return false;
    }
    const std::size_t per_worker = static_cast<std::size_t>(capacity) * per_event;
    if (per_worker == 0 || static_cast<std::size_t>(worker_count) > max_size / per_worker) {
        return false;
    }
    return true;
}

} // namespace inferedge
