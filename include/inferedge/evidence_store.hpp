#pragma once

#include "inferedge/admission_record.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace inferedge {

// Phase 9-C6 evidence-store foundation.
//
// Every operation receives its evidence root, and where a journal or authorization file is
// required, its admission root, from the caller. This module does not search a working
// directory, a home directory, a repository root, or a parent directory, and it does not
// choose a file by directory order or modification time.
//
// Parsing a canonical record never produces authority. This header exposes no verifier
// base class, no friend factory, no registration hook, and no compile-time switch that
// a consumer can define. AuthenticatedCapability::authenticate mints a capability only
// inside src/evidence_store.cpp, after that file accepts a decision from the one strong
// Ed25519 verifier and assigns the internal stamp. Supplying a header, object file, or
// build flag cannot do that. Destructive recovery, reservation cleanup, and quarantine
// deletion require a capability whose bindings match the request and the device, inode,
// and directory kind captured from opened descriptors. A mismatch returns before
// mutation. There is no process-wide grant.
//
// Section 22 quarantine inventory, and nothing else, may appear in quarantine/<run-id>/:
// quarantine.record for every reason code; recipe-echo.bin only when the reason is
// recipe_echo_rejected; deletion.record only while an authorized deletion is in progress.
// The other seven reason codes have no contract-defined payload. Metadata-only content is
// their complete inventory. No other filename is created or deleted.

inline constexpr std::uint64_t kEvidenceCapBytes = 68719476736ull;
inline constexpr std::uint64_t kQuarantineDirectoryCapBytes = 1073741824ull;
inline constexpr std::uint64_t kEvidenceLockWaitSeconds = 600;

enum class EvidenceError {
    kOk = 0,
    kLexical,
    kPath,
    kSymlink,
    kType,
    kLink,
    kSpecial,
    kCap,
    kOverflow,
    kLock,
    kOwner,
    kSize,
    kDigest,
    kChecksum,
    kSchema,
    kDuplicate,
    kAuthorization,
    kUnavailable,
    kDate,
    kRace,
    kIo,
    kPresent,
    kAbsent,
    kState,
    kForbidden,
    kInternal,
};

const char* evidence_error_name(EvidenceError error) noexcept;

enum class EvidenceObjectClass {
    kPublishedBlob,
    kRetainedMetadata,
    kTemporaryMetadata,
};

enum class CleanupObjectClass {
    kEvidenceReservation,
    kPartialStaging,
    kTemporaryMetadata,
};

struct ReservationEntry {
    std::string staging_relative_path;
    std::string final_relative_path;
    std::uint64_t reserved_object_size{0};
    std::string reserved_object_sha256;
    EvidenceObjectClass object_class{EvidenceObjectClass::kPublishedBlob};
};

struct EvidenceLockRecord {
    std::string owner_run_id;
    std::string owner_host_id;
    std::string acquired_date;
    std::string bytes;
};

struct RecoveryLockRecord {
    std::string owner_run_id;
    std::string acquired_date;
    std::string bytes;
};

struct EvidenceReservation {
    std::string owner_run_id;
    std::uint64_t reserved_bytes{0};
    std::uint64_t path_count{0};
    std::vector<ReservationEntry> entries;
    std::string bytes;
};

struct StaleLockAuthorization {
    std::string authorization_id;
    std::string lock_path;
    std::string lock_owner_run_id;
    std::uint64_t lock_file_size{0};
    std::string lock_file_sha256;
    std::string acquired_date;
    std::string authorization_date;
    std::string bytes;
};

struct ReservationDeletionAuthorization {
    std::string authorization_id;
    std::string target_id;
    std::string reservation_path;
    std::uint64_t target_file_size{0};
    std::string target_file_sha256;
    std::string owner_run_id;
    std::string deletion_list_id;
    std::string deletion_list_path;
    std::uint64_t deletion_list_file_size{0};
    std::string deletion_list_file_sha256;
    std::string authorization_date;
    std::string bytes;
};

struct QuarantineDeletionAuthorization {
    std::string authorization_id;
    std::string target_id;
    std::uint64_t target_file_size{0};
    std::string target_file_sha256;
    std::string authorization_date;
    std::string bytes;
};

struct DeletionListEntry {
    std::string relative_path;
    std::uint64_t deletion_entry_size{0};
    std::string deletion_entry_sha256;
    CleanupObjectClass object_class{CleanupObjectClass::kPartialStaging};
};

struct ReservationDeletionList {
    std::string list_id;
    std::string reservation_id;
    std::string owner_run_id;
    std::uint64_t entry_count{0};
    std::vector<DeletionListEntry> entries;
    std::string bytes;
};

struct CleanupPendingRecord {
    std::string authorization_id;
    std::uint64_t authorization_file_size{0};
    std::string authorization_file_sha256;
    std::string deletion_list_id;
    std::uint64_t deletion_list_file_size{0};
    std::string deletion_list_file_sha256;
    std::string owner_run_id;
    std::string creation_date;
    std::string bytes;
};

struct CleanupCompletionRecord {
    std::string authorization_id;
    std::uint64_t pending_file_size{0};
    std::string pending_file_sha256;
    std::uint64_t authorization_file_size{0};
    std::string authorization_file_sha256;
    std::string deletion_list_id;
    std::uint64_t deletion_list_file_size{0};
    std::string deletion_list_file_sha256;
    std::string owner_run_id;
    std::string completion_date;
    std::string bytes;
};

struct QuarantineMetadata {
    std::string quarantine_id;
    std::string associated_phase_id;
    std::string associated_run_id;
    std::string directory_id;
    std::string creation_date;
    std::string reason_code;
    std::string accepted_run_manifest_path;
    std::uint64_t accepted_run_manifest_file_size{0};
    std::string accepted_run_manifest_file_sha256;
    std::string retention_start;
    std::string retention_deadline;
    std::string bytes;
};

struct DeletionJournal {
    std::string quarantine_id;
    std::string authorization_id;
    std::uint64_t authorization_file_size{0};
    std::string authorization_file_sha256;
    std::string bytes;
};

struct QuarantinePayload {
    std::string filename;
    std::string bytes;
};

class CapabilityResult;

// Move-only proof object. A default, moved-from, or failed object holds no authority.
// Copying is rejected. There is no setter and no function an ordinary translation unit
// can define in order to write these fields. Role, revocation, and signature result are
// recorded by this translation unit after a verifier decision, never from caller strings.
class AuthenticatedCapability {
public:
    AuthenticatedCapability() noexcept;
    AuthenticatedCapability(AuthenticatedCapability&&) noexcept;
    AuthenticatedCapability& operator=(AuthenticatedCapability&&) noexcept;
    ~AuthenticatedCapability();
    AuthenticatedCapability(const AuthenticatedCapability&) = delete;
    AuthenticatedCapability& operator=(const AuthenticatedCapability&) = delete;

    bool holds_authority() const noexcept;
    std::string_view operation() const noexcept;
    std::string_view authorization_bytes() const noexcept;
    std::string_view canonical_authorization_path() const noexcept;
    std::string_view target_id() const noexcept;
    std::string_view evidence_root() const noexcept;
    std::string_view admission_root() const noexcept;
    std::uint64_t evidence_device() const noexcept;
    std::uint64_t evidence_inode() const noexcept;
    std::uint32_t evidence_kind() const noexcept;
    std::uint64_t admission_device() const noexcept;
    std::uint64_t admission_inode() const noexcept;
    std::uint32_t admission_kind() const noexcept;
    std::string_view terminal_trust_state_id() const noexcept;
    std::string_view terminal_effective_date() const noexcept;
    std::string_view signer_key_id() const noexcept;
    std::string_view required_role() const noexcept;
    std::string_view key_valid_from() const noexcept;
    std::string_view key_valid_until() const noexcept;
    std::string_view revocation_decision() const noexcept;
    std::string_view signature_verification_result() const noexcept;

    // Caller text is compared with authenticated records. It is not verification.
    static CapabilityResult
    authenticate(std::string_view operation, std::string_view authorization_bytes,
                 std::string_view canonical_authorization_path, std::string_view target_id,
                 std::string_view trust_root, std::string_view evidence_root,
                 std::string_view admission_root, std::string_view terminal_trust_state_id,
                 std::string_view terminal_effective_date, std::string_view signer_key_id,
                 std::string_view key_valid_from, std::string_view key_valid_until);

private:
    struct State;
    std::unique_ptr<State> state_;
};

// Failure and success are separate states. status() != kOk implies has_capability() == false.
// release_capability() moves the owned capability out. The result retains no pointer to it.
class CapabilityResult {
public:
    CapabilityResult() noexcept;
    CapabilityResult(CapabilityResult&&) noexcept;
    CapabilityResult& operator=(CapabilityResult&&) noexcept;
    ~CapabilityResult();
    CapabilityResult(const CapabilityResult&) = delete;
    CapabilityResult& operator=(const CapabilityResult&) = delete;

    EvidenceError status() const noexcept;
    bool has_capability() const noexcept;
    AuthenticatedCapability release_capability();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class AuthenticatedCapability;
};

class EvidenceTime {
public:
    virtual ~EvidenceTime() = default;
    virtual EvidenceError civil_date(std::string& out) const = 0;
    virtual std::chrono::steady_clock::time_point now() const = 0;
    virtual void sleep_for(std::chrono::milliseconds duration) = 0;
};

class BatchProbe {
public:
    virtual ~BatchProbe() = default;
    virtual EvidenceError after_staging_written(std::string_view staging_relative_path);
};

class EvidenceLock {
public:
    EvidenceLock() = default;
    EvidenceLock(EvidenceLock&& other) noexcept;
    EvidenceLock& operator=(EvidenceLock&& other) noexcept;
    EvidenceLock(const EvidenceLock&) = delete;
    EvidenceLock& operator=(const EvidenceLock&) = delete;
    ~EvidenceLock();

    bool held() const { return held_; }
    const std::string& owner_run_id() const { return owner_run_id_; }
    const std::string& evidence_root() const { return evidence_root_; }
    EvidenceError release();

private:
    friend EvidenceError acquire_evidence_lock(std::string_view evidence_root,
                                               std::string_view owner_run_id,
                                               std::string_view owner_host_id,
                                               std::string_view acquired_date, EvidenceTime* clock,
                                               EvidenceLock& out);
    friend EvidenceError publish_batch(EvidenceLock& lock, std::string_view reservation_id,
                                       const std::vector<ReservationEntry>& entries,
                                       const std::vector<std::string>& payloads, BatchProbe* probe);
    friend EvidenceError
    recover_stale_lock(std::string_view evidence_root, std::string_view admission_root,
                       std::string_view terminal_effective_date,
                       std::string_view terminal_trust_state_id, std::string_view signer_key_id,
                       std::string_view authorization_bytes,
                       const AuthenticatedCapability* capability, std::string_view new_owner_run_id,
                       std::string_view new_owner_host_id, EvidenceTime* clock, EvidenceLock& out,
                       bool& evidence_lock_acquired);
    friend EvidenceError cleanup_reservation(EvidenceLock& lock, std::string_view admission_root,
                                             std::string_view terminal_effective_date,
                                             std::string_view terminal_trust_state_id,
                                             std::string_view signer_key_id,
                                             std::string_view authorization_bytes,
                                             const AuthenticatedCapability* capability,
                                             std::string_view deletion_list_bytes,
                                             EvidenceTime* clock);
    friend EvidenceError create_quarantine(EvidenceLock& lock, std::string_view run_id,
                                           std::string_view phase_id, std::string_view reason_code,
                                           std::string_view manifest_path,
                                           const std::vector<QuarantinePayload>& payloads,
                                           EvidenceTime* clock);
    friend EvidenceError delete_quarantine(EvidenceLock& lock, std::string_view admission_root,
                                           std::string_view terminal_effective_date,
                                           std::string_view terminal_trust_state_id,
                                           std::string_view signer_key_id,
                                           std::string_view authorization_bytes,
                                           const AuthenticatedCapability* capability,
                                           EvidenceTime* clock);

    void reset() noexcept;
    void adopt(int root_fd, int lock_dir_fd, int lock_fd, std::uint64_t device, std::uint64_t inode,
               std::string owner_run_id, std::string owner_host_id, std::string evidence_root);
    int root_fd_{-1};
    int lock_fd_{-1};
    int lock_dir_fd_{-1};
    std::uint64_t device_{0};
    std::uint64_t inode_{0};
    std::string owner_run_id_;
    std::string owner_host_id_;
    std::string evidence_root_;
    bool held_{false};
};

EvidenceError validate_evidence_relative_path(std::string_view path);
EvidenceError reservation_fits_cap(std::uint64_t sum_before, std::uint64_t reservation_record_size,
                                   std::uint64_t planned_batch_bytes);

EvidenceError sum_evidence_root(std::string_view evidence_root, std::uint64_t& total);

EvidenceError parse_evidence_lock(std::string_view bytes, EvidenceLockRecord& out);
EvidenceError parse_recovery_lock(std::string_view bytes, RecoveryLockRecord& out);
EvidenceError parse_evidence_reservation(std::string_view bytes, EvidenceReservation& out);
EvidenceError parse_stale_lock_authorization(std::string_view bytes, StaleLockAuthorization& out);
EvidenceError parse_reservation_deletion(std::string_view bytes,
                                         ReservationDeletionAuthorization& out);
EvidenceError parse_quarantine_deletion(std::string_view bytes,
                                        QuarantineDeletionAuthorization& out);
EvidenceError parse_reservation_deletion_list(std::string_view bytes, ReservationDeletionList& out);
EvidenceError parse_cleanup_pending(std::string_view bytes, CleanupPendingRecord& out);
EvidenceError parse_cleanup_completion(std::string_view bytes, CleanupCompletionRecord& out);
EvidenceError parse_quarantine_metadata(std::string_view bytes, QuarantineMetadata& out);
EvidenceError parse_deletion_journal(std::string_view bytes, DeletionJournal& out);

EvidenceError serialize_evidence_lock(const EvidenceLockRecord& fields, std::string& out);
EvidenceError serialize_evidence_reservation(std::string_view owner_run_id,
                                             const std::vector<ReservationEntry>& entries,
                                             EvidenceReservation& out);
EvidenceError serialize_stale_lock_authorization(const StaleLockAuthorization& fields,
                                                 std::string& out);
EvidenceError serialize_reservation_deletion(const ReservationDeletionAuthorization& fields,
                                             std::string& out);
EvidenceError serialize_quarantine_deletion(const QuarantineDeletionAuthorization& fields,
                                            std::string& out);
EvidenceError serialize_reservation_deletion_list(const ReservationDeletionList& fields,
                                                  std::string& out);
EvidenceError serialize_cleanup_pending(const CleanupPendingRecord& fields, std::string& out);
EvidenceError serialize_cleanup_completion(const CleanupCompletionRecord& fields, std::string& out);
EvidenceError serialize_quarantine_metadata(const QuarantineMetadata& fields, std::string& out);
EvidenceError serialize_deletion_journal(const DeletionJournal& fields, std::string& out);

EvidenceError acquire_evidence_lock(std::string_view evidence_root, std::string_view owner_run_id,
                                    std::string_view owner_host_id, std::string_view acquired_date,
                                    EvidenceTime* clock, EvidenceLock& out);

EvidenceError publish_batch(EvidenceLock& lock, std::string_view reservation_id,
                            const std::vector<ReservationEntry>& entries,
                            const std::vector<std::string>& payloads, BatchProbe* probe);

EvidenceError
recover_stale_lock(std::string_view evidence_root, std::string_view admission_root,
                   std::string_view terminal_effective_date,
                   std::string_view terminal_trust_state_id, std::string_view signer_key_id,
                   std::string_view authorization_bytes, const AuthenticatedCapability* capability,
                   std::string_view new_owner_run_id, std::string_view new_owner_host_id,
                   EvidenceTime* clock, EvidenceLock& out, bool& evidence_lock_acquired);

EvidenceError cleanup_reservation(EvidenceLock& lock, std::string_view admission_root,
                                  std::string_view terminal_effective_date,
                                  std::string_view terminal_trust_state_id,
                                  std::string_view signer_key_id,
                                  std::string_view authorization_bytes,
                                  const AuthenticatedCapability* capability,
                                  std::string_view deletion_list_bytes, EvidenceTime* clock);

EvidenceError create_quarantine(EvidenceLock& lock, std::string_view run_id,
                                std::string_view phase_id, std::string_view reason_code,
                                std::string_view manifest_path,
                                const std::vector<QuarantinePayload>& payloads,
                                EvidenceTime* clock);

EvidenceError delete_quarantine(EvidenceLock& lock, std::string_view admission_root,
                                std::string_view terminal_effective_date,
                                std::string_view terminal_trust_state_id,
                                std::string_view signer_key_id,
                                std::string_view authorization_bytes,
                                const AuthenticatedCapability* capability, EvidenceTime* clock);

} // namespace inferedge
