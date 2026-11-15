#ifndef INFEREDGE_TRUST_TRANSITION_HPP
#define INFEREDGE_TRUST_TRANSITION_HPP

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace inferedge {

struct ProjectId {
    std::string value;
};

struct IdOrNone {
    bool is_none = true;
    ProjectId id;
};

struct CivilDate {
    std::uint16_t year = 0;
    std::uint8_t month = 0;
    std::uint8_t day = 0;
};

struct Sha256Digest {
    std::array<std::uint8_t, 32> bytes{};
};

enum class TrustPublicationResult : std::uint8_t {
    kOk = 1,
    kBusyMutex = 2,
    kBusyKernelLock = 3,
    kReentered = 4,
    kRootRejected = 5,
    kFilesystemUnsupported = 6,
    kFilesystemIndeterminate = 7,
    kInterrupted = 8,
    kAllocationFailed = 9,
    kKernelLockInvariant = 10,
    kKernelLockResource = 11,
    kKernelLockIo = 12,
    kConflict = 13,
    kAuthorizationRejected = 14,
    kPrefixMismatch = 15,
    kReleaseFailedBeforeDurable = 16,
    kOutcomeUncertain = 17,
    kReleaseInvariant = 18,
    kDescriptorInvariant = 19,
    kPoisoned = 20
};

enum class TrustPublicationKind : std::uint8_t {
    kWithdrawal = 1,
    kOrdinary = 2,
    kRevocation = 3,
    kRollback = 4,
    kNativeInitialization = 5,
    kImportedInitialization = 6,
    kExportFreeze = 7,
    kRetirement = 8,
    kGovernancePin = 9,
    kActivation = 10
};

enum class TrustPublicationStage : std::uint8_t {
    kNone = 0,
    kRootUnbound = 1,
    kRootBound = 2,
    kP0 = 3,
    kP1 = 4,
    kP2 = 5,
    kP3 = 6,
    kP4 = 7,
    kP5 = 8,
    kP6 = 9,
    kP7 = 10,
    kP8 = 11,
    kP9 = 12,
    kN0 = 13,
    kN1 = 14,
    kN2 = 15,
    kN3 = 16,
    kN4 = 17,
    kN5 = 18,
    kN6 = 19,
    kN7 = 20,
    kN8 = 21,
    kI0 = 22,
    kI1 = 23,
    kI2 = 24,
    kI3 = 25,
    kS0 = 26,
    kS1 = 27,
    kS2 = 28,
    kS3 = 29
};

enum class SignatureFamily : std::uint8_t {
    kTrustTransition = 1,
    kLineageOrigin = 2,
    kWithdrawal = 3,
    kReceipt = 4,
    kExportAuthorization = 5,
    kGovernanceContent = 6,
    kGovernanceReview = 7,
    kActivationReview = 8
};

enum class LineageKind : std::uint8_t { kNone = 0, kNative = 1, kImported = 2 };

enum class ServiceStatus : std::uint8_t {
    kEmpty = 1,
    kActive = 2,
    kExportFrozen = 3,
    kRetired = 4
};

enum class WithdrawalSubjectKind : std::uint8_t {
    kProvenance = 1,
    kLicense = 2,
    kGoldenFixture = 3,
    kRegistryEntry = 4,
    kPolicyPin = 5,
    kTelemetryGovernance = 6,
    kActivationAuthorization = 7
};

enum class TrustRole : std::uint8_t {
    kProvenanceReviewer = 1,
    kLicenseReviewer = 2,
    kGoldenReviewer = 3,
    kRegistryReviewer = 4,
    kPolicyReviewer = 5,
    kTelemetryReviewer = 6,
    kTrustAdmin = 7
};

enum class RuntimeMode : std::uint8_t {
    kDisabled = 1,
    kRecommendOnly = 2,
    kShadow = 3,
    kActivation = 4
};

enum class ActivationMode : std::uint8_t { kInactive = 1, kActive = 2 };

enum class AdmissionMode : std::uint8_t { kNoModel = 1, kActive = 2 };

enum class PredecessorCurrent : std::uint8_t { kAbsent = 1, kExactBytes = 2 };
enum class SuccessorDurability : std::uint8_t { kMustCreate = 1, kAlreadyExact = 2 };

enum class FilesystemClass : std::uint8_t {
    kApfs = 1,
    kExtFamily = 2,
    kTmpfs = 3,
    kUnsupported = 4,
    kIndeterminate = 5
};

struct FileIdentity {
    IdOrNone id;
    std::string canonical_relative_path;
    std::uint64_t size_bytes = 0;
    Sha256Digest sha256{};
};

struct WriterBoundIdentity {
    Sha256Digest portable_plan_sha256{};
    std::uint64_t device = 0;
    std::uint64_t inode = 0;
    std::string canonical_path;
    ProjectId host_id;
    Sha256Digest writer_bound_sha256{};
};

struct PortablePublicationPlan {
    TrustPublicationKind kind = TrustPublicationKind::kOrdinary;
    IdOrNone status_operation_id;
    IdOrNone export_operation_id;
    ProjectId trust_id;
    IdOrNone transition_id;
    ProjectId host_id;
    std::string requested_root_path;
    std::string binding_record_kind;
    Sha256Digest new_binding_record_sha256{};
    std::uint64_t new_binding_record_size = 0;
    Sha256Digest source_binding_record_sha256{};
    std::uint64_t source_binding_record_size = 0;
    std::uint64_t successor_file_size = 0;
    Sha256Digest successor_file_sha256{};
    std::optional<ServiceStatus> expected_status;
    std::optional<ServiceStatus> requested_status;
    Sha256Digest portable_plan_sha256{};
};

struct ReleaseDiagnostic {
    bool lock_un_attempted = false;
    int lock_un_errno = 0;
    bool close_attempted = false;
    int close_errno = 0;
    bool poisoned = false;
};

struct TrustPublicationOutcome {
    TrustPublicationResult result = TrustPublicationResult::kPoisoned;
    TrustPublicationStage stage = TrustPublicationStage::kNone;
    ReleaseDiagnostic release{};
    Sha256Digest operation_identity{};
    bool bytes_written = false;
    bool durable_files_may_exist = false;
};

struct FlockHookResult {
    int rc = -1;
    int err_no = 0;
};

struct CloseHookResult {
    int rc = -1;
    int err_no = 0;
};

struct PublicationHooks {
    FilesystemClass (*classify)(std::uint64_t device, std::uint64_t inode) = nullptr;
    FlockHookResult (*flock_op)(int fd, int operation) = nullptr;
    CloseHookResult (*close_fd)(int fd) = nullptr;
};

struct WithdrawalRequest {
    std::string trust_root_path;
    std::string admission_root_path;
    std::string evidence_root_path;
    ProjectId host_id;
    WithdrawalSubjectKind subject_kind = WithdrawalSubjectKind::kProvenance;
    ProjectId subject_id;
    TrustRole signer_role = TrustRole::kProvenanceReviewer;
    ProjectId signer_key_id;
    CivilDate effective_date{};
    std::array<std::uint8_t, 32> public_key{};
    FileIdentity note;
    std::string note_bytes;
    std::array<std::uint8_t, 64> note_signature{};
    FileIdentity note_signature_record;
    FileIdentity note_signature_index;
    FileIdentity note_verification;
    FileIdentity predecessor;
    std::uint32_t predecessor_generation = 0;
    std::uint64_t predecessor_revocation_generation = 0;
    std::vector<ProjectId> predecessor_tombstones;
    FileIdentity successor;
    std::uint32_t successor_generation = 0;
    std::uint64_t successor_revocation_generation = 0;
    std::vector<ProjectId> successor_tombstones;
    FileIdentity transition;
    FileIdentity lineage_record;
    FileIdentity registry_entry;
    std::string successor_bytes;
    std::string transition_record_bytes;
    std::array<std::uint8_t, 64> transition_signature{};
    std::string signature_index_bytes;
    std::string verification_record_bytes;
    std::string predecessor_current_bytes;
    std::string authorization_preimage_bytes;
};

struct OrdinaryTransitionRequest {
    std::string trust_root_path;
    std::string admission_root_path;
    ProjectId host_id;
    TrustPublicationKind kind = TrustPublicationKind::kOrdinary;
    FileIdentity authorization;
    std::string authorization_preimage_bytes;
    ProjectId signer_key_id;
    TrustRole signer_role = TrustRole::kTrustAdmin;
    CivilDate effective_date{};
    std::array<std::uint8_t, 32> public_key{};
    FileIdentity predecessor;
    std::uint32_t predecessor_generation = 0;
    std::uint64_t predecessor_revocation_generation = 0;
    std::vector<ProjectId> predecessor_tombstones;
    FileIdentity predecessor_registry;
    FileIdentity successor;
    std::uint32_t successor_generation = 0;
    std::uint64_t successor_revocation_generation = 0;
    std::vector<ProjectId> successor_tombstones;
    FileIdentity transition;
    LineageKind lineage_kind = LineageKind::kNone;
    std::string successor_bytes;
    std::string transition_record_bytes;
    std::array<std::uint8_t, 64> transition_signature{};
    std::string signature_index_bytes;
    std::string verification_record_bytes;
    std::string predecessor_current_bytes;
};

struct ImportedInitializationRequest;
struct ExportFreezeRequest;
struct RetirementRequest;
struct NativeInitializationRequest;
struct RollbackRequest;
struct GovernancePinRequest;
struct ActivationPublicationRequest;

#if defined(INFEREDGE_TRUST_PUBLICATION_TEST_SEAM)
struct StatusPublicationTestRequest;
#endif

class VerifiedTransitionPlan;
class VerifiedStatusPlan;

TrustPublicationOutcome publish_verified_transition(VerifiedTransitionPlan&& plan) noexcept;
TrustPublicationOutcome publish_verified_status(VerifiedStatusPlan&& plan) noexcept;
TrustPublicationOutcome publish_withdrawal(const WithdrawalRequest&) noexcept;
TrustPublicationOutcome publish_ordinary_transition(const OrdinaryTransitionRequest&) noexcept;
TrustPublicationOutcome
publish_imported_initialization(const ImportedInitializationRequest&) noexcept;
TrustPublicationOutcome publish_export_freeze(const ExportFreezeRequest&) noexcept;
TrustPublicationOutcome publish_retirement(const RetirementRequest&) noexcept;
TrustPublicationOutcome publish_native_initialization(const NativeInitializationRequest&) noexcept;
TrustPublicationOutcome publish_rollback(const RollbackRequest&) noexcept;
TrustPublicationOutcome publish_governance_pin(const GovernancePinRequest&) noexcept;
TrustPublicationOutcome publish_activation(const ActivationPublicationRequest&) noexcept;

std::optional<TrustPublicationKind> parse_trust_publication_kind(std::string_view) noexcept;
std::string_view canonical_trust_publication_kind(TrustPublicationKind) noexcept;
std::optional<SignatureFamily> parse_signature_family(std::string_view) noexcept;
std::string_view canonical_signature_family(SignatureFamily) noexcept;
std::optional<LineageKind> parse_lineage_kind(std::string_view) noexcept;
std::string_view canonical_lineage_kind(LineageKind) noexcept;
std::optional<ServiceStatus> parse_service_status(std::string_view) noexcept;
std::string_view canonical_service_status(ServiceStatus) noexcept;
std::optional<WithdrawalSubjectKind> parse_withdrawal_subject_kind(std::string_view) noexcept;
std::string_view canonical_withdrawal_subject_kind(WithdrawalSubjectKind) noexcept;
std::optional<TrustRole> parse_trust_role(std::string_view) noexcept;
std::string_view canonical_trust_role(TrustRole) noexcept;
std::optional<RuntimeMode> parse_runtime_mode(std::string_view) noexcept;
std::string_view canonical_runtime_mode(RuntimeMode) noexcept;
std::optional<ActivationMode> parse_activation_mode(std::string_view) noexcept;
std::string_view canonical_activation_mode(ActivationMode) noexcept;
std::optional<AdmissionMode> parse_admission_mode(std::string_view) noexcept;
std::string_view canonical_admission_mode(AdmissionMode) noexcept;
bool project_id_ok(std::string_view) noexcept;
std::string canonical_sha256(const Sha256Digest&) noexcept;
bool sha256_from_hex(std::string_view, Sha256Digest&) noexcept;

class VerifiedTransitionPlan {
public:
    VerifiedTransitionPlan() = delete;
    VerifiedTransitionPlan(const VerifiedTransitionPlan&) = delete;
    VerifiedTransitionPlan& operator=(const VerifiedTransitionPlan&) = delete;
    VerifiedTransitionPlan& operator=(VerifiedTransitionPlan&&) = delete;

private:
    struct Data {
        PortablePublicationPlan portable;
        std::string successor_bytes;
        std::string transition_record_bytes;
        std::array<std::uint8_t, 64> signature_blob{};
        std::string signature_index_bytes;
        std::string verification_record_bytes;
        std::string authorization_preimage_bytes;
        std::array<std::uint8_t, 32> public_key{};
        std::string lineage_record_bytes;
        std::array<std::uint8_t, 64> lineage_signature_blob{};
        std::string lineage_index_bytes;
        std::string lineage_verification_bytes;
        PredecessorCurrent predecessor_current = PredecessorCurrent::kExactBytes;
        std::string predecessor_current_bytes;
        SuccessorDurability successor_durability = SuccessorDurability::kMustCreate;
    };
    Data data_;
    explicit VerifiedTransitionPlan(Data data) noexcept : data_(std::move(data)) {}
    VerifiedTransitionPlan(VerifiedTransitionPlan&&) noexcept = default;
    ~VerifiedTransitionPlan() = default;

    friend TrustPublicationOutcome publish_verified_transition(VerifiedTransitionPlan&&) noexcept;
    friend TrustPublicationOutcome publish_withdrawal(const WithdrawalRequest&) noexcept;
    friend TrustPublicationOutcome
    publish_ordinary_transition(const OrdinaryTransitionRequest&) noexcept;
    friend TrustPublicationOutcome
    publish_imported_initialization(const ImportedInitializationRequest&) noexcept;
    friend TrustPublicationOutcome
    publish_native_initialization(const NativeInitializationRequest&) noexcept;
    friend TrustPublicationOutcome publish_rollback(const RollbackRequest&) noexcept;
    friend TrustPublicationOutcome publish_governance_pin(const GovernancePinRequest&) noexcept;
    friend TrustPublicationOutcome publish_activation(const ActivationPublicationRequest&) noexcept;
};

class VerifiedStatusPlan {
public:
    VerifiedStatusPlan() = delete;
    VerifiedStatusPlan(const VerifiedStatusPlan&) = delete;
    VerifiedStatusPlan& operator=(const VerifiedStatusPlan&) = delete;
    VerifiedStatusPlan& operator=(VerifiedStatusPlan&&) = delete;

private:
    struct Data {
        PortablePublicationPlan portable;
        std::string expected_status_bytes;
        std::string requested_status_bytes;
        std::string status_path;
        std::string status_partial_path;
        std::string authorization_preimage_bytes;
        std::string signature_record_bytes;
        std::array<std::uint8_t, 64> signature_blob{};
        std::string signature_index_bytes;
        std::string verification_record_bytes;
        std::array<std::uint8_t, 32> public_key{};
        std::string current_bytes;
        std::string portable_authorization_prefix;
    };
    Data data_;
    explicit VerifiedStatusPlan(Data data) noexcept : data_(std::move(data)) {}
    VerifiedStatusPlan(VerifiedStatusPlan&&) noexcept = default;
    ~VerifiedStatusPlan() = default;

    friend TrustPublicationOutcome publish_verified_status(VerifiedStatusPlan&&) noexcept;
    friend TrustPublicationOutcome publish_export_freeze(const ExportFreezeRequest&) noexcept;
    friend TrustPublicationOutcome publish_retirement(const RetirementRequest&) noexcept;
    friend TrustPublicationOutcome
    publish_native_initialization(const NativeInitializationRequest&) noexcept;
    friend TrustPublicationOutcome
    publish_imported_initialization(const ImportedInitializationRequest&) noexcept;
#if defined(INFEREDGE_TRUST_PUBLICATION_TEST_SEAM)
    friend TrustPublicationOutcome
    publish_verified_status_for_test(StatusPublicationTestRequest request) noexcept;
#endif
};

#if defined(INFEREDGE_TRUST_PUBLICATION_TEST_SEAM)

struct StatusPublicationTestRequest {
    PortablePublicationPlan portable{};
    std::string expected_status_bytes;
    std::string requested_status_bytes;
    std::string status_path;
    std::string status_partial_path;
    std::string authorization_preimage_bytes;
    std::string signature_record_bytes;
    std::array<std::uint8_t, 64> signature_blob{};
    std::string signature_index_bytes;
    std::string verification_record_bytes;
    std::array<std::uint8_t, 32> public_key{};
    std::string current_bytes;
    std::string portable_authorization_prefix;
};

TrustPublicationOutcome
publish_verified_status_for_test(StatusPublicationTestRequest request) noexcept;

bool set_publication_hooks_for_test(const PublicationHooks& hooks) noexcept;

#endif

static_assert(!std::is_default_constructible_v<VerifiedTransitionPlan>);
static_assert(!std::is_copy_constructible_v<VerifiedTransitionPlan>);
static_assert(!std::is_move_constructible_v<VerifiedTransitionPlan>);
static_assert(!std::is_move_assignable_v<VerifiedTransitionPlan>);
static_assert(!std::is_destructible_v<VerifiedTransitionPlan>);
static_assert(!std::is_default_constructible_v<VerifiedStatusPlan>);
static_assert(!std::is_copy_constructible_v<VerifiedStatusPlan>);
static_assert(!std::is_move_constructible_v<VerifiedStatusPlan>);
static_assert(!std::is_destructible_v<VerifiedStatusPlan>);

} // namespace inferedge

#endif
