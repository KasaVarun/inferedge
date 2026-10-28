#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace inferedge {

// Phase 9-C7 Ed25519 verification decision. This header does not construct an
// AuthenticatedCapability, expose a stamp, or provide a signing operation.

inline constexpr std::string_view kEd25519VerifierId{"inferedge-ed25519-v1"};

enum class VerificationStatus : std::uint8_t {
    kAccepted = 1,
    kRejected = 2,
    kUnavailable = 3,
    kMalformed = 4,
};

// Identity views are claims. A decision copies them only after the signature
// check succeeds, and only when every view is non-empty. They are not proof.
struct VerificationRequest {
    std::span<const std::uint8_t> message;
    std::span<const std::uint8_t> signature;
    std::span<const std::uint8_t> public_key;
    std::string_view authorization_digest;
    std::string_view signature_family_id;
    std::string_view terminal_trust_state_id;
    std::string_view terminal_effective_date;
    std::string_view registry_entry_id;
    std::string_view signer_key_id;
    std::string_view key_valid_from;
    std::string_view key_valid_until;
    std::string_view role;
    std::string_view revocation_result;
    std::string_view verification_date;
    std::string_view operation;
    std::string_view canonical_authorization_path;
    std::string_view target_id;
};

struct VerificationDecision {
    VerificationStatus status{VerificationStatus::kMalformed};
    std::string authorization_digest;
    std::string signature_family_id;
    std::string terminal_trust_state_id;
    std::string terminal_effective_date;
    std::string registry_entry_id;
    std::string signer_key_id;
    std::string key_valid_from;
    std::string key_valid_until;
    std::string role;
    std::string revocation_result;
    std::string verification_date;
    std::string operation;
    std::string canonical_authorization_path;
    std::string target_id;

    bool accepted() const noexcept;
};

// The one strong verifier. A missing definition is an unresolved symbol.
VerificationDecision verify_ed25519(const VerificationRequest& request);

} // namespace inferedge
