#include "inferedge/ed25519_verify.hpp"

#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

std::vector<std::uint8_t> from_hex(std::string_view text) {
    std::vector<std::uint8_t> out;
    out.reserve(text.size() / 2);
    for (std::size_t index = 0; index < text.size(); index += 2) {
        unsigned value = 0;
        const std::string byte(text.substr(index, 2));
        value = static_cast<unsigned>(std::stoul(byte, nullptr, 16));
        out.push_back(static_cast<std::uint8_t>(value));
    }
    return out;
}

inferedge::VerificationRequest request_for(std::span<const std::uint8_t> message,
                                           std::span<const std::uint8_t> signature,
                                           std::span<const std::uint8_t> key) {
    inferedge::VerificationRequest request;
    request.message = message;
    request.signature = signature;
    request.public_key = key;
    request.authorization_digest = "authorization-digest";
    request.signature_family_id = "stale_lock_authorization";
    request.terminal_trust_state_id = "trust-v1";
    request.terminal_effective_date = "2026-06-15";
    request.registry_entry_id = "admin-v1";
    request.signer_key_id = "admin-v1";
    request.key_valid_from = "2020-01-01";
    request.key_valid_until = "2100-12-31";
    request.role = "trust_admin";
    request.revocation_result = "unrevoked";
    request.verification_date = "2026-06-15";
    request.operation = "stale-lock";
    request.canonical_authorization_path = "lock/stale-auth.record";
    request.target_id = "lock/evidence.lock";
    return request;
}

void expect_rejected_without_authority(const inferedge::VerificationDecision& decision) {
    EXPECT_EQ(decision.status, inferedge::VerificationStatus::kRejected);
    EXPECT_FALSE(decision.accepted());
    EXPECT_TRUE(decision.authorization_digest.empty());
    EXPECT_TRUE(decision.signature_family_id.empty());
    EXPECT_TRUE(decision.terminal_trust_state_id.empty());
    EXPECT_TRUE(decision.signer_key_id.empty());
    EXPECT_TRUE(decision.role.empty());
    EXPECT_TRUE(decision.operation.empty());
    EXPECT_TRUE(decision.target_id.empty());
}

} // namespace

TEST(Ed25519Verify, Rfc8032EmptyMessage) {
    const auto key = from_hex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
    const auto signature = from_hex("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901"
                                    "555fb8821590a33bacc61e39701cf9"
                                    "b46bd25bf5f0595bbe24655141438e7a100b");
    const auto decision = inferedge::verify_ed25519(request_for({}, signature, key));
    EXPECT_TRUE(decision.accepted());
    EXPECT_EQ(decision.authorization_digest, "authorization-digest");
    EXPECT_EQ(decision.signature_family_id, "stale_lock_authorization");
    EXPECT_EQ(decision.terminal_trust_state_id, "trust-v1");
    EXPECT_EQ(decision.terminal_effective_date, "2026-06-15");
    EXPECT_EQ(decision.registry_entry_id, "admin-v1");
    EXPECT_EQ(decision.signer_key_id, "admin-v1");
    EXPECT_EQ(decision.key_valid_from, "2020-01-01");
    EXPECT_EQ(decision.key_valid_until, "2100-12-31");
    EXPECT_EQ(decision.role, "trust_admin");
    EXPECT_EQ(decision.revocation_result, "unrevoked");
    EXPECT_EQ(decision.verification_date, "2026-06-15");
    EXPECT_EQ(decision.operation, "stale-lock");
    EXPECT_EQ(decision.canonical_authorization_path, "lock/stale-auth.record");
    EXPECT_EQ(decision.target_id, "lock/evidence.lock");
}

TEST(Ed25519Verify, Rfc8032OneByteMessage) {
    const auto key = from_hex("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c");
    const auto signature = from_hex("92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69"
                                    "da085ac1e43e15996e458f3613d0f1"
                                    "1d8c387b2eaeb4302aeeb00d291612bb0c00");
    const std::uint8_t message[] = {0x72};
    EXPECT_TRUE(inferedge::verify_ed25519(request_for(message, signature, key)).accepted());
}

TEST(Ed25519Verify, Rfc8032TwoByteMessage) {
    const auto key = from_hex("fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025");
    const auto signature = from_hex("6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3"
                                    "ac18ff9b538d16f290ae67f760984dc"
                                    "6594a7c15e9716ed28dc027beceea1ec40a");
    const std::uint8_t message[] = {0xaf, 0x82};
    EXPECT_TRUE(inferedge::verify_ed25519(request_for(message, signature, key)).accepted());
}

TEST(Ed25519Verify, AlteredMessageIsRejected) {
    const auto key = from_hex("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c");
    const auto signature = from_hex("92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69"
                                    "da085ac1e43e15996e458f3613d0f1"
                                    "1d8c387b2eaeb4302aeeb00d291612bb0c00");
    const std::uint8_t message[] = {0x73};
    expect_rejected_without_authority(
        inferedge::verify_ed25519(request_for(message, signature, key)));
}

TEST(Ed25519Verify, AlteredSignatureIsRejected) {
    auto signature = from_hex("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8"
                              "821590a33bacc61e39701cf9"
                              "b46bd25bf5f0595bbe24655141438e7a100b");
    const auto original = signature;
    signature[0] ^= 0x01;
    const auto key = from_hex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
    expect_rejected_without_authority(inferedge::verify_ed25519(request_for({}, signature, key)));
    EXPECT_NE(signature, original);
}

TEST(Ed25519Verify, AlteredPublicKeyIsRejected) {
    auto key = from_hex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
    key[0] ^= 0x01;
    const auto signature = from_hex("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901"
                                    "555fb8821590a33bacc61e39701cf9"
                                    "b46bd25bf5f0595bbe24655141438e7a100b");
    expect_rejected_without_authority(inferedge::verify_ed25519(request_for({}, signature, key)));
}

TEST(Ed25519Verify, TruncatedAndOversizedLengthsAreMalformed) {
    const auto key = from_hex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
    auto signature = from_hex("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8"
                              "821590a33bacc61e39701cf9"
                              "b46bd25bf5f0595bbe24655141438e7a100b");
    auto short_signature = signature;
    short_signature.pop_back();
    auto long_signature = signature;
    long_signature.push_back(0);
    auto short_key = key;
    short_key.pop_back();
    auto long_key = key;
    long_key.push_back(0);
    EXPECT_EQ(inferedge::verify_ed25519(request_for({}, short_signature, key)).status,
              inferedge::VerificationStatus::kMalformed);
    EXPECT_EQ(inferedge::verify_ed25519(request_for({}, long_signature, key)).status,
              inferedge::VerificationStatus::kMalformed);
    EXPECT_EQ(inferedge::verify_ed25519(request_for({}, signature, short_key)).status,
              inferedge::VerificationStatus::kMalformed);
    EXPECT_EQ(inferedge::verify_ed25519(request_for({}, signature, long_key)).status,
              inferedge::VerificationStatus::kMalformed);
}

TEST(Ed25519Verify, MissingSignatureOrKeyIsUnavailable) {
    const auto key = from_hex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
    const auto signature = from_hex("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901"
                                    "555fb8821590a33bacc61e39701cf9"
                                    "b46bd25bf5f0595bbe24655141438e7a100b");
    EXPECT_EQ(inferedge::verify_ed25519(request_for({}, {}, key)).status,
              inferedge::VerificationStatus::kUnavailable);
    EXPECT_EQ(inferedge::verify_ed25519(request_for({}, signature, {})).status,
              inferedge::VerificationStatus::kUnavailable);
}

TEST(Ed25519Verify, ScalarMalleabilityIsRejected) {
    const auto key = from_hex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
    const auto signature = from_hex("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901"
                                    "554c8c7872aa064e049dbb3013fbf2"
                                    "9380d25bf5f0595bbe24655141438e7a101b");
    expect_rejected_without_authority(inferedge::verify_ed25519(request_for({}, signature, key)));
}

TEST(Ed25519Verify, NoncanonicalScalarIsRejected) {
    const auto key = from_hex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
    const auto signature = from_hex("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901"
                                    "554c8c7872aa064e049dbb3013fbf2"
                                    "9380d25bf5f0595bbe24655141438e7a101b");
    expect_rejected_without_authority(inferedge::verify_ed25519(request_for({}, signature, key)));
}

TEST(Ed25519Verify, NoncanonicalPointEncodingsAreRejected) {
    const auto prime = from_hex("edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f");
    const auto signature = from_hex("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901"
                                    "555fb8821590a33bacc61e39701cf9"
                                    "b46bd25bf5f0595bbe24655141438e7a100b");
    auto noncanonical_r = signature;
    std::memcpy(noncanonical_r.data(), prime.data(), 32);
    const auto key = from_hex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
    expect_rejected_without_authority(
        inferedge::verify_ed25519(request_for({}, noncanonical_r, key)));
    expect_rejected_without_authority(inferedge::verify_ed25519(request_for({}, signature, prime)));
}

TEST(Ed25519Verify, SmallOrderAndZeroEncodingsAreRejected) {
    const auto key = from_hex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
    const auto signature = from_hex("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901"
                                    "555fb8821590a33bacc61e39701cf9"
                                    "b46bd25bf5f0595bbe24655141438e7a100b");
    std::vector<std::uint8_t> identity(32, 0);
    identity[0] = 1;
    std::vector<std::uint8_t> zero(32, 0);
    auto small_r = signature;
    std::memcpy(small_r.data(), identity.data(), 32);
    expect_rejected_without_authority(
        inferedge::verify_ed25519(request_for({}, signature, identity)));
    expect_rejected_without_authority(inferedge::verify_ed25519(request_for({}, signature, zero)));
    expect_rejected_without_authority(inferedge::verify_ed25519(request_for({}, small_r, key)));
    expect_rejected_without_authority(
        inferedge::verify_ed25519(request_for({}, std::vector<std::uint8_t>(64, 0), key)));
}

TEST(Ed25519Verify, RepeatedVerificationDoesNotMutateInputs) {
    const auto key = from_hex("fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025");
    const auto signature = from_hex("6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3"
                                    "ac18ff9b538d16f290ae67f760984dc"
                                    "6594a7c15e9716ed28dc027beceea1ec40a");
    std::vector<std::uint8_t> message = {0xaf, 0x82};
    const auto key_copy = key;
    const auto signature_copy = signature;
    const auto message_copy = message;
    const auto first = inferedge::verify_ed25519(request_for(message, signature, key));
    const auto second = inferedge::verify_ed25519(request_for(message, signature, key));
    EXPECT_TRUE(first.accepted());
    EXPECT_TRUE(second.accepted());
    EXPECT_EQ(first.target_id, second.target_id);
    EXPECT_EQ(key, key_copy);
    EXPECT_EQ(signature, signature_copy);
    EXPECT_EQ(message, message_copy);
}

TEST(Ed25519Verify, AcceptedDecisionMissingFieldsIsMalformed) {
    const auto key = from_hex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
    const auto signature = from_hex("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901"
                                    "555fb8821590a33bacc61e39701cf9"
                                    "b46bd25bf5f0595bbe24655141438e7a100b");
    auto request = request_for({}, signature, key);
    request.authorization_digest = {};
    const auto decision = inferedge::verify_ed25519(request);
    EXPECT_EQ(decision.status, inferedge::VerificationStatus::kMalformed);
    EXPECT_FALSE(decision.accepted());
    EXPECT_TRUE(decision.role.empty());
    EXPECT_TRUE(decision.signer_key_id.empty());
}
