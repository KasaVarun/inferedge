#include "inferedge/ed25519_verify.hpp"

#include <array>
#include <cstring>

namespace inferedge {
namespace {

// Field and group arithmetic follows the public-domain TweetNaCl verification
// path (Daniel J. Bernstein, Bernard van Gastel, Wesley Janssen, Tanja Lange,
// Peter Schwabe, and Sjaak Smetsers), https://tweetnacl.cr.yp.to.
// Canonical scalar and point checks, and small-order rejection, follow RFC 8032.
// SHA-512 is the FIPS 180-4 function. This translation unit verifies only.
// It has no signing operation and no private-key or seed input.

using Limb = long long;
using Fe = Limb[16];

constexpr Fe kFe0 = {0};
constexpr Fe kFe1 = {1};
constexpr Fe kD = {0x78a3, 0x1359, 0x4dca, 0x75eb, 0xd8ab, 0x4141, 0x0a4d, 0x0070,
                   0xe898, 0x7779, 0x4079, 0x8cc7, 0xfe73, 0x2b6f, 0x6cee, 0x5203};
constexpr Fe kD2 = {0xf159, 0x26b2, 0x9b94, 0xebd6, 0xb156, 0x8283, 0x149a, 0x00e0,
                    0xd130, 0xeef3, 0x80f2, 0x198e, 0xfce7, 0x56df, 0xd9dc, 0x2406};
constexpr Fe kX = {0xd51a, 0x8f25, 0x2d60, 0xc956, 0xa7b2, 0x9525, 0xc760, 0x692c,
                   0xdc5c, 0xfdd6, 0xe231, 0xc0a4, 0x53fe, 0xcd6e, 0x36d3, 0x2169};
constexpr Fe kY = {0x6658, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666,
                   0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666};
constexpr Fe kI = {0xa0b0, 0x4a0e, 0x1b27, 0xc4ee, 0xe478, 0xad2f, 0x1806, 0x2f43,
                   0xd7a7, 0x3dfb, 0x0099, 0x2b4d, 0xdf0b, 0x4fc1, 0x2480, 0x2b83};
constexpr Limb kOrder[32] = {0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7,
                             0xa2, 0xde, 0xf9, 0xde, 0x14, 0,    0,    0,    0,    0,    0,
                             0,    0,    0,    0,    0,    0,    0,    0,    0,    0x10};

constexpr std::uint64_t kShaK[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL};

std::uint64_t rotr(std::uint64_t value, int bits) {
    return (value >> bits) | (value << (64 - bits));
}

class Sha512 {
public:
    Sha512() { reset(); }

    void update(const std::uint8_t* data, std::size_t size) {
        bits_ += static_cast<std::uint64_t>(size) * 8;
        while (size > 0) {
            const std::size_t room = 128 - used_;
            const std::size_t take = size < room ? size : room;
            std::memcpy(block_ + used_, data, take);
            used_ += take;
            data += take;
            size -= take;
            if (used_ == 128) {
                compress(block_);
                used_ = 0;
            }
        }
    }

    void final(std::uint8_t out[64]) {
        block_[used_++] = 0x80;
        if (used_ > 112) {
            while (used_ < 128) {
                block_[used_++] = 0;
            }
            compress(block_);
            used_ = 0;
        }
        while (used_ < 112) {
            block_[used_++] = 0;
        }
        for (int index = 0; index < 8; ++index) {
            block_[112 + index] = 0;
            block_[120 + index] = static_cast<std::uint8_t>((bits_ >> (56 - 8 * index)) & 0xff);
        }
        compress(block_);
        for (int index = 0; index < 8; ++index) {
            for (int byte = 0; byte < 8; ++byte) {
                out[index * 8 + byte] =
                    static_cast<std::uint8_t>((state_[index] >> (56 - byte * 8)) & 0xff);
            }
        }
    }

private:
    void reset() {
        state_[0] = 0x6a09e667f3bcc908ULL;
        state_[1] = 0xbb67ae8584caa73bULL;
        state_[2] = 0x3c6ef372fe94f82bULL;
        state_[3] = 0xa54ff53a5f1d36f1ULL;
        state_[4] = 0x510e527fade682d1ULL;
        state_[5] = 0x9b05688c2b3e6c1fULL;
        state_[6] = 0x1f83d9abfb41bd6bULL;
        state_[7] = 0x5be0cd19137e2179ULL;
        bits_ = 0;
        used_ = 0;
    }

    void compress(const std::uint8_t block[128]) {
        std::uint64_t words[80];
        for (int index = 0; index < 16; ++index) {
            words[index] = 0;
            for (int byte = 0; byte < 8; ++byte) {
                words[index] = (words[index] << 8) | block[index * 8 + byte];
            }
        }
        for (int index = 16; index < 80; ++index) {
            const std::uint64_t s0 =
                rotr(words[index - 15], 1) ^ rotr(words[index - 15], 8) ^ (words[index - 15] >> 7);
            const std::uint64_t s1 =
                rotr(words[index - 2], 19) ^ rotr(words[index - 2], 61) ^ (words[index - 2] >> 6);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        std::uint64_t a = state_[0];
        std::uint64_t b = state_[1];
        std::uint64_t c = state_[2];
        std::uint64_t d = state_[3];
        std::uint64_t e = state_[4];
        std::uint64_t f = state_[5];
        std::uint64_t g = state_[6];
        std::uint64_t h = state_[7];
        for (int index = 0; index < 80; ++index) {
            const std::uint64_t s1 = rotr(e, 14) ^ rotr(e, 18) ^ rotr(e, 41);
            const std::uint64_t ch = (e & f) ^ ((~e) & g);
            const std::uint64_t temp1 = h + s1 + ch + kShaK[index] + words[index];
            const std::uint64_t s0 = rotr(a, 28) ^ rotr(a, 34) ^ rotr(a, 39);
            const std::uint64_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint64_t temp2 = s0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::uint64_t state_[8]{};
    std::uint64_t bits_{0};
    std::uint8_t block_[128]{};
    std::size_t used_{0};
};

void set25519(Fe out, const Fe in) {
    for (int index = 0; index < 16; ++index) {
        out[index] = in[index];
    }
}

void car25519(Fe out) {
    for (int index = 0; index < 16; ++index) {
        out[index] += static_cast<Limb>(1) << 16;
        const Limb carry = out[index] >> 16;
        out[(index + 1) * (index < 15)] += carry - 1 + 37 * (carry - 1) * (index == 15);
        out[index] -= carry << 16;
    }
}

void sel25519(Fe left, Fe right, int bit) {
    const Limb mask = ~(static_cast<Limb>(bit) - 1);
    for (int index = 0; index < 16; ++index) {
        const Limb mixed = mask & (left[index] ^ right[index]);
        left[index] ^= mixed;
        right[index] ^= mixed;
    }
}

void pack25519(unsigned char* out, const Fe in) {
    Fe t{};
    set25519(t, in);
    car25519(t);
    car25519(t);
    car25519(t);
    for (int round = 0; round < 2; ++round) {
        Fe m{};
        m[0] = t[0] - 0xffed;
        for (int index = 1; index < 15; ++index) {
            m[index] = t[index] - 0xffff - ((m[index - 1] >> 16) & 1);
            m[index - 1] &= 0xffff;
        }
        m[15] = t[15] - 0x7fff - ((m[14] >> 16) & 1);
        const int bit = static_cast<int>((m[15] >> 16) & 1);
        m[14] &= 0xffff;
        sel25519(t, m, 1 - bit);
    }
    for (int index = 0; index < 16; ++index) {
        out[2 * index] = static_cast<unsigned char>(t[index] & 0xff);
        out[2 * index + 1] = static_cast<unsigned char>(t[index] >> 8);
    }
}

int neq25519(const Fe left, const Fe right) {
    unsigned char a[32];
    unsigned char b[32];
    pack25519(a, left);
    pack25519(b, right);
    unsigned char diff = 0;
    for (int index = 0; index < 32; ++index) {
        diff |= static_cast<unsigned char>(a[index] ^ b[index]);
    }
    return diff;
}

unsigned char par25519(const Fe value) {
    unsigned char packed[32];
    pack25519(packed, value);
    return packed[0] & 1;
}

void unpack25519(Fe out, const unsigned char* in) {
    for (int index = 0; index < 16; ++index) {
        out[index] = in[2 * index] + (static_cast<Limb>(in[2 * index + 1]) << 8);
    }
    out[15] &= 0x7fff;
}

void add_fe(Fe out, const Fe left, const Fe right) {
    for (int index = 0; index < 16; ++index) {
        out[index] = left[index] + right[index];
    }
}

void sub_fe(Fe out, const Fe left, const Fe right) {
    for (int index = 0; index < 16; ++index) {
        out[index] = left[index] - right[index];
    }
}

void mul_fe(Fe out, const Fe left, const Fe right) {
    Limb product[31] = {0};
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            product[i + j] += left[i] * right[j];
        }
    }
    for (int index = 0; index < 15; ++index) {
        product[index] += 38 * product[index + 16];
    }
    for (int index = 0; index < 16; ++index) {
        out[index] = product[index];
    }
    car25519(out);
    car25519(out);
}

void square_fe(Fe out, const Fe in) {
    mul_fe(out, in, in);
}

void inv25519(Fe out, const Fe in) {
    Fe c{};
    set25519(c, in);
    for (int index = 253; index >= 0; --index) {
        square_fe(c, c);
        if (index != 2 && index != 4) {
            mul_fe(c, c, in);
        }
    }
    set25519(out, c);
}

void pow2523(Fe out, const Fe in) {
    Fe c{};
    set25519(c, in);
    for (int index = 250; index >= 0; --index) {
        square_fe(c, c);
        if (index != 1) {
            mul_fe(c, c, in);
        }
    }
    set25519(out, c);
}

int unpackneg(Fe r[4], const unsigned char p[32]) {
    Fe t{};
    Fe chk{};
    Fe num{};
    Fe den{};
    Fe den2{};
    Fe den4{};
    Fe den6{};
    set25519(r[2], kFe1);
    unpack25519(r[1], p);
    square_fe(num, r[1]);
    mul_fe(den, num, kD);
    sub_fe(num, num, r[2]);
    add_fe(den, r[2], den);
    square_fe(den2, den);
    square_fe(den4, den2);
    mul_fe(den6, den4, den2);
    mul_fe(t, den6, num);
    mul_fe(t, t, den);
    pow2523(t, t);
    mul_fe(t, t, num);
    mul_fe(t, t, den);
    mul_fe(t, t, den);
    mul_fe(r[0], t, den);
    square_fe(chk, r[0]);
    mul_fe(chk, chk, den);
    if (neq25519(chk, num)) {
        mul_fe(r[0], r[0], kI);
    }
    square_fe(chk, r[0]);
    mul_fe(chk, chk, den);
    if (neq25519(chk, num)) {
        return -1;
    }
    if (par25519(r[0]) == (p[31] >> 7)) {
        sub_fe(r[0], kFe0, r[0]);
    }
    mul_fe(r[3], r[0], r[1]);
    return 0;
}

void point_add(Fe p[4], Fe q[4]) {
    Fe a{};
    Fe b{};
    Fe c{};
    Fe d{};
    Fe t{};
    Fe e{};
    Fe f{};
    Fe g{};
    Fe h{};
    sub_fe(a, p[1], p[0]);
    sub_fe(t, q[1], q[0]);
    mul_fe(a, a, t);
    add_fe(b, p[0], p[1]);
    add_fe(t, q[0], q[1]);
    mul_fe(b, b, t);
    mul_fe(c, p[3], q[3]);
    mul_fe(c, c, kD2);
    mul_fe(d, p[2], q[2]);
    add_fe(d, d, d);
    sub_fe(e, b, a);
    sub_fe(f, d, c);
    add_fe(g, d, c);
    add_fe(h, b, a);
    mul_fe(p[0], e, f);
    mul_fe(p[1], h, g);
    mul_fe(p[2], g, f);
    mul_fe(p[3], e, h);
}

void cswap(Fe p[4], Fe q[4], unsigned char bit) {
    for (int index = 0; index < 4; ++index) {
        sel25519(p[index], q[index], bit);
    }
}

void pack_point(unsigned char* out, Fe p[4]) {
    Fe tx{};
    Fe ty{};
    Fe zi{};
    inv25519(zi, p[2]);
    mul_fe(tx, p[0], zi);
    mul_fe(ty, p[1], zi);
    pack25519(out, ty);
    out[31] ^= static_cast<unsigned char>(par25519(tx) << 7);
}

void scalarmult(Fe p[4], Fe q[4], const unsigned char* scalar) {
    set25519(p[0], kFe0);
    set25519(p[1], kFe1);
    set25519(p[2], kFe1);
    set25519(p[3], kFe0);
    for (int index = 255; index >= 0; --index) {
        const unsigned char bit =
            static_cast<unsigned char>((scalar[index / 8] >> (index & 7)) & 1);
        cswap(p, q, bit);
        point_add(q, p);
        point_add(p, p);
        cswap(p, q, bit);
    }
}

void scalarbase(Fe p[4], const unsigned char* scalar) {
    Fe q[4];
    set25519(q[0], kX);
    set25519(q[1], kY);
    set25519(q[2], kFe1);
    mul_fe(q[3], kX, kY);
    scalarmult(p, q, scalar);
}

void mod_l(unsigned char* out, Limb x[64]) {
    for (int i = 63; i >= 32; --i) {
        Limb carry = 0;
        int j = i - 32;
        for (; j < i - 12; ++j) {
            x[j] += carry - 16 * x[i] * kOrder[j - (i - 32)];
            carry = (x[j] + 128) >> 8;
            x[j] -= carry << 8;
        }
        x[j] += carry;
        x[i] = 0;
    }
    Limb carry = 0;
    for (int j = 0; j < 32; ++j) {
        x[j] += carry - (x[31] >> 4) * kOrder[j];
        carry = x[j] >> 8;
        x[j] &= 255;
    }
    for (int j = 0; j < 32; ++j) {
        x[j] -= carry * kOrder[j];
    }
    for (int i = 0; i < 32; ++i) {
        x[i + 1] += x[i] >> 8;
        out[i] = static_cast<unsigned char>(x[i] & 255);
    }
}

void reduce_scalar(unsigned char* scalar) {
    Limb x[64];
    for (int index = 0; index < 64; ++index) {
        x[index] = static_cast<unsigned char>(scalar[index]);
    }
    for (int index = 0; index < 64; ++index) {
        scalar[index] = 0;
    }
    mod_l(scalar, x);
}

bool canonical_point(const unsigned char encoded[32]) {
    unsigned char y[32];
    std::memcpy(y, encoded, 32);
    y[31] = static_cast<unsigned char>(y[31] & 127);
    static constexpr unsigned char kPrime[32] = {0xed, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                                 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                                 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                                 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f};
    for (int index = 31; index >= 0; --index) {
        if (y[index] < kPrime[index]) {
            return true;
        }
        if (y[index] > kPrime[index]) {
            return false;
        }
    }
    return false;
}

bool canonical_scalar(const unsigned char scalar[32]) {
    for (int index = 31; index >= 0; --index) {
        if (scalar[index] < static_cast<unsigned char>(kOrder[index])) {
            return true;
        }
        if (scalar[index] > static_cast<unsigned char>(kOrder[index])) {
            return false;
        }
    }
    return false;
}

bool small_order(const unsigned char encoded[32]) {
    Fe point[4];
    if (unpackneg(point, encoded) != 0) {
        return true;
    }
    unsigned char eight[32] = {8};
    Fe image[4];
    scalarmult(image, point, eight);
    unsigned char packed[32];
    pack_point(packed, image);
    static constexpr unsigned char kIdentity[32] = {1};
    return std::memcmp(packed, kIdentity, 32) == 0;
}

bool signature_matches(const unsigned char* public_key, const unsigned char* signature,
                       const unsigned char* message, std::size_t message_size) {
    Fe neg_key[4];
    if (unpackneg(neg_key, public_key) != 0) {
        return false;
    }
    Sha512 hash;
    hash.update(signature, 32);
    hash.update(public_key, 32);
    if (message_size > 0) {
        hash.update(message, message_size);
    }
    unsigned char digest[64];
    hash.final(digest);
    reduce_scalar(digest);
    Fe challenge[4];
    scalarmult(challenge, neg_key, digest);
    Fe base[4];
    scalarbase(base, signature + 32);
    point_add(challenge, base);
    unsigned char packed[32];
    pack_point(packed, challenge);
    unsigned char diff = 0;
    for (int index = 0; index < 32; ++index) {
        diff |= static_cast<unsigned char>(packed[index] ^ signature[index]);
    }
    return diff == 0;
}

bool field_present(std::string_view value) {
    return !value.empty();
}

bool request_complete(const VerificationRequest& request) {
    return field_present(request.authorization_digest) &&
           field_present(request.signature_family_id) &&
           field_present(request.terminal_trust_state_id) &&
           field_present(request.terminal_effective_date) &&
           field_present(request.registry_entry_id) && field_present(request.signer_key_id) &&
           field_present(request.key_valid_from) && field_present(request.key_valid_until) &&
           field_present(request.role) && field_present(request.revocation_result) &&
           field_present(request.verification_date) && field_present(request.operation) &&
           field_present(request.canonical_authorization_path) && field_present(request.target_id);
}

} // namespace

bool VerificationDecision::accepted() const noexcept {
    return status == VerificationStatus::kAccepted && !authorization_digest.empty() &&
           !signature_family_id.empty() && !terminal_trust_state_id.empty() &&
           !terminal_effective_date.empty() && !registry_entry_id.empty() &&
           !signer_key_id.empty() && !key_valid_from.empty() && !key_valid_until.empty() &&
           !role.empty() && !revocation_result.empty() && !verification_date.empty() &&
           !operation.empty() && !canonical_authorization_path.empty() && !target_id.empty();
}

VerificationDecision verify_ed25519(const VerificationRequest& request) {
    VerificationDecision decision;
    if (request.signature.empty() || request.public_key.empty()) {
        decision.status = VerificationStatus::kUnavailable;
        return decision;
    }
    if (request.signature.size() != 64 || request.public_key.size() != 32) {
        decision.status = VerificationStatus::kMalformed;
        return decision;
    }
    const auto* signature = request.signature.data();
    const auto* public_key = request.public_key.data();
    if (!canonical_point(signature) || !canonical_point(public_key) ||
        !canonical_scalar(signature + 32) || small_order(signature) || small_order(public_key) ||
        !signature_matches(public_key, signature, request.message.data(), request.message.size())) {
        decision.status = VerificationStatus::kRejected;
        return decision;
    }
    if (!request_complete(request)) {
        decision.status = VerificationStatus::kMalformed;
        return decision;
    }
    decision.status = VerificationStatus::kAccepted;
    decision.authorization_digest = std::string(request.authorization_digest);
    decision.signature_family_id = std::string(request.signature_family_id);
    decision.terminal_trust_state_id = std::string(request.terminal_trust_state_id);
    decision.terminal_effective_date = std::string(request.terminal_effective_date);
    decision.registry_entry_id = std::string(request.registry_entry_id);
    decision.signer_key_id = std::string(request.signer_key_id);
    decision.key_valid_from = std::string(request.key_valid_from);
    decision.key_valid_until = std::string(request.key_valid_until);
    decision.role = std::string(request.role);
    decision.revocation_result = std::string(request.revocation_result);
    decision.verification_date = std::string(request.verification_date);
    decision.operation = std::string(request.operation);
    decision.canonical_authorization_path = std::string(request.canonical_authorization_path);
    decision.target_id = std::string(request.target_id);
    if (!decision.accepted()) {
        decision = VerificationDecision{};
        decision.status = VerificationStatus::kMalformed;
    }
    return decision;
}

} // namespace inferedge
