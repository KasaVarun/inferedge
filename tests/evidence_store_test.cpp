#include "inferedge/evidence_store.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <mutex>
#include <set>
#include <string>
#include <sys/stat.h>
#include <system_error>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <vector>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#include <mach-o/fixup-chains.h>
#include <mach-o/loader.h>
#include <mach/mach.h>
#include <sys/mman.h>
#endif

namespace {

// Test-only coordination for the payload revalidation open. It is linked only into
// inferedge_tests. Production binaries do not contain this gate.
struct PayloadOpenGate {
    std::mutex mutex;
    std::condition_variable ready;
    bool armed{false};
    bool paused{false};
    bool release{false};
    bool timed_out{false};
    int matches{0};
    int pause_at{2};
    bool has_owner{false};
    std::thread::id owner{};
    std::string leaf;
};

PayloadOpenGate& payload_open_gate() {
    static PayloadOpenGate gate;
    return gate;
}

void reset_payload_open_gate() {
    PayloadOpenGate& gate = payload_open_gate();
    std::lock_guard<std::mutex> lock(gate.mutex);
    gate.armed = false;
    gate.release = true;
    gate.has_owner = false;
    gate.ready.notify_all();
}

void pause_matching_open(const char* path) {
    if (path == nullptr) {
        return;
    }
    PayloadOpenGate& gate = payload_open_gate();
    std::unique_lock<std::mutex> lock(gate.mutex);
    if (!gate.armed || std::string_view(path) != gate.leaf) {
        return;
    }
    if (!gate.has_owner) {
        gate.has_owner = true;
        gate.owner = std::this_thread::get_id();
    } else if (gate.owner != std::this_thread::get_id()) {
        return;
    }
    ++gate.matches;
    if (gate.matches != gate.pause_at) {
        return;
    }
    gate.paused = true;
    gate.ready.notify_all();
    const bool released = gate.ready.wait_for(lock, std::chrono::seconds(5),
                                              [&gate] { return gate.release || !gate.armed; });
    if (!released) {
        gate.timed_out = true;
    }
}

class JoinWorker {
public:
    explicit JoinWorker(std::thread& thread) : thread_(thread) {}

    ~JoinWorker() {
        reset_payload_open_gate();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    JoinWorker(const JoinWorker&) = delete;
    JoinWorker& operator=(const JoinWorker&) = delete;

private:
    std::thread& thread_;
};

#if defined(__APPLE__)
#ifndef LC_DYLD_CHAINED_FIXUPS
#define LC_DYLD_CHAINED_FIXUPS 0x80000034
#endif
using OpenAtFunction = int (*)(int, const char*, int, ...);

OpenAtFunction real_openat = nullptr;
int openat_rebind_count = 0;
const char* openat_rebind_failure = "openat rebind has not run";

int coordinated_openat(int fd, const char* path, int flags, ...) {
    mode_t mode = 0;
    const bool created = (flags & O_CREAT) != 0;
    if (created) {
        va_list args;
        va_start(args, flags);
        mode = static_cast<mode_t>(va_arg(args, int));
        va_end(args);
    }
    pause_matching_open(path);
    if (real_openat == nullptr) {
        errno = ENOSYS;
        return -1;
    }
    if (created) {
        return real_openat(fd, path, flags, mode);
    }
    return real_openat(fd, path, flags);
}

void patch_openat_slot(uintptr_t runtime) {
    if (real_openat == nullptr) {
        return;
    }
    auto* slot = reinterpret_cast<void**>(runtime);
    if (*slot != reinterpret_cast<void*>(real_openat)) {
        openat_rebind_failure = "resolved openat slot does not match libSystem";
        return;
    }
    const uintptr_t page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
    const uintptr_t base = runtime & ~(page - 1U);
    if (mprotect(reinterpret_cast<void*>(base), static_cast<std::size_t>(page),
                 PROT_READ | PROT_WRITE) != 0) {
        const kern_return_t protected_page = vm_protect(
            mach_task_self(), static_cast<vm_address_t>(base), static_cast<vm_size_t>(page), FALSE,
            VM_PROT_READ | VM_PROT_WRITE | VM_PROT_COPY);
        if (protected_page != KERN_SUCCESS) {
            openat_rebind_failure = "openat slot page is not writable";
            return;
        }
    }
    *slot = reinterpret_cast<void*>(&coordinated_openat);
    mprotect(reinterpret_cast<void*>(base), static_cast<std::size_t>(page), PROT_READ);
    ++openat_rebind_count;
}

void rebind_openat_fixups() {
    real_openat = reinterpret_cast<OpenAtFunction>(dlsym(RTLD_NEXT, "openat"));
    if (real_openat == nullptr) {
        openat_rebind_failure = "libSystem openat is unavailable";
        return;
    }
    char executable[4096];
    uint32_t executable_size = sizeof(executable);
    if (_NSGetExecutablePath(executable, &executable_size) != 0) {
        openat_rebind_failure = "executable path is unavailable";
        return;
    }
    const int fd = open(executable, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        openat_rebind_failure = "executable file cannot be read";
        return;
    }
    const off_t file_size = lseek(fd, 0, SEEK_END);
    if (file_size < 32 || static_cast<std::uint64_t>(file_size) > 64U * 1024U * 1024U) {
        close(fd);
        openat_rebind_failure = "executable file size is unusable";
        return;
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(file_size));
    if (lseek(fd, 0, SEEK_SET) < 0 ||
        read(fd, bytes.data(), bytes.size()) != static_cast<ssize_t>(bytes.size())) {
        close(fd);
        openat_rebind_failure = "executable file read failed";
        return;
    }
    close(fd);

    uint32_t magic = 0;
    std::memcpy(&magic, bytes.data(), sizeof(magic));
    if (magic != MH_MAGIC_64) {
        openat_rebind_failure = "executable is not a 64-bit Mach-O";
        return;
    }
    uint32_t command_count = 0;
    std::memcpy(&command_count, bytes.data() + 16, sizeof(command_count));
    struct SegmentSpan {
        std::uint64_t vmaddr;
        std::uint64_t vmsize;
        std::uint64_t fileoff;
        std::uint64_t filesize;
    };
    SegmentSpan segments[16]{};
    int segment_count = 0;
    std::uint64_t preferred_base = 0;
    bool found_base = false;
    std::uint32_t fixups_offset = 0;
    std::uint32_t fixups_size = 0;
    std::size_t command = 32;
    for (uint32_t index = 0; index < command_count; ++index) {
        if (command + 8 > bytes.size()) {
            openat_rebind_failure = "Mach-O load commands are truncated";
            return;
        }
        uint32_t cmd = 0;
        uint32_t cmdsize = 0;
        std::memcpy(&cmd, bytes.data() + command, sizeof(cmd));
        std::memcpy(&cmdsize, bytes.data() + command + 4, sizeof(cmdsize));
        if (cmdsize < 8 || command + cmdsize > bytes.size()) {
            openat_rebind_failure = "Mach-O command size is invalid";
            return;
        }
        if (cmd == LC_SEGMENT_64 && cmdsize >= 72 && segment_count < 16) {
            SegmentSpan& span = segments[segment_count++];
            std::memcpy(&span.vmaddr, bytes.data() + command + 24, sizeof(span.vmaddr));
            std::memcpy(&span.vmsize, bytes.data() + command + 32, sizeof(span.vmsize));
            std::memcpy(&span.fileoff, bytes.data() + command + 40, sizeof(span.fileoff));
            std::memcpy(&span.filesize, bytes.data() + command + 48, sizeof(span.filesize));
            if (!found_base && span.fileoff == 0 && span.filesize > 0) {
                preferred_base = span.vmaddr;
                found_base = true;
            }
        } else if (cmd == LC_DYLD_CHAINED_FIXUPS && cmdsize >= 16) {
            std::memcpy(&fixups_offset, bytes.data() + command + 8, sizeof(fixups_offset));
            std::memcpy(&fixups_size, bytes.data() + command + 12, sizeof(fixups_size));
        }
        command += cmdsize;
    }
    if (!found_base || fixups_size < sizeof(dyld_chained_fixups_header) ||
        static_cast<std::uint64_t>(fixups_offset) + fixups_size > bytes.size()) {
        openat_rebind_failure = "chained fixups are absent";
        return;
    }
    dyld_chained_fixups_header fixups{};
    std::memcpy(&fixups, bytes.data() + fixups_offset, sizeof(fixups));
    if (fixups.imports_format != DYLD_CHAINED_IMPORT || fixups.symbols_format != 0) {
        openat_rebind_failure = "chained import format is unsupported";
        return;
    }
    const std::size_t imports_at = static_cast<std::size_t>(fixups_offset) + fixups.imports_offset;
    const std::size_t symbols_at = static_cast<std::size_t>(fixups_offset) + fixups.symbols_offset;
    if (imports_at + static_cast<std::uint64_t>(fixups.imports_count) * 4 > bytes.size() ||
        symbols_at >= bytes.size()) {
        openat_rebind_failure = "chained imports are truncated";
        return;
    }
    int ordinal = -1;
    for (uint32_t index = 0; index < fixups.imports_count; ++index) {
        dyld_chained_import imported{};
        std::memcpy(&imported, bytes.data() + imports_at + static_cast<std::size_t>(index) * 4,
                    sizeof(imported));
        const std::size_t name_at = symbols_at + imported.name_offset;
        if (name_at >= bytes.size()) {
            continue;
        }
        const char* name = reinterpret_cast<const char*>(bytes.data() + name_at);
        if (std::strcmp(name, "_openat") == 0) {
            ordinal = static_cast<int>(index);
            break;
        }
    }
    if (ordinal < 0) {
        openat_rebind_failure = "openat import is absent";
        return;
    }
    const std::size_t starts_at = static_cast<std::size_t>(fixups_offset) + fixups.starts_offset;
    if (starts_at + 4 > bytes.size()) {
        openat_rebind_failure = "chained starts are truncated";
        return;
    }
    uint32_t segment_starts = 0;
    std::memcpy(&segment_starts, bytes.data() + starts_at, sizeof(segment_starts));
    if (starts_at + 4 + static_cast<std::uint64_t>(segment_starts) * 4 > bytes.size()) {
        openat_rebind_failure = "chained segment table is truncated";
        return;
    }
    const mach_header_64* header =
        reinterpret_cast<const mach_header_64*>(_dyld_get_image_header(0));
    auto file_offset_for = [&](std::uint64_t vm, std::uint64_t& file_offset) {
        for (int index = 0; index < segment_count; ++index) {
            const SegmentSpan& span = segments[index];
            if (vm < span.vmaddr || vm >= span.vmaddr + span.vmsize) {
                continue;
            }
            const std::uint64_t delta = vm - span.vmaddr;
            if (delta + 8 > span.filesize) {
                return false;
            }
            file_offset = span.fileoff + delta;
            return file_offset + 8 <= bytes.size();
        }
        return false;
    };
    for (uint32_t segment_index = 0; segment_index < segment_starts; ++segment_index) {
        uint32_t relative = 0;
        std::memcpy(&relative,
                    bytes.data() + starts_at + 4 + static_cast<std::size_t>(segment_index) * 4,
                    sizeof(relative));
        if (relative == 0) {
            continue;
        }
        const std::size_t info = starts_at + relative;
        if (info + 22 > bytes.size()) {
            continue;
        }
        uint16_t page_size = 0;
        uint16_t pointer_format = 0;
        std::uint64_t segment_offset = 0;
        uint16_t page_count = 0;
        std::memcpy(&page_size, bytes.data() + info + 4, sizeof(page_size));
        std::memcpy(&pointer_format, bytes.data() + info + 6, sizeof(pointer_format));
        std::memcpy(&segment_offset, bytes.data() + info + 8, sizeof(segment_offset));
        std::memcpy(&page_count, bytes.data() + info + 20, sizeof(page_count));
        if (pointer_format != DYLD_CHAINED_PTR_64 && pointer_format != DYLD_CHAINED_PTR_64_OFFSET) {
            openat_rebind_failure = "chained pointer format is unsupported";
            return;
        }
        if (info + 22 + static_cast<std::uint64_t>(page_count) * 2 > bytes.size()) {
            continue;
        }
        auto walk = [&](uint32_t page, uint32_t offset) {
            for (int step = 0; step < 1000000; ++step) {
                const std::uint64_t vm = preferred_base + segment_offset +
                                         static_cast<std::uint64_t>(page) * page_size + offset;
                std::uint64_t file_offset = 0;
                if (!file_offset_for(vm, file_offset)) {
                    return;
                }
                std::uint64_t raw = 0;
                std::memcpy(&raw, bytes.data() + file_offset, sizeof(raw));
                dyld_chained_ptr_64_bind bound{};
                std::memcpy(&bound, &raw, sizeof(bound));
                if (bound.bind == 1 && static_cast<int>(bound.ordinal) == ordinal) {
                    patch_openat_slot(reinterpret_cast<uintptr_t>(header) +
                                      static_cast<uintptr_t>(vm - preferred_base));
                }
                if (bound.next == 0) {
                    return;
                }
                offset += static_cast<uint32_t>(bound.next) * 4U;
            }
        };
        for (uint16_t page = 0; page < page_count; ++page) {
            uint16_t start = 0;
            std::memcpy(&start, bytes.data() + info + 22 + static_cast<std::size_t>(page) * 2,
                        sizeof(start));
            if (start == DYLD_CHAINED_PTR_START_NONE) {
                continue;
            }
            if ((start & DYLD_CHAINED_PTR_START_MULTI) != 0) {
                uint32_t overflow = start & ~DYLD_CHAINED_PTR_START_MULTI;
                const std::size_t overflow_at =
                    info + 22 + static_cast<std::size_t>(page_count) * 2;
                for (int extra = 0; extra < 64; ++extra) {
                    if (overflow_at + (static_cast<std::uint64_t>(overflow) + 1U) * 2 >
                        bytes.size()) {
                        break;
                    }
                    uint16_t item = 0;
                    std::memcpy(&item,
                                bytes.data() + overflow_at + static_cast<std::size_t>(overflow) * 2,
                                sizeof(item));
                    walk(page, item & ~DYLD_CHAINED_PTR_START_LAST);
                    ++overflow;
                    if ((item & DYLD_CHAINED_PTR_START_LAST) != 0) {
                        break;
                    }
                }
                continue;
            }
            walk(page, start);
        }
    }
    if (openat_rebind_count == 0 && openat_rebind_failure != nullptr &&
        std::strcmp(openat_rebind_failure, "openat rebind has not run") == 0) {
        openat_rebind_failure = "no openat fixup was rebound";
    }
}

__attribute__((constructor)) void install_test_openat_rebind() {
    rebind_openat_fixups();
}
#endif

class ArmedPayloadOpen {
public:
    ArmedPayloadOpen(std::string leaf, int pause_at) {
        PayloadOpenGate& gate = payload_open_gate();
        std::lock_guard<std::mutex> lock(gate.mutex);
        gate.armed = true;
        gate.paused = false;
        gate.release = false;
        gate.timed_out = false;
        gate.matches = 0;
        gate.pause_at = pause_at;
        gate.has_owner = false;
        gate.leaf = std::move(leaf);
    }

    ~ArmedPayloadOpen() { reset_payload_open_gate(); }

    bool wait_until_paused() {
        PayloadOpenGate& gate = payload_open_gate();
        std::unique_lock<std::mutex> lock(gate.mutex);
        return gate.ready.wait_for(lock, std::chrono::seconds(5),
                                   [&gate] { return gate.paused || gate.timed_out; });
    }

    void release() {
        PayloadOpenGate& gate = payload_open_gate();
        std::lock_guard<std::mutex> lock(gate.mutex);
        gate.release = true;
        gate.ready.notify_all();
    }

    int matches() const {
        PayloadOpenGate& gate = payload_open_gate();
        std::lock_guard<std::mutex> lock(gate.mutex);
        return gate.matches;
    }

    bool timed_out() const {
        PayloadOpenGate& gate = payload_open_gate();
        std::lock_guard<std::mutex> lock(gate.mutex);
        return gate.timed_out;
    }

    bool paused() const {
        PayloadOpenGate& gate = payload_open_gate();
        std::lock_guard<std::mutex> lock(gate.mutex);
        return gate.paused;
    }
};

// Test-only signer. Production verification stays in src/ed25519_verify.cpp.
using Limb = long long;
using Fe = Limb[16];

constexpr Fe kFe0 = {0};
constexpr Fe kFe1 = {1};
constexpr Fe kD2 = {0xf159, 0x26b2, 0x9b94, 0xebd6, 0xb156, 0x8283, 0x149a, 0x00e0,
                    0xd130, 0xeef3, 0x80f2, 0x198e, 0xfce7, 0x56df, 0xd9dc, 0x2406};
constexpr Fe kX = {0xd51a, 0x8f25, 0x2d60, 0xc956, 0xa7b2, 0x9525, 0xc760, 0x692c,
                   0xdc5c, 0xfdd6, 0xe231, 0xc0a4, 0x53fe, 0xcd6e, 0x36d3, 0x2169};
constexpr Fe kY = {0x6658, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666,
                   0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666};
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

unsigned char par25519(const Fe value) {
    unsigned char packed[32];
    pack25519(packed, value);
    return packed[0] & 1;
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

void mul_scalar_add(unsigned char out[32], const unsigned char a[32], const unsigned char b[32],
                    const unsigned char c[32]) {
    Limb product[64] = {};
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 32; ++j) {
            product[i + j] += static_cast<Limb>(a[i]) * static_cast<Limb>(b[j]);
        }
    }
    for (int i = 0; i < 32; ++i) {
        product[i] += c[i];
    }
    mod_l(out, product);
}

struct TestKey {
    unsigned char prefix[32];
    unsigned char scalar[32];
    unsigned char public_key[32];
};

void test_keypair(TestKey& key, const unsigned char seed[32]) {
    Sha512 hash;
    hash.update(seed, 32);
    unsigned char digest[64];
    hash.final(digest);
    digest[0] = static_cast<unsigned char>(digest[0] & 248);
    digest[31] = static_cast<unsigned char>(digest[31] & 63);
    digest[31] = static_cast<unsigned char>(digest[31] | 64);
    std::memcpy(key.scalar, digest, 32);
    std::memcpy(key.prefix, digest + 32, 32);
    Fe point[4];
    scalarbase(point, key.scalar);
    pack_point(key.public_key, point);
}

void test_sign(unsigned char signature[64], const unsigned char* message, std::size_t size,
               const TestKey& key) {
    Sha512 nonce_hash;
    nonce_hash.update(key.prefix, 32);
    if (size > 0) {
        nonce_hash.update(message, size);
    }
    unsigned char nonce[64];
    nonce_hash.final(nonce);
    reduce_scalar(nonce);
    Fe point[4];
    scalarbase(point, nonce);
    pack_point(signature, point);
    Sha512 challenge;
    challenge.update(signature, 32);
    challenge.update(key.public_key, 32);
    if (size > 0) {
        challenge.update(message, size);
    }
    unsigned char hashed[64];
    challenge.final(hashed);
    reduce_scalar(hashed);
    mul_scalar_add(signature + 32, hashed, key.scalar, nonce);
}

using inferedge::acquire_evidence_lock;
using inferedge::BatchProbe;
using inferedge::cleanup_reservation;
using inferedge::CleanupObjectClass;
using inferedge::CleanupPendingRecord;
using inferedge::create_quarantine;
using inferedge::delete_quarantine;
using inferedge::DeletionJournal;
using inferedge::DeletionListEntry;
using inferedge::digest_hex;
using inferedge::EvidenceError;
using inferedge::EvidenceLock;
using inferedge::EvidenceLockRecord;
using inferedge::EvidenceObjectClass;
using inferedge::EvidenceReservation;
using inferedge::EvidenceTime;
using inferedge::kEvidenceCapBytes;
using inferedge::kQuarantineDirectoryCapBytes;
using inferedge::kZeroDigest;
using inferedge::parse_cleanup_pending;
using inferedge::parse_evidence_reservation;
using inferedge::parse_quarantine_metadata;
using inferedge::parse_stale_lock_authorization;
using inferedge::publish_batch;
using inferedge::QuarantineDeletionAuthorization;
using inferedge::QuarantineMetadata;
using inferedge::QuarantinePayload;
using inferedge::recover_stale_lock;
using inferedge::reservation_fits_cap;
using inferedge::ReservationDeletionAuthorization;
using inferedge::ReservationDeletionList;
using inferedge::ReservationEntry;
using inferedge::serialize_cleanup_pending;
using inferedge::serialize_evidence_lock;
using inferedge::serialize_evidence_reservation;
using inferedge::serialize_quarantine_deletion;
using inferedge::serialize_quarantine_metadata;
using inferedge::serialize_reservation_deletion;
using inferedge::serialize_reservation_deletion_list;
using inferedge::serialize_stale_lock_authorization;
using inferedge::StaleLockAuthorization;
using inferedge::sum_evidence_root;
using inferedge::validate_evidence_relative_path;

class ManualClock final : public EvidenceTime {
public:
    std::string date{"2026-01-15"};
    std::chrono::steady_clock::time_point current{std::chrono::steady_clock::time_point::min()};
    std::chrono::milliseconds last_sleep{0};
    bool jump{false};

    EvidenceError civil_date(std::string& out) const override {
        out = date;
        return EvidenceError::kOk;
    }
    std::chrono::steady_clock::time_point now() const override { return current; }
    void sleep_for(std::chrono::milliseconds duration) override {
        last_sleep = duration;
        current += jump ? std::chrono::seconds(601) : duration;
    }
};

// Host temporary directory from the standard library. Failure stays empty.
// There is no hardcoded temporary root and no silent fallback.
std::string make_temporary_directory(const char* prefix) {
    std::error_code error;
    const std::filesystem::path reported = std::filesystem::temp_directory_path(error);
    if (error || reported.empty() || !reported.is_absolute()) {
        return {};
    }
    // The store opens a caller root with O_NOFOLLOW from '/'. Darwin reports
    // this directory through a symlink. The canonical path is the same directory.
    const std::filesystem::path directory = std::filesystem::canonical(reported, error);
    if (error || directory.empty() || !directory.is_absolute()) {
        return {};
    }
    const std::string pattern = (directory / (std::string(prefix) + "XXXXXX")).string();
    if (pattern.size() < 6 || pattern.compare(pattern.size() - 6, 6, "XXXXXX") != 0) {
        return {};
    }
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    if (mkdtemp(buffer.data()) == nullptr) {
        return {};
    }
    return std::string(buffer.data());
}

class TempRoot {
public:
    explicit TempRoot(const char* prefix = "inferedge-9c6-") {
        path_ = make_temporary_directory(prefix);
    }
    ~TempRoot() {
        if (!path_.empty()) {
            remove_tree(-1, path_.c_str());
        }
    }
    const std::string& path() const { return path_; }
    bool ok() const { return !path_.empty(); }

private:
    static void remove_tree(int parent, const char* name) {
        const int fd = parent < 0
                           ? open(name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
                           : openat(parent, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (fd < 0) {
            if (parent >= 0) {
                unlinkat(parent, name, 0);
            }
            return;
        }
        DIR* directory = fdopendir(fd);
        if (directory == nullptr) {
            close(fd);
            return;
        }
        std::vector<std::string> names;
        while (const dirent* entry = readdir(directory)) {
            if (std::strcmp(entry->d_name, ".") != 0 && std::strcmp(entry->d_name, "..") != 0) {
                names.emplace_back(entry->d_name);
            }
        }
        for (const std::string& child : names) {
            struct stat status{};
            if (fstatat(dirfd(directory), child.c_str(), &status, AT_SYMLINK_NOFOLLOW) != 0) {
                continue;
            }
            if (S_ISDIR(status.st_mode) && !S_ISLNK(status.st_mode)) {
                remove_tree(dirfd(directory), child.c_str());
            } else {
                unlinkat(dirfd(directory), child.c_str(), 0);
            }
        }
        closedir(directory);
        if (parent >= 0) {
            unlinkat(parent, name, AT_REMOVEDIR);
        } else {
            rmdir(name);
        }
    }

    std::string path_;
};

bool path_exists(const std::string& path) {
    struct stat status{};
    return lstat(path.c_str(), &status) == 0;
}

std::string join_path(const std::string& root, std::string_view relative) {
    return root + "/" + std::string(relative);
}

void make_parents(const std::string& path) {
    for (std::size_t index = 1; index < path.size(); ++index) {
        if (path[index] != '/') {
            continue;
        }
        const std::string prefix = path.substr(0, index);
        mkdir(prefix.c_str(), 0700);
    }
}

bool trust_family_relative(std::string_view relative) {
    const std::size_t slash = relative.rfind('/');
    const std::string_view base =
        slash == std::string_view::npos ? relative : relative.substr(slash + 1);
    if (base != "transition.payload" && base != "transition.record" && base != "signature.bin" &&
        base != "signature.index" && base != "verification.record") {
        return false;
    }
    return relative.starts_with("genesis/") || relative.starts_with("transitions/");
}

void plant(const std::string& root, std::string_view relative, std::string_view bytes) {
    const std::string full = join_path(root, relative);
    make_parents(full);
    const int fd = open(full.c_str(), O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC | O_WRONLY, 0600);
    ASSERT_GE(fd, 0) << full;
    ASSERT_EQ(write(fd, bytes.data(), bytes.size()), static_cast<ssize_t>(bytes.size()));
    close(fd);
    if (trust_family_relative(relative)) {
        ASSERT_EQ(chmod(full.c_str(), 0444), 0);
    }
}

std::string slurp(const std::string& path) {
    const int fd = open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    EXPECT_GE(fd, 0);
    std::string bytes;
    char buffer[256];
    while (true) {
        const ssize_t count = read(fd, buffer, sizeof(buffer));
        if (count <= 0) {
            break;
        }
        bytes.append(buffer, static_cast<std::size_t>(count));
    }
    if (fd >= 0) {
        close(fd);
    }
    return bytes;
}

ReservationEntry entry_for(std::string staging, std::string final_path, std::string_view bytes,
                           EvidenceObjectClass object_class) {
    ReservationEntry entry;
    entry.staging_relative_path = std::move(staging);
    entry.final_relative_path = std::move(final_path);
    entry.reserved_object_size = bytes.size();
    entry.reserved_object_sha256 = digest_hex(bytes);
    entry.object_class = object_class;
    return entry;
}

EvidenceLock hold_lock(const std::string& root, ManualClock& clock,
                       std::string_view owner = "run-a") {
    EvidenceLock lock;
    EXPECT_EQ(acquire_evidence_lock(root, owner, "host-a", "2026-01-01", &clock, lock),
              EvidenceError::kOk);
    return lock;
}

} // namespace

#if !defined(__APPLE__)
extern "C" int __real_openat(int fd, const char* path, int flags, ...);
extern "C" int __wrap_openat(int fd, const char* path, int flags, ...) {
    mode_t mode = 0;
    const bool created = (flags & O_CREAT) != 0;
    if (created) {
        va_list args;
        va_start(args, flags);
        mode = static_cast<mode_t>(va_arg(args, int));
        va_end(args);
    }
    pause_matching_open(path);
    if (created) {
        return __real_openat(fd, path, flags, mode);
    }
    return __real_openat(fd, path, flags);
}
#endif

TEST(EvidenceStore, CallerRootAndLexicalPaths) {
    TempRoot first;
    TempRoot second;
    TempRoot decoy;
    ASSERT_TRUE(first.ok() && second.ok() && decoy.ok());
    ManualClock clock;
    char previous[4096];
    ASSERT_NE(getcwd(previous, sizeof(previous)), nullptr);
    ASSERT_EQ(chdir(decoy.path().c_str()), 0);
    plant(decoy.path(), "lock/evidence.lock", "decoy");
    EvidenceLock lock = hold_lock(first.path(), clock);
    EXPECT_TRUE(path_exists(join_path(first.path(), "lock/evidence.lock")));
    EXPECT_FALSE(path_exists(join_path(second.path(), "lock/evidence.lock")));
    EXPECT_EQ(slurp(join_path(decoy.path(), "lock/evidence.lock")), "decoy");
    EXPECT_EQ(chdir(previous), 0);
    EXPECT_EQ(validate_evidence_relative_path("/tmp/outside"), EvidenceError::kPath);
    EXPECT_EQ(validate_evidence_relative_path("partial//blob.bin"), EvidenceError::kPath);
    EXPECT_EQ(validate_evidence_relative_path("partial/./blob.bin"), EvidenceError::kPath);
    EXPECT_EQ(validate_evidence_relative_path("partial/../blob.bin"), EvidenceError::kPath);
    EXPECT_EQ(validate_evidence_relative_path("partial/run-a/blob\\bin"), EvidenceError::kPath);
    EXPECT_EQ(validate_evidence_relative_path(std::string("partial/\0blob", 13)),
              EvidenceError::kPath);
    std::uint64_t ignored = 7;
    EXPECT_EQ(sum_evidence_root("relative-root", ignored), EvidenceError::kPath);
    EXPECT_EQ(ignored, 7u);
    EXPECT_EQ(lock.release(), EvidenceError::kOk);
}

TEST(EvidenceStore, SymlinkHardLinkAndSpecialFiles) {
    std::uint64_t total = 0;
    {
        TempRoot root;
        ASSERT_TRUE(root.ok());
        plant(root.path(), "real/file.bin", "abcd");
        ASSERT_EQ(symlink("real", join_path(root.path(), "link").c_str()), 0);
        EXPECT_EQ(sum_evidence_root(root.path(), total), EvidenceError::kSymlink);
    }
    {
        TempRoot root;
        ASSERT_TRUE(root.ok());
        plant(root.path(), "real/file.bin", "abcd");
        ASSERT_EQ(symlink(join_path(root.path(), "real/file.bin").c_str(),
                          join_path(root.path(), "leaf.link").c_str()),
                  0);
        EXPECT_EQ(sum_evidence_root(root.path(), total), EvidenceError::kSymlink);
    }
    {
        TempRoot root;
        ASSERT_TRUE(root.ok());
        plant(root.path(), "linked.bin", "abcd");
        ASSERT_EQ(link(join_path(root.path(), "linked.bin").c_str(),
                       join_path(root.path(), "linked-too.bin").c_str()),
                  0);
        EXPECT_EQ(sum_evidence_root(root.path(), total), EvidenceError::kLink);
    }
    {
        TempRoot root;
        ASSERT_TRUE(root.ok());
        ASSERT_EQ(mkfifo(join_path(root.path(), "pipe.fifo").c_str(), 0600), 0);
        EXPECT_EQ(sum_evidence_root(root.path(), total), EvidenceError::kSpecial);
    }
}

TEST(EvidenceStore, RootEscapePreservesOutsideFile) {
    TempRoot root;
    TempRoot outside;
    ASSERT_TRUE(root.ok() && outside.ok());
    plant(outside.path(), "secret.bin", "secret");
    ManualClock clock;
    EvidenceLock lock = hold_lock(root.path(), clock);
    const ReservationEntry entry = entry_for("partial/run-a/blob.bin", "../secret.bin", "secret",
                                             EvidenceObjectClass::kPublishedBlob);
    EXPECT_NE(publish_batch(lock, "batch-a", {entry}, {"secret"}, nullptr), EvidenceError::kOk);
    EXPECT_EQ(slurp(join_path(outside.path(), "secret.bin")), "secret");
    EXPECT_EQ(lock.release(), EvidenceError::kOk);
}

TEST(EvidenceStore, AccountingIncludesEveryRegularFile) {
    TempRoot root;
    ASSERT_TRUE(root.ok());
    std::uint64_t total = 99;
    EXPECT_EQ(sum_evidence_root(root.path(), total), EvidenceError::kOk);
    EXPECT_EQ(total, 0u);
    plant(root.path(), ".hidden", "hide");
    plant(root.path(), "nested/deep/file.bin", "xyz");
    plant(root.path(), "lock/reservation-old.record", "rr");
    plant(root.path(), "partial/run-a/temp.bin", "t");
    plant(root.path(), "quarantine/qrun-a/payload.bin", "q");
    EXPECT_EQ(sum_evidence_root(root.path(), total), EvidenceError::kOk);
    EXPECT_EQ(total, std::string("hide").size() + 3 + 2 + 1 + 1);
    ManualClock clock;
    EvidenceLock lock = hold_lock(root.path(), clock);
    const std::string lock_bytes = slurp(join_path(root.path(), "lock/evidence.lock"));
    EXPECT_EQ(sum_evidence_root(root.path(), total), EvidenceError::kOk);
    EXPECT_EQ(total, std::string("hide").size() + 3 + 2 + 1 + 1 + lock_bytes.size());
    EXPECT_EQ(lock.release(), EvidenceError::kOk);
}

TEST(EvidenceStore, CapAndOverflow) {
    EXPECT_EQ(reservation_fits_cap(kEvidenceCapBytes, 0, 0), EvidenceError::kOk);
    EXPECT_EQ(reservation_fits_cap(kEvidenceCapBytes - 5, 2, 3), EvidenceError::kOk);
    EXPECT_EQ(reservation_fits_cap(kEvidenceCapBytes - 5, 2, 4), EvidenceError::kCap);
    EXPECT_EQ(reservation_fits_cap(UINT64_MAX, 1, 0), EvidenceError::kOverflow);
    EXPECT_EQ(std::string(inferedge::evidence_error_name(EvidenceError::kOverflow)), "overflow");
}

TEST(EvidenceStore, LockCreationTimeoutOwnerAndReplacement) {
    TempRoot root;
    ASSERT_TRUE(root.ok());
    ManualClock clock;
    EvidenceLock lock = hold_lock(root.path(), clock);
    EXPECT_EQ(clock.last_sleep.count(), 0);
    struct stat status{};
    ASSERT_EQ(stat(join_path(root.path(), "lock/evidence.lock").c_str(), &status), 0);
    EXPECT_TRUE(S_ISREG(status.st_mode));
    EXPECT_EQ(status.st_nlink, static_cast<nlink_t>(1));
    EXPECT_EQ(status.st_mode & 0777, static_cast<mode_t>(0600));
    const std::string original = slurp(join_path(root.path(), "lock/evidence.lock"));
    ManualClock impatient;
    impatient.jump = true;
    EvidenceLock second;
    EXPECT_EQ(
        acquire_evidence_lock(root.path(), "run-b", "host-b", "2026-01-02", &impatient, second),
        EvidenceError::kLock);
    EXPECT_LE(impatient.last_sleep, std::chrono::seconds(inferedge::kEvidenceLockWaitSeconds));
    EXPECT_EQ(slurp(join_path(root.path(), "lock/evidence.lock")), original);
    inferedge::EvidenceLockRecord rewritten;
    rewritten.owner_run_id = "run-b";
    rewritten.owner_host_id = "host-a";
    rewritten.acquired_date = "2026-01-01";
    std::string replacement;
    ASSERT_EQ(inferedge::serialize_evidence_lock(rewritten, replacement), EvidenceError::kOk);
    const int fd = open(join_path(root.path(), "lock/evidence.lock").c_str(),
                        O_WRONLY | O_NOFOLLOW | O_CLOEXEC);
    ASSERT_GE(fd, 0);
    ASSERT_EQ(ftruncate(fd, 0), 0);
    ASSERT_EQ(write(fd, replacement.data(), replacement.size()),
              static_cast<ssize_t>(replacement.size()));
    close(fd);
    EXPECT_EQ(lock.release(), EvidenceError::kOwner);
    EXPECT_EQ(slurp(join_path(root.path(), "lock/evidence.lock")), replacement);
    plant(root.path(), "lock/other.lock", "new-inode");
    unlink(join_path(root.path(), "lock/evidence.lock").c_str());
    ASSERT_EQ(rename(join_path(root.path(), "lock/other.lock").c_str(),
                     join_path(root.path(), "lock/evidence.lock").c_str()),
              0);
    EXPECT_EQ(lock.release(), EvidenceError::kRace);
    EXPECT_EQ(slurp(join_path(root.path(), "lock/evidence.lock")), "new-inode");
    EvidenceLock idle;
    EXPECT_EQ(publish_batch(idle, "batch-a", {}, {}, nullptr), EvidenceError::kLock);
}

TEST(EvidenceStore, ReservationGrammarSizeAndCap) {
    const std::string alpha = "alpha";
    const std::string beta = "beta";
    std::vector<ReservationEntry> entries{
        entry_for("partial/run-a/b.bin", "runs/item-a/b.bin", beta,
                  EvidenceObjectClass::kRetainedMetadata),
        entry_for("partial/run-a/a.bin", "provenance/item-a/a.bin", alpha,
                  EvidenceObjectClass::kPublishedBlob),
    };
    EvidenceReservation reservation;
    ASSERT_EQ(serialize_evidence_reservation("run-a", entries, reservation), EvidenceError::kOk);
    EXPECT_EQ(reservation.entries.front().staging_relative_path, "partial/run-a/a.bin");
    EXPECT_EQ(reservation.reserved_bytes, reservation.bytes.size() + alpha.size() + beta.size());
    EvidenceReservation parsed;
    ASSERT_EQ(parse_evidence_reservation(reservation.bytes, parsed), EvidenceError::kOk);
    EXPECT_EQ(parsed.bytes, reservation.bytes);
    entries[0].staging_relative_path = entries[1].staging_relative_path;
    EXPECT_EQ(serialize_evidence_reservation("run-a", entries, reservation),
              EvidenceError::kDuplicate);
    entries[0].staging_relative_path = "partial/run-a/b.bin";
    entries[0].final_relative_path = entries[1].final_relative_path;
    EXPECT_EQ(serialize_evidence_reservation("run-a", entries, reservation),
              EvidenceError::kDuplicate);
    entries[0].final_relative_path = "runs/item-a/b.bin";
    entries[0].staging_relative_path = "partial/other/b.bin";
    EXPECT_EQ(serialize_evidence_reservation("run-a", entries, reservation), EvidenceError::kPath);
    entries[0].staging_relative_path = "partial/run-a/.hidden";
    EXPECT_NE(serialize_evidence_reservation("run-a", entries, reservation), EvidenceError::kOk);
    entries[0].staging_relative_path = "partial/run-a/b.bin";
    entries[0].final_relative_path = "none";
    EXPECT_NE(serialize_evidence_reservation("run-a", entries, reservation), EvidenceError::kOk);
    entries[0].object_class = EvidenceObjectClass::kTemporaryMetadata;
    entries[0].final_relative_path = "runs/item-a/b.bin";
    EXPECT_EQ(serialize_evidence_reservation("run-a", entries, reservation), EvidenceError::kPath);
    entries[0].final_relative_path = "none";
    entries[0].staging_relative_path = entries[1].final_relative_path;
    EXPECT_NE(serialize_evidence_reservation("run-a", entries, reservation), EvidenceError::kOk);
    ReservationEntry huge = entry_for("partial/run-a/huge.bin", "provenance/item-a/huge.bin", "x",
                                      EvidenceObjectClass::kPublishedBlob);
    huge.reserved_object_size = UINT64_MAX;
    ReservationEntry one = entry_for("partial/run-a/one.bin", "provenance/item-a/one.bin", "y",
                                     EvidenceObjectClass::kPublishedBlob);
    EXPECT_EQ(serialize_evidence_reservation("run-a", {huge, one}, reservation),
              EvidenceError::kOverflow);
    EXPECT_EQ(reservation_fits_cap(kEvidenceCapBytes - 10, reservation.bytes.size(), 11),
              EvidenceError::kCap);
}

TEST(EvidenceStore, PublicationAndConfinedCleanup) {
    TempRoot root;
    ASSERT_TRUE(root.ok());
    ManualClock clock;
    EvidenceLock lock = hold_lock(root.path(), clock);
    plant(root.path(), "provenance/item-a/keep.bin", "keep");
    plant(root.path(), "partial/other/secret.bin", "secret");
    plant(root.path(), "unexpected.bin", "odd");
    const std::string alpha = "alpha";
    const std::string meta = "meta";
    const std::string temp = "temp";
    const std::vector<ReservationEntry> entries{
        entry_for("partial/run-a/a.bin", "provenance/item-a/a.bin", alpha,
                  EvidenceObjectClass::kPublishedBlob),
        entry_for("partial/run-a/m.bin", "runs/item-a/m.bin", meta,
                  EvidenceObjectClass::kRetainedMetadata),
        entry_for("partial/run-a/t.bin", "none", temp, EvidenceObjectClass::kTemporaryMetadata),
    };
    ASSERT_EQ(publish_batch(lock, "batch-a", entries, {alpha, meta, temp}, nullptr),
              EvidenceError::kOk);
    EXPECT_EQ(slurp(join_path(root.path(), "provenance/item-a/a.bin")), alpha);
    EXPECT_EQ(slurp(join_path(root.path(), "runs/item-a/m.bin")), meta);
    EXPECT_FALSE(path_exists(join_path(root.path(), "partial/run-a/a.bin")));
    EXPECT_FALSE(path_exists(join_path(root.path(), "partial/run-a/m.bin")));
    EXPECT_FALSE(path_exists(join_path(root.path(), "partial/run-a/t.bin")));
    EXPECT_FALSE(path_exists(join_path(root.path(), "lock/reservation-batch-a.record")));
    EXPECT_EQ(slurp(join_path(root.path(), "provenance/item-a/keep.bin")), "keep");
    const ReservationEntry bad = entry_for("partial/run-a/bad.bin", "provenance/item-a/bad.bin",
                                           "good", EvidenceObjectClass::kPublishedBlob);
    EXPECT_EQ(publish_batch(lock, "batch-b", {bad}, {"nope"}, nullptr), EvidenceError::kDigest);
    EXPECT_FALSE(path_exists(join_path(root.path(), "partial/run-a/bad.bin")));
    EXPECT_FALSE(path_exists(join_path(root.path(), "provenance/item-a/bad.bin")));
    EXPECT_FALSE(path_exists(join_path(root.path(), "lock/reservation-batch-b.record")));
    EXPECT_EQ(slurp(join_path(root.path(), "partial/other/secret.bin")), "secret");
    EXPECT_EQ(slurp(join_path(root.path(), "unexpected.bin")), "odd");
    const ReservationEntry sized = entry_for("partial/run-a/sized.bin", "golden/item-a/sized.bin",
                                             "abcd", EvidenceObjectClass::kPublishedBlob);
    EXPECT_EQ(publish_batch(lock, "batch-c", {sized}, {"abc"}, nullptr), EvidenceError::kSize);
    EXPECT_FALSE(path_exists(join_path(root.path(), "golden/item-a/sized.bin")));
    plant(root.path(), "partial/run-a/already.bin", "old");
    const ReservationEntry exists =
        entry_for("partial/run-a/already.bin", "classb/item-a/already.bin", "new",
                  EvidenceObjectClass::kPublishedBlob);
    EXPECT_EQ(publish_batch(lock, "batch-d", {exists}, {"new"}, nullptr), EvidenceError::kPresent);
    EXPECT_EQ(slurp(join_path(root.path(), "partial/run-a/already.bin")), "old");
    EXPECT_EQ(lock.release(), EvidenceError::kOk);
}

TEST(EvidenceStore, ReplacementTargetIsNotUnlinked) {
    TempRoot root;
    ASSERT_TRUE(root.ok());
    ManualClock clock;
    EvidenceLock lock = hold_lock(root.path(), clock);
    class Replace final : public BatchProbe {
    public:
        explicit Replace(std::string root) : root_(std::move(root)) {}
        EvidenceError after_staging_written(std::string_view staging_relative_path) override {
            const std::string full = join_path(root_, staging_relative_path);
            unlink(full.c_str());
            const int fd =
                open(full.c_str(), O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC | O_WRONLY, 0600);
            EXPECT_GE(fd, 0);
            const char bytes[] = "replaced";
            EXPECT_EQ(write(fd, bytes, sizeof(bytes) - 1), static_cast<ssize_t>(sizeof(bytes) - 1));
            close(fd);
            return EvidenceError::kIo;
        }

    private:
        std::string root_;
    } probe(root.path());
    const ReservationEntry entry = entry_for("partial/run-a/race.bin", "provenance/item-a/race.bin",
                                             "original", EvidenceObjectClass::kPublishedBlob);
    EXPECT_EQ(publish_batch(lock, "batch-e", {entry}, {"original"}, &probe), EvidenceError::kRace);
    EXPECT_EQ(slurp(join_path(root.path(), "partial/run-a/race.bin")), "replaced");
    EXPECT_FALSE(path_exists(join_path(root.path(), "provenance/item-a/race.bin")));
    EXPECT_EQ(lock.release(), EvidenceError::kOk);
}

TEST(EvidenceStore, PublicDestructiveApiFailsClosed) {
    TempRoot evidence;
    TempRoot admission;
    ASSERT_TRUE(evidence.ok() && admission.ok());
    ManualClock clock;
    EvidenceLock lock = hold_lock(evidence.path(), clock);
    plant(evidence.path(), "provenance/item-a/keep.bin", "keep");
    bool acquired = true;
    EvidenceLock recovered;
    EXPECT_EQ(recover_stale_lock(evidence.path(), admission.path(), "2026-01-03", "trust-a",
                                 "key-a", "bytes", nullptr, "run-b", "host-b", &clock, recovered,
                                 acquired),
              EvidenceError::kUnavailable);
    EXPECT_TRUE(acquired);
    EXPECT_FALSE(recovered.held());
    EXPECT_EQ(slurp(join_path(evidence.path(), "provenance/item-a/keep.bin")), "keep");
    EXPECT_FALSE(path_exists(join_path(evidence.path(), "lock/recovery.lock")));
    EXPECT_EQ(cleanup_reservation(lock, admission.path(), "2026-02-01", "trust-a", "key-a", "bytes",
                                  nullptr, "list", &clock),
              EvidenceError::kUnavailable);
    EXPECT_EQ(delete_quarantine(lock, admission.path(), "2026-03-31", "trust-a", "key-a", "bytes",
                                nullptr, &clock),
              EvidenceError::kUnavailable);
    inferedge::AuthenticatedCapability blank;
    EXPECT_FALSE(blank.holds_authority());
    EXPECT_EQ(delete_quarantine(lock, admission.path(), "2026-03-31", "trust-a", "key-a", "bytes",
                                &blank, &clock),
              EvidenceError::kUnavailable);
    EXPECT_EQ(slurp(join_path(evidence.path(), "provenance/item-a/keep.bin")), "keep");
    EXPECT_EQ(lock.release(), EvidenceError::kOk);
}

TEST(EvidenceStore, QuarantinePayloadGrammar) {
    TempRoot root;
    ASSERT_TRUE(root.ok());
    ManualClock clock;
    clock.date = "2026-03-01";
    EvidenceLock lock = hold_lock(root.path(), clock);
    EXPECT_EQ(create_quarantine(lock, "qrun-a", "9-c6", "checksum_rejected", "none",
                                {QuarantinePayload{"payload.bin", "data"}}, &clock),
              EvidenceError::kForbidden);
    EXPECT_FALSE(path_exists(join_path(root.path(), "quarantine")));
    const char* reasons[] = {"checksum_rejected",  "reproduction_mismatch", "recipe_echo_rejected",
                             "cap_rejected",       "symlink_rejected",      "signature_failed",
                             "invariant_mismatch", "other_rejected"};
    int index = 0;
    for (const char* reason : reasons) {
        const std::string run = "qrun-" + std::to_string(index);
        EXPECT_EQ(create_quarantine(lock, run, "9-c6", reason, "none", {}, &clock),
                  EvidenceError::kOk);
        EXPECT_TRUE(
            path_exists(join_path(root.path(), "quarantine/" + run + "/quarantine.record")));
        if (std::string(reason) != "recipe_echo_rejected") {
            EXPECT_EQ(create_quarantine(lock, run + "x", "9-c6", reason, "none",
                                        {QuarantinePayload{"recipe-echo.bin", "echo"}}, &clock),
                      EvidenceError::kForbidden);
        }
        ++index;
    }
    EXPECT_EQ(create_quarantine(lock, "qrun-a", "9-c6", "recipe_echo_rejected", "none",
                                {QuarantinePayload{"recipe-echo.bin", "echo"}}, &clock),
              EvidenceError::kOk);
    const ReservationEntry promoted =
        entry_for("partial/run-a/recipe-echo.bin", "provenance/item-a/recipe-echo.bin", "echo",
                  EvidenceObjectClass::kPublishedBlob);
    EXPECT_EQ(publish_batch(lock, "batch-q", {promoted}, {"echo"}, nullptr),
              EvidenceError::kForbidden);
    EXPECT_EQ(lock.release(), EvidenceError::kOk);
}

TEST(EvidenceStore, QuarantineDirectoryCap) {
    TempRoot root;
    ASSERT_TRUE(root.ok());
    ManualClock clock;
    clock.date = "2026-04-01";
    QuarantineMetadata fields;
    fields.quarantine_id = "qrun-c";
    fields.associated_phase_id = "9-c6";
    fields.associated_run_id = "qrun-c";
    fields.directory_id = "quarantine/qrun-c";
    fields.creation_date = clock.date;
    fields.reason_code = "cap_rejected";
    fields.accepted_run_manifest_path = "none";
    fields.accepted_run_manifest_file_sha256 = std::string(kZeroDigest);
    fields.retention_start = clock.date;
    fields.retention_deadline = "2026-05-01";
    std::string record;
    ASSERT_EQ(serialize_quarantine_metadata(fields, record), EvidenceError::kOk);
    ASSERT_EQ(mkdir(join_path(root.path(), "quarantine").c_str(), 0700), 0);
    ASSERT_EQ(mkdir(join_path(root.path(), "quarantine/qrun-c").c_str(), 0700), 0);
    const std::string sparse = join_path(root.path(), "quarantine/qrun-c/big.bin");
    const int fd = open(sparse.c_str(), O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC | O_WRONLY, 0600);
    ASSERT_GE(fd, 0);
    ASSERT_EQ(ftruncate(fd, static_cast<off_t>(kQuarantineDirectoryCapBytes - record.size())), 0);
    close(fd);
    EvidenceLock lock = hold_lock(root.path(), clock);
    EXPECT_EQ(create_quarantine(lock, "qrun-c", "9-c6", "cap_rejected", "none", {}, &clock),
              EvidenceError::kOk);
    EXPECT_TRUE(path_exists(join_path(root.path(), "quarantine/qrun-c/quarantine.record")));
    unlink(join_path(root.path(), "quarantine/qrun-c/quarantine.record").c_str());
    const int grown = open(sparse.c_str(), O_WRONLY | O_NOFOLLOW | O_CLOEXEC);
    ASSERT_GE(grown, 0);
    ASSERT_EQ(
        ftruncate(grown, static_cast<off_t>(kQuarantineDirectoryCapBytes - record.size() + 1)), 0);
    close(grown);
    EXPECT_EQ(create_quarantine(lock, "qrun-c", "9-c6", "cap_rejected", "none", {}, &clock),
              EvidenceError::kCap);
    EXPECT_FALSE(path_exists(join_path(root.path(), "quarantine/qrun-c/quarantine.record")));
    EXPECT_EQ(lock.release(), EvidenceError::kOk);
}

TEST(EvidenceStore, ParseFailureLeavesCallerOutput) {
    StaleLockAuthorization stale;
    stale.authorization_id = "KEEP";
    EXPECT_NE(parse_stale_lock_authorization("not canonical\n", stale), EvidenceError::kOk);
    EXPECT_EQ(stale.authorization_id, "KEEP");
    EvidenceReservation reservation;
    reservation.owner_run_id = "KEEP";
    EXPECT_NE(parse_evidence_reservation("schema_version=1\n", reservation), EvidenceError::kOk);
    EXPECT_EQ(reservation.owner_run_id, "KEEP");
    CleanupPendingRecord pending;
    pending.authorization_id = "KEEP";
    EXPECT_NE(parse_cleanup_pending("nope\n", pending), EvidenceError::kOk);
    EXPECT_EQ(pending.authorization_id, "KEEP");
    std::uint64_t total = 11;
    EXPECT_EQ(sum_evidence_root("/no/such/inferedge-9c6-root", total), EvidenceError::kAbsent);
    EXPECT_EQ(total, 11u);
    EXPECT_EQ(std::string(inferedge::evidence_error_name(EvidenceError::kUnavailable)),
              "unavailable");
}

TEST(EvidenceStore, LockWaitBudgetIsIndependentOfClockProgress) {
    class BudgetClock final : public EvidenceTime {
    public:
        std::uint64_t requested{0};
        int sleeps{0};
        std::chrono::milliseconds maximum{0};
        bool freeze{false};
        bool backward{false};
        std::string release_path;
        std::uint64_t release_at{UINT64_MAX};

        EvidenceError civil_date(std::string& out) const override {
            out = "2026-01-15";
            return EvidenceError::kOk;
        }
        std::chrono::steady_clock::time_point now() const override {
            if (backward && sleeps > 0) {
                return std::chrono::steady_clock::time_point::min();
            }
            return std::chrono::steady_clock::time_point{} + std::chrono::seconds(5);
        }
        void sleep_for(std::chrono::milliseconds duration) override {
            ++sleeps;
            if (duration > maximum) {
                maximum = duration;
            }
            requested += static_cast<std::uint64_t>(duration.count());
            if (!release_path.empty() && requested >= release_at) {
                unlink(release_path.c_str());
            }
            if (!freeze) {
                static_cast<void>(duration);
            }
        }
    };

    TempRoot root;
    ASSERT_TRUE(root.ok());
    BudgetClock immediate;
    EvidenceLock first;
    EXPECT_EQ(
        acquire_evidence_lock(root.path(), "run-a", "host-a", "2026-01-01", &immediate, first),
        EvidenceError::kOk);
    EXPECT_EQ(immediate.requested, 0u);
    EXPECT_EQ(immediate.sleeps, 0);

    BudgetClock frozen;
    frozen.freeze = true;
    EvidenceLock second;
    EXPECT_EQ(acquire_evidence_lock(root.path(), "run-b", "host-b", "2026-01-02", &frozen, second),
              EvidenceError::kLock);
    EXPECT_EQ(frozen.requested, inferedge::kEvidenceLockWaitSeconds * 1000ull);
    EXPECT_EQ(frozen.maximum, std::chrono::milliseconds(50));
    EXPECT_FALSE(second.held());

    BudgetClock early;
    early.release_path = join_path(root.path(), "lock/evidence.lock");
    early.release_at = inferedge::kEvidenceLockWaitSeconds * 1000ull - 50ull;
    EvidenceLock third;
    EXPECT_EQ(acquire_evidence_lock(root.path(), "run-c", "host-c", "2026-01-03", &early, third),
              EvidenceError::kOk);
    EXPECT_EQ(early.requested, inferedge::kEvidenceLockWaitSeconds * 1000ull - 50ull);
    EXPECT_EQ(third.release(), EvidenceError::kOk);
    EvidenceLock blocker;
    EXPECT_EQ(
        acquire_evidence_lock(root.path(), "run-e", "host-a", "2026-01-01", &immediate, blocker),
        EvidenceError::kOk);

    BudgetClock boundary;
    boundary.release_path = join_path(root.path(), "lock/evidence.lock");
    boundary.release_at = inferedge::kEvidenceLockWaitSeconds * 1000ull;
    EvidenceLock fourth;
    EXPECT_EQ(
        acquire_evidence_lock(root.path(), "run-d", "host-d", "2026-01-04", &boundary, fourth),
        EvidenceError::kOk);
    EXPECT_EQ(boundary.requested, inferedge::kEvidenceLockWaitSeconds * 1000ull);
    EXPECT_EQ(fourth.release(), EvidenceError::kOk);
    EXPECT_EQ(blocker.release(), EvidenceError::kAbsent);

    EvidenceLock held;
    EXPECT_EQ(acquire_evidence_lock(root.path(), "run-g", "host-a", "2026-01-01", &immediate, held),
              EvidenceError::kOk);
    BudgetClock backward;
    backward.backward = true;
    EvidenceLock fifth;
    EXPECT_EQ(acquire_evidence_lock(root.path(), "run-f", "host-f", "2026-01-05", &backward, fifth),
              EvidenceError::kInternal);
    EXPECT_EQ(backward.requested, 50u);
    EXPECT_EQ(held.release(), EvidenceError::kOk);
}

TEST(EvidenceStore, SymlinkAncestorsOfCallerRootsAreRejected) {
    TempRoot real("inferedge-real-");
    TempRoot links("inferedge-link-");
    ASSERT_TRUE(real.ok());
    ASSERT_TRUE(links.ok());
    ASSERT_EQ(mkdir(join_path(real.path(), "chain").c_str(), 0700), 0);
    ASSERT_EQ(mkdir(join_path(real.path(), "chain/mid").c_str(), 0700), 0);
    ASSERT_EQ(mkdir(join_path(real.path(), "chain/mid/root").c_str(), 0700), 0);
    ASSERT_EQ(mkdir(join_path(real.path(), "chain-evil").c_str(), 0700), 0);
    plant(real.path(), "chain-evil/secret.bin", "secret");
    plant(real.path(), "chain/mid/root/keep.bin", "keep");
    std::uint64_t total = 0;
    EXPECT_EQ(sum_evidence_root(join_path(real.path(), "chain/mid/root"), total),
              EvidenceError::kOk);
    EXPECT_EQ(total, 4u);
    EXPECT_EQ(slurp(join_path(real.path(), "chain-evil/secret.bin")), "secret");

    ASSERT_EQ(symlink(join_path(real.path(), "chain/mid/root").c_str(),
                      join_path(links.path(), "final").c_str()),
              0);
    EXPECT_EQ(sum_evidence_root(join_path(links.path(), "final"), total), EvidenceError::kSymlink);
    ASSERT_EQ(symlink(join_path(real.path(), "chain/mid").c_str(),
                      join_path(links.path(), "parent").c_str()),
              0);
    EXPECT_EQ(sum_evidence_root(join_path(links.path(), "parent/root"), total),
              EvidenceError::kSymlink);
    ASSERT_EQ(
        symlink(join_path(real.path(), "chain").c_str(), join_path(links.path(), "middle").c_str()),
        0);
    EXPECT_EQ(sum_evidence_root(join_path(links.path(), "middle/mid/root"), total),
              EvidenceError::kSymlink);
    ASSERT_EQ(symlink(real.path().c_str(), join_path(links.path(), "first").c_str()), 0);
    EXPECT_EQ(sum_evidence_root(join_path(links.path(), "first/chain/mid/root"), total),
              EvidenceError::kSymlink);
    EXPECT_EQ(slurp(join_path(real.path(), "chain/mid/root/keep.bin")), "keep");
    EXPECT_EQ(slurp(join_path(real.path(), "chain-evil/secret.bin")), "secret");
}

TEST(EvidenceStore, FrozenDigestAndParserRoundTrip) {
    const std::string frozen =
        "schema_version=1\n"
        "record_kind=evidence_lock\n"
        "owner_run_id=run-a\n"
        "owner_host_id=host-a\n"
        "acquired_date=2026-01-15\n"
        "record_sha256=fa54c94b400238157212156615ddf913f3fb9551ae547653679dd225d986f44c\n";
    inferedge::EvidenceLockRecord parsed;
    ASSERT_EQ(inferedge::parse_evidence_lock(frozen, parsed), EvidenceError::kOk);
    EXPECT_EQ(parsed.bytes, frozen);
    EXPECT_EQ(parsed.owner_run_id, "run-a");
    std::string serialized;
    ASSERT_EQ(inferedge::serialize_evidence_lock(parsed, serialized), EvidenceError::kOk);
    EXPECT_EQ(serialized, frozen);
    EXPECT_NE(inferedge::parse_evidence_lock(frozen + "extra=1\n", parsed), EvidenceError::kOk);
    std::string missing = frozen;
    missing.replace(missing.find("owner_host_id=host-a\n"),
                    std::string("owner_host_id=host-a\n").size(), "");
    EXPECT_NE(inferedge::parse_evidence_lock(missing, parsed), EvidenceError::kOk);
    std::string duplicate = frozen;
    duplicate.insert(duplicate.find("record_sha256"), "owner_run_id=run-a\n");
    EXPECT_NE(inferedge::parse_evidence_lock(duplicate, parsed), EvidenceError::kOk);
    std::string misordered = frozen;
    const std::string host = "owner_host_id=host-a\n";
    const std::string owner = "owner_run_id=run-a\n";
    misordered.replace(misordered.find(owner), owner.size(), "");
    misordered.insert(misordered.find(host) + host.size(), owner);
    EXPECT_NE(inferedge::parse_evidence_lock(misordered, parsed), EvidenceError::kOk);

    for (std::uint64_t size : {9ull, 10ull, 99ull, 100ull, 999ull, 1000ull}) {
        EvidenceReservation reservation;
        ReservationEntry item = entry_for("partial/run-a/a.bin", "provenance/item-a/a.bin",
                                          std::string(static_cast<std::size_t>(size), 'a'),
                                          EvidenceObjectClass::kPublishedBlob);
        item.reserved_object_size = size;
        ASSERT_EQ(serialize_evidence_reservation("run-a", {item}, reservation), EvidenceError::kOk);
        EXPECT_EQ(reservation.reserved_bytes, reservation.bytes.size() + size);
        EvidenceReservation round;
        ASSERT_EQ(parse_evidence_reservation(reservation.bytes, round), EvidenceError::kOk);
        EXPECT_EQ(round.bytes, reservation.bytes);
    }
}

TEST(EvidenceStore, EveryRecordRoundTripRejectsMalformedKeys) {
    auto reject_malformed = [](const std::string& canonical, auto parse) {
        EXPECT_NE(parse(canonical + "extra=1\n"), EvidenceError::kOk);
        std::string duplicate = canonical;
        const std::string first = canonical.substr(0, canonical.find('\n') + 1);
        duplicate.insert(first.size(), first);
        EXPECT_NE(parse(duplicate), EvidenceError::kOk);
        std::string missing = canonical;
        const auto second = missing.find('\n');
        missing.erase(0, second + 1);
        EXPECT_NE(parse(missing), EvidenceError::kOk);
        std::string misordered = canonical;
        const auto line_end = misordered.find('\n');
        const std::string line = misordered.substr(0, line_end + 1);
        misordered.erase(0, line_end + 1);
        const auto next = misordered.find('\n');
        misordered.insert(next + 1, line);
        EXPECT_NE(parse(misordered), EvidenceError::kOk);
    };

    inferedge::EvidenceLockRecord lock_fields;
    lock_fields.owner_run_id = "run-a";
    lock_fields.owner_host_id = "host-a";
    lock_fields.acquired_date = "2026-01-15";
    std::string lock_bytes;
    ASSERT_EQ(inferedge::serialize_evidence_lock(lock_fields, lock_bytes), EvidenceError::kOk);
    inferedge::EvidenceLockRecord lock_round;
    ASSERT_EQ(inferedge::parse_evidence_lock(lock_bytes, lock_round), EvidenceError::kOk);
    EXPECT_EQ(lock_round.bytes, lock_bytes);
    reject_malformed(lock_bytes, [](const std::string& bytes) {
        inferedge::EvidenceLockRecord out;
        return inferedge::parse_evidence_lock(bytes, out);
    });

    const std::string recovery =
        "schema_version=1\n"
        "record_kind=recovery_lock\n"
        "owner_run_id=run-a\n"
        "acquired_date=2026-01-01\n"
        "record_sha256=cd83b4ca5cf953db04967f4440f8cbe40f441068b15a6817c866fa7cb5590d93\n";
    inferedge::RecoveryLockRecord recovery_out;
    ASSERT_EQ(inferedge::parse_recovery_lock(recovery, recovery_out), EvidenceError::kOk);
    EXPECT_EQ(recovery_out.bytes, recovery);
    EXPECT_EQ(recovery_out.owner_run_id, "run-a");
    reject_malformed(recovery, [](const std::string& bytes) {
        inferedge::RecoveryLockRecord out;
        return inferedge::parse_recovery_lock(bytes, out);
    });

    StaleLockAuthorization stale;
    stale.authorization_id = "a";
    stale.lock_path = "lock/evidence.lock";
    stale.lock_owner_run_id = "run-a";
    stale.lock_file_size = 1;
    stale.lock_file_sha256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    stale.acquired_date = "2026-01-01";
    stale.authorization_date = "2026-01-03";
    std::string stale_bytes;
    ASSERT_EQ(serialize_stale_lock_authorization(stale, stale_bytes), EvidenceError::kOk);
    StaleLockAuthorization stale_round;
    ASSERT_EQ(parse_stale_lock_authorization(stale_bytes, stale_round), EvidenceError::kOk);
    EXPECT_EQ(stale_round.bytes, stale_bytes);
    reject_malformed(stale_bytes, [](const std::string& bytes) {
        StaleLockAuthorization out;
        return parse_stale_lock_authorization(bytes, out);
    });

    QuarantineDeletionAuthorization quarantine;
    quarantine.authorization_id = "a";
    quarantine.target_id = "qrun-a";
    quarantine.target_file_size = 1;
    quarantine.target_file_sha256 = stale.lock_file_sha256;
    quarantine.authorization_date = "2026-03-31";
    std::string quarantine_bytes;
    ASSERT_EQ(serialize_quarantine_deletion(quarantine, quarantine_bytes), EvidenceError::kOk);
    QuarantineDeletionAuthorization quarantine_round;
    ASSERT_EQ(parse_quarantine_deletion(quarantine_bytes, quarantine_round), EvidenceError::kOk);
    EXPECT_EQ(quarantine_round.bytes, quarantine_bytes);

    DeletionJournal journal;
    journal.quarantine_id = "qrun-a";
    journal.authorization_id = "a";
    journal.authorization_file_size = quarantine_bytes.size();
    journal.authorization_file_sha256 = stale.lock_file_sha256;
    std::string journal_bytes;
    ASSERT_EQ(inferedge::serialize_deletion_journal(journal, journal_bytes), EvidenceError::kOk);
    DeletionJournal journal_round;
    ASSERT_EQ(inferedge::parse_deletion_journal(journal_bytes, journal_round), EvidenceError::kOk);
    EXPECT_EQ(journal_round.bytes, journal_bytes);
    reject_malformed(journal_bytes, [](const std::string& bytes) {
        DeletionJournal out;
        return inferedge::parse_deletion_journal(bytes, out);
    });

    QuarantineMetadata metadata;
    metadata.quarantine_id = "qrun-a";
    metadata.associated_phase_id = "9-c6";
    metadata.associated_run_id = "qrun-a";
    metadata.directory_id = "quarantine/qrun-a";
    metadata.creation_date = "2026-03-01";
    metadata.reason_code = "other_rejected";
    metadata.accepted_run_manifest_path = "none";
    metadata.accepted_run_manifest_file_size = 0;
    metadata.accepted_run_manifest_file_sha256 = std::string(kZeroDigest);
    metadata.retention_start = "2026-03-01";
    metadata.retention_deadline = "2026-03-31";
    std::string metadata_bytes;
    ASSERT_EQ(serialize_quarantine_metadata(metadata, metadata_bytes), EvidenceError::kOk);
    QuarantineMetadata metadata_round;
    ASSERT_EQ(parse_quarantine_metadata(metadata_bytes, metadata_round), EvidenceError::kOk);
    EXPECT_EQ(metadata_round.bytes, metadata_bytes);
    reject_malformed(metadata_bytes, [](const std::string& bytes) {
        QuarantineMetadata out;
        return parse_quarantine_metadata(bytes, out);
    });

    CleanupPendingRecord pending;
    pending.authorization_id = "auth-a";
    pending.authorization_file_size = 4;
    pending.authorization_file_sha256 = stale.lock_file_sha256;
    pending.deletion_list_id = "list-a";
    pending.deletion_list_file_size = 4;
    pending.deletion_list_file_sha256 = stale.lock_file_sha256;
    pending.owner_run_id = "run-a";
    pending.creation_date = "2026-02-01";
    std::string pending_bytes;
    ASSERT_EQ(serialize_cleanup_pending(pending, pending_bytes), EvidenceError::kOk);
    CleanupPendingRecord pending_round;
    ASSERT_EQ(parse_cleanup_pending(pending_bytes, pending_round), EvidenceError::kOk);
    EXPECT_EQ(pending_round.bytes, pending_bytes);
    reject_malformed(pending_bytes, [](const std::string& bytes) {
        CleanupPendingRecord out;
        return parse_cleanup_pending(bytes, out);
    });

    inferedge::CleanupCompletionRecord completion;
    completion.authorization_id = "auth-a";
    completion.pending_file_size = pending_bytes.size();
    completion.pending_file_sha256 = stale.lock_file_sha256;
    completion.authorization_file_size = 4;
    completion.authorization_file_sha256 = stale.lock_file_sha256;
    completion.deletion_list_id = "list-a";
    completion.deletion_list_file_size = 4;
    completion.deletion_list_file_sha256 = stale.lock_file_sha256;
    completion.owner_run_id = "run-a";
    completion.completion_date = "2026-02-02";
    std::string completion_bytes;
    ASSERT_EQ(inferedge::serialize_cleanup_completion(completion, completion_bytes),
              EvidenceError::kOk);
    inferedge::CleanupCompletionRecord completion_round;
    ASSERT_EQ(inferedge::parse_cleanup_completion(completion_bytes, completion_round),
              EvidenceError::kOk);
    EXPECT_EQ(completion_round.bytes, completion_bytes);
    reject_malformed(completion_bytes, [](const std::string& bytes) {
        inferedge::CleanupCompletionRecord out;
        return inferedge::parse_cleanup_completion(bytes, out);
    });

    const std::string frozen_deletion =
        "schema_version=1\n"
        "record_kind=deletion_authorization\n"
        "authorization_id=a\n"
        "target_kind=quarantine_metadata\n"
        "target_id=qrun-a\n"
        "target_file_size=1\n"
        "target_file_sha256=ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n"
        "authorization_date=2026-03-31\n"
        "test_only=no\n"
        "record_sha256=a9383b3a3c5a6e6fd1067333ed6d1e10d0c8be3454b57d4c611d680e7ab80009\n";
    QuarantineDeletionAuthorization frozen_out;
    frozen_out.authorization_id = "KEEP";
    ASSERT_EQ(parse_quarantine_deletion(frozen_deletion, frozen_out), EvidenceError::kOk);
    EXPECT_EQ(frozen_out.authorization_id, "a");
    std::string frozen_bad = frozen_deletion;
    frozen_bad.replace(frozen_bad.find("record_sha256=") + std::string("record_sha256=").size(), 1,
                       "b");
    frozen_out.authorization_id = "KEEP";
    EXPECT_NE(parse_quarantine_deletion(frozen_bad, frozen_out), EvidenceError::kOk);
    EXPECT_EQ(frozen_out.authorization_id, "KEEP");
    std::string no_lf = frozen_deletion.substr(0, frozen_deletion.size() - 1);
    EXPECT_NE(parse_quarantine_deletion(no_lf, frozen_out), EvidenceError::kOk);
    EXPECT_EQ(frozen_out.authorization_id, "KEEP");
    std::string crlf = frozen_deletion;
    crlf.insert(crlf.find('\n'), "\r");
    EXPECT_NE(parse_quarantine_deletion(crlf, frozen_out), EvidenceError::kOk);
    EXPECT_EQ(frozen_out.authorization_id, "KEEP");
    reject_malformed(frozen_deletion, [](const std::string& bytes) {
        QuarantineDeletionAuthorization out;
        out.authorization_id = "KEEP";
        const EvidenceError status = parse_quarantine_deletion(bytes, out);
        EXPECT_EQ(out.authorization_id, "KEEP");
        return status;
    });

    ReservationDeletionAuthorization reservation_deletion;
    reservation_deletion.authorization_id = "auth-a";
    reservation_deletion.target_id = "batch-a";
    reservation_deletion.reservation_path = "lock/reservation-batch-a.record";
    reservation_deletion.target_file_size = 1;
    reservation_deletion.target_file_sha256 = stale.lock_file_sha256;
    reservation_deletion.owner_run_id = "run-a";
    reservation_deletion.deletion_list_id = "list-a";
    reservation_deletion.deletion_list_path = "admission/deletion/reservation-batch-a.files.record";
    reservation_deletion.deletion_list_file_size = 1;
    reservation_deletion.deletion_list_file_sha256 = stale.lock_file_sha256;
    reservation_deletion.authorization_date = "2026-02-01";
    std::string reservation_deletion_bytes;
    ASSERT_EQ(serialize_reservation_deletion(reservation_deletion, reservation_deletion_bytes),
              EvidenceError::kOk);
    ReservationDeletionAuthorization reservation_round;
    ASSERT_EQ(inferedge::parse_reservation_deletion(reservation_deletion_bytes, reservation_round),
              EvidenceError::kOk);
    EXPECT_EQ(reservation_round.bytes, reservation_deletion_bytes);
    reject_malformed(reservation_deletion_bytes, [](const std::string& bytes) {
        ReservationDeletionAuthorization out;
        out.authorization_id = "KEEP";
        const EvidenceError status = inferedge::parse_reservation_deletion(bytes, out);
        EXPECT_EQ(out.authorization_id, "KEEP");
        return status;
    });

    ReservationDeletionList list;
    list.list_id = "list-a";
    list.reservation_id = "batch-a";
    list.owner_run_id = "run-a";
    DeletionListEntry line;
    line.relative_path = "lock/reservation-batch-a.record";
    line.deletion_entry_size = 1;
    line.deletion_entry_sha256 = stale.lock_file_sha256;
    line.object_class = CleanupObjectClass::kEvidenceReservation;
    list.entries = {line};
    list.entry_count = 1;
    std::string list_bytes;
    ASSERT_EQ(serialize_reservation_deletion_list(list, list_bytes), EvidenceError::kOk);
    ReservationDeletionList list_round;
    ASSERT_EQ(inferedge::parse_reservation_deletion_list(list_bytes, list_round),
              EvidenceError::kOk);
    EXPECT_EQ(list_round.bytes, list_bytes);
    reject_malformed(list_bytes, [](const std::string& bytes) {
        ReservationDeletionList out;
        out.list_id = "KEEP";
        const EvidenceError status = inferedge::parse_reservation_deletion_list(bytes, out);
        EXPECT_EQ(out.list_id, "KEEP");
        return status;
    });
}

TEST(EvidenceStore, AuthenticationFailureExposesNoCapability) {
    static_assert(!std::is_copy_constructible_v<inferedge::AuthenticatedCapability>);
    static_assert(!std::is_copy_assignable_v<inferedge::AuthenticatedCapability>);
    static_assert(std::is_move_constructible_v<inferedge::AuthenticatedCapability>);
    static_assert(!std::is_copy_constructible_v<inferedge::CapabilityResult>);
    TempRoot evidence;
    TempRoot admission;
    TempRoot trust;
    ASSERT_TRUE(evidence.ok() && admission.ok() && trust.ok());
    ManualClock clock;
    EvidenceLock lock = hold_lock(evidence.path(), clock);
    plant(evidence.path(), "provenance/item-a/keep.bin", "keep");
    inferedge::CapabilityResult failed = inferedge::AuthenticatedCapability::authenticate(
        "quarantine-deletion", "bytes", "deletion/deletion-a.record", "qrun-a", trust.path(),
        evidence.path(), admission.path(), "trust-a", "2026-03-31", "key-a", "2020-01-01",
        "2100-12-31");
    EXPECT_EQ(failed.status(), EvidenceError::kUnavailable);
    EXPECT_FALSE(failed.has_capability());
    inferedge::AuthenticatedCapability extracted = failed.release_capability();
    EXPECT_FALSE(extracted.holds_authority());
    EXPECT_FALSE(failed.has_capability());
    const EvidenceError ignored = failed.status();
    EXPECT_EQ(ignored, EvidenceError::kUnavailable);
    EXPECT_FALSE(failed.release_capability().holds_authority());
    EXPECT_EQ(delete_quarantine(lock, admission.path(), "2026-03-31", "trust-a", "key-a", "bytes",
                                &extracted, &clock),
              EvidenceError::kUnavailable);
    inferedge::CapabilityResult again = inferedge::AuthenticatedCapability::authenticate(
        "stale-lock", "other", "lock/stale-a.record", "lock/evidence.lock", trust.path(),
        evidence.path(), admission.path(), "trust-a", "2026-01-03", "key-a", "2020-01-01",
        "2100-12-31");
    failed = std::move(again);
    EXPECT_EQ(failed.status(), EvidenceError::kUnavailable);
    EXPECT_FALSE(failed.has_capability());
    EXPECT_EQ(again.status(), EvidenceError::kUnavailable);
    EXPECT_FALSE(again.has_capability());
    inferedge::AuthenticatedCapability moved_from;
    inferedge::AuthenticatedCapability destination = std::move(moved_from);
    EXPECT_FALSE(moved_from.holds_authority());
    EXPECT_FALSE(destination.holds_authority());
    EXPECT_EQ(delete_quarantine(lock, admission.path(), "2026-03-31", "trust-a", "key-a", "bytes",
                                &destination, &clock),
              EvidenceError::kUnavailable);
    EXPECT_EQ(slurp(join_path(evidence.path(), "provenance/item-a/keep.bin")), "keep");
    EXPECT_EQ(lock.release(), EvidenceError::kOk);
}

TEST(EvidenceStore, ConcurrentNullCapabilitiesStayUnavailable) {
    TempRoot evidence_a;
    TempRoot evidence_b;
    TempRoot admission;
    ASSERT_TRUE(evidence_a.ok() && evidence_b.ok() && admission.ok());
    ManualClock clock_a;
    ManualClock clock_b;
    EvidenceLock lock_a = hold_lock(evidence_a.path(), clock_a);
    EvidenceLock lock_b = hold_lock(evidence_b.path(), clock_b, "run-b");
    std::atomic<int> unavailable{0};
    std::thread first([&] {
        if (delete_quarantine(lock_a, admission.path(), "2026-03-31", "trust-a", "key-a", "bytes",
                              nullptr, &clock_a) == EvidenceError::kUnavailable) {
            unavailable.fetch_add(1);
        }
    });
    std::thread second([&] {
        if (delete_quarantine(lock_b, admission.path(), "2026-03-31", "trust-b", "key-b", "bytes",
                              nullptr, &clock_b) == EvidenceError::kUnavailable) {
            unavailable.fetch_add(1);
        }
    });
    first.join();
    second.join();
    EXPECT_EQ(unavailable.load(), 2);
    EXPECT_EQ(lock_a.release(), EvidenceError::kOk);
    EXPECT_EQ(lock_b.release(), EvidenceError::kOk);
}

TEST(EvidenceStore, CapabilityResultLifetime) {
    inferedge::CapabilityResult initial;
    EXPECT_EQ(initial.status(), EvidenceError::kUnavailable);
    EXPECT_FALSE(initial.has_capability());
    EXPECT_FALSE(initial.release_capability().holds_authority());
    EXPECT_FALSE(initial.release_capability().holds_authority());

    TempRoot evidence;
    TempRoot admission;
    TempRoot trust;
    ASSERT_TRUE(evidence.ok() && admission.ok() && trust.ok());
    auto unavailable = inferedge::AuthenticatedCapability::authenticate(
        "quarantine-deletion", "bytes", "deletion/deletion-a.record", "qrun-a", trust.path(),
        evidence.path(), admission.path(), "trust-a", "2026-03-31", "key-a", "2020-01-01",
        "2100-12-31");
    EXPECT_EQ(unavailable.status(), EvidenceError::kUnavailable);
    EXPECT_FALSE(unavailable.has_capability());
    const EvidenceError ignored = unavailable.status();
    EXPECT_EQ(ignored, EvidenceError::kUnavailable);
    EXPECT_FALSE(unavailable.has_capability());
    {
        auto destroyed = inferedge::AuthenticatedCapability::authenticate(
            "stale-lock", "bytes", "lock/stale-a.record", "lock/evidence.lock", trust.path(),
            evidence.path(), admission.path(), "trust-a", "2026-01-03", "key-a", "2020-01-01",
            "2100-12-31");
        EXPECT_FALSE(destroyed.has_capability());
    }
    inferedge::AuthenticatedCapability owned;
    {
        auto extracted_from = inferedge::AuthenticatedCapability::authenticate(
            "reservation-deletion", "bytes", "deletion/deletion-a.record", "batch-a", trust.path(),
            evidence.path(), admission.path(), "trust-a", "2026-02-01", "key-a", "2020-01-01",
            "2100-12-31");
        owned = extracted_from.release_capability();
        EXPECT_FALSE(extracted_from.has_capability());
        EXPECT_FALSE(extracted_from.release_capability().holds_authority());
    }
    EXPECT_FALSE(owned.holds_authority());
    inferedge::AuthenticatedCapability moved_from;
    inferedge::AuthenticatedCapability destination = std::move(moved_from);
    EXPECT_FALSE(moved_from.holds_authority());
    EXPECT_FALSE(destination.holds_authority());
    auto later = inferedge::AuthenticatedCapability::authenticate(
        "quarantine-deletion", "later", "deletion/deletion-b.record", "qrun-b", trust.path(),
        evidence.path(), admission.path(), "trust-b", "2026-04-01", "key-b", "2020-01-01",
        "2100-12-31");
    unavailable = std::move(later);
    EXPECT_EQ(unavailable.status(), EvidenceError::kUnavailable);
    EXPECT_FALSE(unavailable.has_capability());
    EXPECT_EQ(later.status(), EvidenceError::kUnavailable);
    EXPECT_FALSE(later.has_capability());
    EXPECT_FALSE(unavailable.release_capability().holds_authority());
}

TEST(EvidenceStore, UnavailableCapabilityPreservesQuarantine) {
    TempRoot evidence;
    TempRoot admission;
    TempRoot trust;
    ASSERT_TRUE(evidence.ok() && admission.ok() && trust.ok());
    ManualClock clock;
    clock.date = "2026-03-01";
    EvidenceLock lock = hold_lock(evidence.path(), clock);
    ASSERT_EQ(create_quarantine(lock, "qrun-a", "9-c6", "other_rejected", "none", {}, &clock),
              EvidenceError::kOk);
    const std::string metadata =
        slurp(join_path(evidence.path(), "quarantine/qrun-a/quarantine.record"));
    QuarantineDeletionAuthorization fields;
    fields.authorization_id = "a";
    fields.target_id = "qrun-a";
    fields.target_file_size = metadata.size();
    fields.target_file_sha256 = digest_hex(metadata);
    fields.authorization_date = "2026-03-31";
    std::string bytes;
    ASSERT_EQ(serialize_quarantine_deletion(fields, bytes), EvidenceError::kOk);
    plant(admission.path(), "deletion/deletion-a.record", bytes);
    plant(evidence.path(), "quarantine/qrun-a/extra.bin", "extra");
    clock.date = "2026-03-31";
    auto failed = inferedge::AuthenticatedCapability::authenticate(
        "quarantine-deletion", bytes, "deletion/deletion-a.record", "qrun-a", trust.path(),
        evidence.path(), admission.path(), "trust-a", "2026-03-31", "key-a", "2020-01-01",
        "2100-12-31");
    inferedge::AuthenticatedCapability capability = failed.release_capability();
    EXPECT_EQ(delete_quarantine(lock, admission.path(), "2026-03-31", "trust-a", "key-a", bytes,
                                &capability, &clock),
              EvidenceError::kUnavailable);
    EXPECT_EQ(slurp(join_path(evidence.path(), "quarantine/qrun-a/quarantine.record")), metadata);
    EXPECT_EQ(slurp(join_path(evidence.path(), "quarantine/qrun-a/extra.bin")), "extra");
    EXPECT_FALSE(path_exists(join_path(evidence.path(), "quarantine/qrun-a/deletion.record")));
    EXPECT_EQ(lock.release(), EvidenceError::kOk);
}

TEST(EvidenceStore, UnavailableCleanupLeavesReservation) {
    TempRoot evidence;
    TempRoot admission;
    ASSERT_TRUE(evidence.ok() && admission.ok());
    ManualClock clock;
    clock.date = "2026-02-01";
    EvidenceLock lock = hold_lock(evidence.path(), clock, "cleaner");
    ReservationEntry staged;
    staged.staging_relative_path = "partial/run-a/a.bin";
    staged.final_relative_path = "provenance/item-a/a.bin";
    staged.reserved_object_size = 6;
    staged.reserved_object_sha256 = digest_hex("file-a");
    staged.object_class = EvidenceObjectClass::kPublishedBlob;
    EvidenceReservation reservation;
    ASSERT_EQ(serialize_evidence_reservation("run-a", {staged}, reservation), EvidenceError::kOk);
    plant(evidence.path(), "lock/reservation-batch-a.record", reservation.bytes);
    plant(evidence.path(), "partial/run-a/a.bin", "file-a");
    plant(evidence.path(), "provenance/item-a/keep.bin", "keep");
    EXPECT_EQ(cleanup_reservation(lock, admission.path(), "2026-02-01", "trust-a", "key-a", "bytes",
                                  nullptr, "list", &clock),
              EvidenceError::kUnavailable);
    EXPECT_EQ(slurp(join_path(evidence.path(), "partial/run-a/a.bin")), "file-a");
    EXPECT_EQ(slurp(join_path(evidence.path(), "lock/reservation-batch-a.record")),
              reservation.bytes);
    EXPECT_EQ(slurp(join_path(evidence.path(), "provenance/item-a/keep.bin")), "keep");
    EXPECT_EQ(lock.release(), EvidenceError::kOk);
}

TEST(EvidenceStore, RootReplacementWithoutAuthorityPreservesTrees) {
    TempRoot parent;
    ASSERT_TRUE(parent.ok());
    const std::string evidence = join_path(parent.path(), "evidence");
    const std::string admission = join_path(parent.path(), "admission");
    ASSERT_EQ(mkdir(evidence.c_str(), 0700), 0);
    ASSERT_EQ(mkdir(admission.c_str(), 0700), 0);
    ManualClock clock;
    clock.date = "2026-03-01";
    EvidenceLock lock = hold_lock(evidence, clock);
    ASSERT_EQ(create_quarantine(lock, "qrun-a", "9-c6", "other_rejected", "none", {}, &clock),
              EvidenceError::kOk);
    const std::string metadata = slurp(join_path(evidence, "quarantine/qrun-a/quarantine.record"));
    const std::string saved = join_path(parent.path(), "evidence-saved");
    ASSERT_EQ(rename(evidence.c_str(), saved.c_str()), 0);
    ASSERT_EQ(mkdir(evidence.c_str(), 0700), 0);
    plant(evidence, "decoy.txt", "new");
    clock.date = "2026-03-31";
    EXPECT_EQ(delete_quarantine(lock, admission, "2026-03-31", "trust-a", "key-a", "bytes", nullptr,
                                &clock),
              EvidenceError::kUnavailable);
    EXPECT_EQ(slurp(join_path(evidence, "decoy.txt")), "new");
    EXPECT_EQ(slurp(join_path(saved, "quarantine/qrun-a/quarantine.record")), metadata);
    EXPECT_FALSE(path_exists(join_path(saved, "quarantine/qrun-a/deletion.record")));
    ASSERT_EQ(rename(evidence.c_str(), join_path(parent.path(), "evidence-new").c_str()), 0);
    ASSERT_EQ(rename(saved.c_str(), evidence.c_str()), 0);
    const std::string linked = join_path(parent.path(), "linked-evidence");
    ASSERT_EQ(symlink(evidence.c_str(), linked.c_str()), 0);
    EXPECT_EQ(
        delete_quarantine(lock, linked, "2026-03-31", "trust-a", "key-a", "bytes", nullptr, &clock),
        EvidenceError::kUnavailable);
    EXPECT_EQ(slurp(join_path(evidence, "quarantine/qrun-a/quarantine.record")), metadata);
    EXPECT_EQ(lock.release(), EvidenceError::kOk);
}

TEST(EvidenceStore, QuarantineOpenInventoryRejectsUndeclaredPayloads) {
    TempRoot evidence;
    TempRoot admission;
    ASSERT_TRUE(evidence.ok() && admission.ok());
    ManualClock clock;
    clock.date = "2026-03-01";
    EvidenceLock lock = hold_lock(evidence.path(), clock);
    const char* metadata_only[] = {
        "checksum_rejected", "reproduction_mismatch", "cap_rejected",  "symlink_rejected",
        "signature_failed",  "invariant_mismatch",    "other_rejected"};
    for (const char* reason : metadata_only) {
        EXPECT_EQ(create_quarantine(lock, std::string("echo-") + reason, "9-c6", reason, "none",
                                    {QuarantinePayload{"recipe-echo.bin", "echo"}}, &clock),
                  EvidenceError::kForbidden);
    }
    ASSERT_EQ(create_quarantine(lock, "qrun-meta", "9-c6", "checksum_rejected", "none", {}, &clock),
              EvidenceError::kOk);
    ASSERT_EQ(create_quarantine(lock, "qrun-echo", "9-c6", "recipe_echo_rejected", "none",
                                {QuarantinePayload{"recipe-echo.bin", "echo"}}, &clock),
              EvidenceError::kOk);
    plant(evidence.path(), "quarantine/qrun-meta/.hidden", "secret");
    clock.date = "2026-03-31";
    EXPECT_EQ(delete_quarantine(lock, admission.path(), "2026-03-31", "trust-a", "key-a", "bytes",
                                nullptr, &clock),
              EvidenceError::kUnavailable);
    EXPECT_EQ(slurp(join_path(evidence.path(), "quarantine/qrun-meta/.hidden")), "secret");
    EXPECT_TRUE(path_exists(join_path(evidence.path(), "quarantine/qrun-meta/quarantine.record")));
    EXPECT_TRUE(path_exists(join_path(evidence.path(), "quarantine/qrun-echo/recipe-echo.bin")));
    EXPECT_FALSE(path_exists(join_path(evidence.path(), "quarantine/qrun-meta/deletion.record")));
    EXPECT_EQ(lock.release(), EvidenceError::kOk);
}

TEST(EvidenceStore, LockMoveDoesNotRelease) {
    TempRoot root;
    ASSERT_TRUE(root.ok());
    ManualClock clock;
    EvidenceLock original = hold_lock(root.path(), clock);
    EvidenceLock moved = std::move(original);
    EXPECT_FALSE(original.held());
    EXPECT_TRUE(moved.held());
    EXPECT_TRUE(path_exists(join_path(root.path(), "lock/evidence.lock")));
    {
        EvidenceLock dropped = std::move(moved);
        EXPECT_TRUE(dropped.held());
    }
    EXPECT_TRUE(path_exists(join_path(root.path(), "lock/evidence.lock")));
}

std::string seal_record(std::string prefix) {
    if (prefix.empty() || prefix.back() != '\n') {
        prefix.push_back('\n');
    }
    const std::string digest = digest_hex(prefix);
    prefix += "record_sha256=";
    prefix += digest;
    prefix.push_back('\n');
    return prefix;
}

void append_field(std::string& prefix, std::string_view key, std::string_view value) {
    prefix.append(key);
    prefix.push_back('=');
    prefix.append(value);
    prefix.push_back('\n');
}

std::string generation_one_trust_bytes(bool reverse_digest) {
    const std::string zero(64, '0');
    std::string prefix;
    append_field(prefix, "schema_version", "1");
    append_field(prefix, "record_kind", "trust_state");
    append_field(prefix, "trust_id", "trust-gen1");
    append_field(prefix, "trust_generation", "1");
    append_field(prefix, "effective_date", "2026-03-31");
    append_field(prefix, "terminal_transition_id", "genesis-v1");
    append_field(prefix, "terminal_transition_path", "genesis/transition.record");
    if (reverse_digest) {
        append_field(prefix, "terminal_transition_file_size", "1");
        append_field(prefix, "terminal_transition_file_sha256", zero);
    }
    append_field(prefix, "revocation_generation", "0");
    append_field(prefix, "revocation_id", "none");
    append_field(prefix, "revocation_file_size", "0");
    append_field(prefix, "revocation_file_sha256", zero);
    append_field(prefix, "config_id", "none");
    append_field(prefix, "configuration_file_size", "0");
    append_field(prefix, "configuration_file_sha256", zero);
    append_field(prefix, "registry_id", "registry-gen1");
    append_field(prefix, "authority_registry_file_size", "1");
    append_field(prefix, "authority_registry_file_sha256", zero);
    append_field(prefix, "authority_envelope_id", "envelope-gen1");
    append_field(prefix, "authority_envelope_file_size", "1");
    append_field(prefix, "authority_envelope_file_sha256", zero);
    append_field(prefix, "bootstrap_key_handle", "bootstrap-key-v1");
    append_field(prefix, "bootstrap_key_size", "32");
    append_field(prefix, "bootstrap_key_sha256", zero);
    append_field(prefix, "admission_manifest_id", "manifest-gen1");
    append_field(prefix, "admission_root_manifest_file_size", "1");
    append_field(prefix, "admission_root_manifest_file_sha256", zero);
    append_field(prefix, "host_id", "host-gen1");
    append_field(prefix, "publication_kind", "native_initialization");
    append_field(prefix, "publication_authorization_id", "none");
    append_field(prefix, "publication_authorization_path", "none");
    append_field(prefix, "publication_authorization_file_size", "0");
    append_field(prefix, "publication_authorization_file_sha256", zero);
    append_field(prefix, "publication_authorization_preimage_sha256", zero);
    append_field(prefix, "signer_key_id", "none");
    append_field(prefix, "signer_publication_key_size", "32");
    append_field(prefix, "signer_publication_key_sha256", std::string(63, '0') + "1");
    append_field(prefix, "signature_family_id", "trust_transition");
    append_field(prefix, "lineage_kind", "native");
    append_field(prefix, "binding_record_kind", "transition_operation_binding");
    append_field(prefix, "binding_record_size", "1");
    append_field(prefix, "binding_record_sha256", zero);
    append_field(prefix, "telemetry_mode", "disabled");
    append_field(prefix, "telemetry_governance_id", "none");
    append_field(prefix, "telemetry_governance_file_size", "0");
    append_field(prefix, "telemetry_governance_file_sha256", zero);
    append_field(prefix, "predecessor_trust_id", "none");
    append_field(prefix, "predecessor_file_size", "0");
    append_field(prefix, "predecessor_file_sha256", zero);
    return seal_record(std::move(prefix));
}

TEST(EvidenceStore, TrustStateRejectsReverseTransitionDigest) {
    TempRoot evidence;
    TempRoot admission;
    TempRoot trust;
    ASSERT_TRUE(evidence.ok() && admission.ok() && trust.ok());
    plant(trust.path(), "current", generation_one_trust_bytes(true));
    const inferedge::CapabilityResult rejected = inferedge::AuthenticatedCapability::authenticate(
        "quarantine-deletion", "bytes", "deletion/deletion-a.record", "qrun-a", trust.path(),
        evidence.path(), admission.path(), "trust-gen1", "2026-03-31", "key-a", "2020-01-01",
        "2100-12-31");
    EXPECT_EQ(rejected.status(), EvidenceError::kSchema);
    EXPECT_FALSE(rejected.has_capability());

    TempRoot trust_forward;
    ASSERT_TRUE(trust_forward.ok());
    plant(trust_forward.path(), "current", generation_one_trust_bytes(false));
    const inferedge::CapabilityResult parsed = inferedge::AuthenticatedCapability::authenticate(
        "quarantine-deletion", "bytes", "deletion/deletion-a.record", "qrun-a",
        trust_forward.path(), evidence.path(), admission.path(), "trust-gen1", "2026-03-31",
        "key-a", "2020-01-01", "2100-12-31");
    EXPECT_EQ(parsed.status(), EvidenceError::kUnavailable);
    EXPECT_FALSE(parsed.has_capability());
}

std::string signed_message(const TestKey& key, std::string_view message) {
    unsigned char signature[64];
    test_sign(signature, reinterpret_cast<const unsigned char*>(message.data()), message.size(),
              key);
    return std::string(reinterpret_cast<const char*>(signature), 64);
}

std::string index_self_prefix_digest(const std::string& sealed) {
    const std::size_t checksum = sealed.rfind("\nrecord_sha256=");
    EXPECT_NE(checksum, std::string::npos);
    return digest_hex(std::string_view(sealed).substr(0, checksum + 1));
}

const char* kTransitionDomain = "INFEREDGE-ED25519-V1 TRUST-TRANSITION-PAYLOAD 1\n";

std::string trust_transition_message(std::string_view payload) {
    std::string message(kTransitionDomain);
    message.append(payload);
    return message;
}

void force_mode(const std::string& root, std::string_view relative, mode_t mode) {
    ASSERT_EQ(chmod(join_path(root, relative).c_str(), mode), 0);
}

std::string seal_fields(const std::vector<std::pair<std::string, std::string>>& fields) {
    std::string prefix;
    for (const auto& field : fields) {
        append_field(prefix, field.first, field.second);
    }
    return seal_record(std::move(prefix));
}

const char* kGenerationDate = "2026-03-31";
const char* kKeyFrom = "2020-01-01";
const char* kKeyUntil = "2100-12-31";
const char* kTrustId = "trust-gen1";
const char* kRegistryId = "registry-gen1";
const char* kListId = "list-gen1";
const char* kAdminId = "admin-gen1";
const char* kHostId = "host-gen1";

struct ChainOptions {
    const char* role = "trust_admin";
    const char* revoked = "no";
    const char* valid_from = kKeyFrom;
    const char* valid_until = kKeyUntil;
    bool duplicate_id = false;
    bool duplicate_digest = false;
    bool flip_registry_signature = false;
    bool flip_genesis_signature = false;
    const char* revocation_generation = "0";
    const char* revocation_id = "none";
};

struct ChainMaterial {
    TestKey admin{};
    std::string key_index;
    std::string trust_bytes;
    std::string binding;
};

void plant_generation_one(const std::string& trust_root, const std::string& admission_root,
                          const std::string& evidence_root, ChainMaterial& out,
                          const ChainOptions& options = {}) {
    const std::string zero(64, '0');
    unsigned char seed[32];
    std::memset(seed, 0x42, sizeof(seed));
    TestKey bootstrap;
    test_keypair(bootstrap, seed);
    std::memset(seed, 0x43, sizeof(seed));
    test_keypair(out.admin, seed);
    const std::string bootstrap_raw(reinterpret_cast<const char*>(bootstrap.public_key), 32);
    const std::string admin_raw(reinterpret_cast<const char*>(out.admin.public_key), 32);
    const std::string admin_digest = digest_hex(admin_raw);
    const std::string key_path = "benchmarks/generative_phase9c/keys/admin-gen1.bin";
    out.key_index = seal_fields({{"schema_version", "1"},
                                 {"record_kind", "blob_index"},
                                 {"blob_id", kAdminId},
                                 {"blob_class", "public_key"},
                                 {"raw_size", "32"},
                                 {"raw_sha256", admin_digest},
                                 {"storage_path", key_path},
                                 {"review_note_id", "none"},
                                 {"review_note_file_size", "0"},
                                 {"review_note_file_sha256", zero}});
    const int entry_count = (options.duplicate_id || options.duplicate_digest) ? 2 : 1;
    std::string list_prefix;
    append_field(list_prefix, "schema_version", "1");
    append_field(list_prefix, "record_kind", "authority_entry_list");
    append_field(list_prefix, "list_id", kListId);
    append_field(list_prefix, "entry_count", std::to_string(entry_count));
    const std::string entry =
        std::string(kAdminId) + " " + options.role + " " + kAdminId + " 32 " + admin_digest + " " +
        kAdminId + " " + std::to_string(out.key_index.size()) + " " + digest_hex(out.key_index) +
        " " + options.valid_from + " " + options.valid_until + " " + options.revoked + "\n";
    list_prefix += entry;
    if (options.duplicate_id) {
        list_prefix += entry;
    }
    if (options.duplicate_digest) {
        list_prefix += std::string("admin-gen2 ") + options.role + " admin-gen2 32 " +
                       admin_digest + " admin-gen2 " + std::to_string(out.key_index.size()) + " " +
                       digest_hex(out.key_index) + " " + options.valid_from + " " +
                       options.valid_until + " " + options.revoked + "\n";
    }
    const std::string list_bytes = seal_record(std::move(list_prefix));
    const std::string registry =
        seal_fields({{"schema_version", "1"},
                     {"record_kind", "authority_registry"},
                     {"registry_id", kRegistryId},
                     {"predecessor_registry_id", "none"},
                     {"predecessor_registry_file_size", "0"},
                     {"predecessor_registry_file_sha256", zero},
                     {"entry_list_id", kListId},
                     {"entry_list_file_size", std::to_string(list_bytes.size())},
                     {"entry_list_file_sha256", digest_hex(list_bytes)},
                     {"entry_count", std::to_string(entry_count)}});
    std::string registry_signature = signed_message(bootstrap, registry);
    if (options.flip_registry_signature) {
        registry_signature[63] = static_cast<char>(registry_signature[63] ^ 0x01);
    }
    const std::string registry_blob = "registry-envelope-sigblob-registry-gen1";
    const std::string registry_index = "registry-envelope-sigindex-registry-gen1";
    const std::string registry_signature_id = "registry-envelope-signature-registry-gen1";
    const std::string registry_blob_path =
        "benchmarks/generative_phase9c/signatures/" + registry_blob + ".bin";
    const std::string registry_index_bytes =
        seal_fields({{"schema_version", "1"},
                     {"record_kind", "blob_index"},
                     {"blob_id", registry_blob},
                     {"blob_class", "signature_bytes"},
                     {"raw_size", "64"},
                     {"raw_sha256", digest_hex(registry_signature)},
                     {"storage_path", registry_blob_path},
                     {"review_note_id", "none"},
                     {"review_note_file_size", "0"},
                     {"review_note_file_sha256", zero}});
    const std::string envelope =
        seal_fields({{"schema_version", "1"},
                     {"record_kind", "signature_verification"},
                     {"signature_id", registry_signature_id},
                     {"signed_object_kind", "structured_file"},
                     {"signed_object_id", "registry-registry-gen1"},
                     {"signed_object_path", "none"},
                     {"signed_object_size", std::to_string(registry.size())},
                     {"signed_object_sha256", digest_hex(registry)},
                     {"signature_blob_id", registry_blob},
                     {"signature_index_id", registry_index},
                     {"signature_index_path", "none"},
                     {"signature_index_size", std::to_string(registry_index_bytes.size())},
                     {"signature_index_sha256", digest_hex(registry_index_bytes)},
                     {"signature_size", "64"},
                     {"signature_sha256", digest_hex(registry_signature)},
                     {"signer_key_source", "bootstrap_handle"},
                     {"signer_key_id", "none"},
                     {"public_key_size", "32"},
                     {"public_key_sha256", digest_hex(bootstrap_raw)},
                     {"key_index_id", "none"},
                     {"key_index_size", "0"},
                     {"key_index_sha256", zero},
                     {"fingerprint_sha256", digest_hex(bootstrap_raw)},
                     {"algorithm", "ed25519"},
                     {"verification_result", "verified"},
                     {"verification_date", kGenerationDate},
                     {"verifier_id", "inferedge-ed25519-v1"}});
    const std::vector<std::pair<std::string, std::string>> copied = {
        {"publication_kind", "native_initialization"},
        {"publication_authorization_id", "none"},
        {"publication_authorization_path", "none"},
        {"publication_authorization_file_size", "0"},
        {"publication_authorization_file_sha256", zero},
        {"publication_authorization_preimage_sha256", zero},
        {"signer_key_id", "none"},
        {"signer_publication_key_size", "32"},
        {"signer_publication_key_sha256", admin_digest},
        {"signature_family_id", "trust_transition"},
        {"trust_id", kTrustId},
        {"trust_generation", "1"},
        {"effective_date", kGenerationDate},
        {"terminal_transition_id", "genesis-v1"},
        {"terminal_transition_path", "genesis/transition.record"},
        {"predecessor_trust_id", "none"},
        {"predecessor_file_size", "0"},
        {"predecessor_file_sha256", zero},
        {"host_id", kHostId},
        {"lineage_kind", "native"},
        {"revocation_generation", options.revocation_generation},
        {"revocation_id", options.revocation_id},
        {"revocation_file_size", "0"},
        {"revocation_file_sha256", zero},
        {"config_id", "none"},
        {"configuration_file_size", "0"},
        {"configuration_file_sha256", zero},
        {"registry_id", kRegistryId},
        {"authority_registry_file_size", std::to_string(registry.size())},
        {"authority_registry_file_sha256", digest_hex(registry)},
        {"authority_envelope_id", registry_signature_id},
        {"authority_envelope_file_size", std::to_string(envelope.size())},
        {"authority_envelope_file_sha256", digest_hex(envelope)},
        {"bootstrap_key_handle", "bootstrap-key-v1"},
        {"bootstrap_key_size", "32"},
        {"bootstrap_key_sha256", digest_hex(bootstrap_raw)},
        {"admission_manifest_id", "none"},
        {"admission_root_manifest_file_size", "0"},
        {"admission_root_manifest_file_sha256", zero}};
    std::string binding_prefix;
    append_field(binding_prefix, "schema_version", "1");
    append_field(binding_prefix, "record_kind", "transition_operation_binding");
    for (const auto& field : copied) {
        append_field(binding_prefix, field.first, field.second);
    }
    out.binding = seal_record(binding_prefix);
    out.trust_bytes = seal_fields({
        {"schema_version", "1"},
        {"record_kind", "trust_state"},
        {"trust_id", kTrustId},
        {"trust_generation", "1"},
        {"effective_date", kGenerationDate},
        {"terminal_transition_id", "genesis-v1"},
        {"terminal_transition_path", "genesis/transition.record"},
        {"revocation_generation", options.revocation_generation},
        {"revocation_id", options.revocation_id},
        {"revocation_file_size", "0"},
        {"revocation_file_sha256", zero},
        {"config_id", "none"},
        {"configuration_file_size", "0"},
        {"configuration_file_sha256", zero},
        {"registry_id", kRegistryId},
        {"authority_registry_file_size", std::to_string(registry.size())},
        {"authority_registry_file_sha256", digest_hex(registry)},
        {"authority_envelope_id", registry_signature_id},
        {"authority_envelope_file_size", std::to_string(envelope.size())},
        {"authority_envelope_file_sha256", digest_hex(envelope)},
        {"bootstrap_key_handle", "bootstrap-key-v1"},
        {"bootstrap_key_size", "32"},
        {"bootstrap_key_sha256", digest_hex(bootstrap_raw)},
        {"admission_manifest_id", "none"},
        {"admission_root_manifest_file_size", "0"},
        {"admission_root_manifest_file_sha256", zero},
        {"host_id", kHostId},
        {"publication_kind", "native_initialization"},
        {"publication_authorization_id", "none"},
        {"publication_authorization_path", "none"},
        {"publication_authorization_file_size", "0"},
        {"publication_authorization_file_sha256", zero},
        {"publication_authorization_preimage_sha256", zero},
        {"signer_key_id", "none"},
        {"signer_publication_key_size", "32"},
        {"signer_publication_key_sha256", admin_digest},
        {"signature_family_id", "trust_transition"},
        {"lineage_kind", "native"},
        {"binding_record_kind", "transition_operation_binding"},
        {"binding_record_size", std::to_string(out.binding.size())},
        {"binding_record_sha256", digest_hex(out.binding)},
        {"telemetry_mode", "disabled"},
        {"telemetry_governance_id", "none"},
        {"telemetry_governance_file_size", "0"},
        {"telemetry_governance_file_sha256", zero},
        {"predecessor_trust_id", "none"},
        {"predecessor_file_size", "0"},
        {"predecessor_file_sha256", zero},
    });
    const std::vector<std::pair<std::string, std::string>> payload_fields = {
        {"schema_version", "1"},
        {"record_kind", "trust_transition_signed_payload"},
        {"transition_id", "genesis-v1"},
        {"trust_generation", "1"},
        {"transition_date", kGenerationDate},
        {"predecessor_transition_id", "none"},
        {"predecessor_transition_path", "none"},
        {"predecessor_transition_file_size", "0"},
        {"predecessor_transition_file_sha256", zero},
        {"predecessor_successor_trust_id", "none"},
        {"predecessor_successor_trust_file_size", "0"},
        {"predecessor_successor_trust_file_sha256", zero},
        {"successor_trust_id", kTrustId},
        {"successor_trust_path", "admission/trust/trust-trust-gen1.record"},
        {"successor_trust_file_size", std::to_string(out.trust_bytes.size())},
        {"successor_trust_file_sha256", digest_hex(out.trust_bytes)},
        {"bootstrap_key_sha256", digest_hex(bootstrap_raw)},
        {"signer_key_id", "none"},
        {"registry_id", kRegistryId},
        {"authority_registry_file_size", std::to_string(registry.size())},
        {"authority_registry_file_sha256", digest_hex(registry)},
        {"publication_kind", "native_initialization"},
        {"binding_record_kind", "transition_operation_binding"},
        {"binding_record_size", std::to_string(out.binding.size())},
        {"binding_record_sha256", digest_hex(out.binding)}};
    const std::string payload = seal_fields(payload_fields);
    std::string genesis_signature = signed_message(bootstrap, trust_transition_message(payload));
    if (options.flip_genesis_signature) {
        genesis_signature[0] = static_cast<char>(genesis_signature[0] ^ 0x01);
    }
    const std::string genesis_index = seal_fields({{"schema_version", "1"},
                                                   {"record_kind", "blob_index"},
                                                   {"blob_id", "genesis-sigblob-v1"},
                                                   {"blob_class", "signature_bytes"},
                                                   {"raw_size", "64"},
                                                   {"raw_sha256", digest_hex(genesis_signature)},
                                                   {"storage_path", "genesis/signature.bin"},
                                                   {"review_note_id", "none"},
                                                   {"review_note_file_size", "0"},
                                                   {"review_note_file_sha256", zero}});
    const std::string genesis_verification =
        seal_fields({{"schema_version", "1"},
                     {"record_kind", "signature_verification"},
                     {"signature_id", "transition-signature-genesis-v1"},
                     {"signed_object_kind", "structured_file"},
                     {"signed_object_id", "transition-payload-genesis-v1"},
                     {"signed_object_path", "none"},
                     {"signed_object_size", std::to_string(payload.size())},
                     {"signed_object_sha256", digest_hex(payload)},
                     {"signature_blob_id", "genesis-sigblob-v1"},
                     {"signature_index_id", "genesis-sigindex-v1"},
                     {"signature_index_path", "none"},
                     {"signature_index_size", std::to_string(genesis_index.size())},
                     {"signature_index_sha256", index_self_prefix_digest(genesis_index)},
                     {"signature_size", "64"},
                     {"signature_sha256", digest_hex(genesis_signature)},
                     {"signer_key_source", "bootstrap_handle"},
                     {"signer_key_id", "none"},
                     {"public_key_size", "32"},
                     {"public_key_sha256", digest_hex(bootstrap_raw)},
                     {"key_index_id", "none"},
                     {"key_index_size", "0"},
                     {"key_index_sha256", zero},
                     {"fingerprint_sha256", digest_hex(bootstrap_raw)},
                     {"algorithm", "ed25519"},
                     {"verification_result", "verified"},
                     {"verification_date", kGenerationDate},
                     {"verifier_id", "inferedge-ed25519-v1"}});
    const std::string transition = seal_fields({
        {"schema_version", "1"},
        {"record_kind", "trust_transition"},
        {"transition_id", "genesis-v1"},
        {"trust_generation", "1"},
        {"transition_date", kGenerationDate},
        {"predecessor_transition_id", "none"},
        {"predecessor_transition_path", "none"},
        {"predecessor_transition_file_size", "0"},
        {"predecessor_transition_file_sha256", zero},
        {"predecessor_successor_trust_id", "none"},
        {"predecessor_successor_trust_file_size", "0"},
        {"predecessor_successor_trust_file_sha256", zero},
        {"successor_trust_id", kTrustId},
        {"successor_trust_path", "admission/trust/trust-trust-gen1.record"},
        {"successor_trust_file_size", std::to_string(out.trust_bytes.size())},
        {"successor_trust_file_sha256", digest_hex(out.trust_bytes)},
        {"bootstrap_key_sha256", digest_hex(bootstrap_raw)},
        {"signer_key_id", "none"},
        {"registry_id", kRegistryId},
        {"authority_registry_file_size", std::to_string(registry.size())},
        {"authority_registry_file_sha256", digest_hex(registry)},
        {"signed_payload_id", "transition-payload-genesis-v1"},
        {"signed_payload_path", "genesis/transition.payload"},
        {"signed_payload_size", std::to_string(payload.size())},
        {"signed_payload_sha256", digest_hex(payload)},
        {"signature_id", "transition-signature-genesis-v1"},
        {"signature_blob_id", "genesis-sigblob-v1"},
        {"signature_index_id", "genesis-sigindex-v1"},
        {"signature_file_size", std::to_string(genesis_verification.size())},
        {"signature_file_sha256", digest_hex(genesis_verification)},
        {"publication_kind", "native_initialization"},
        {"binding_record_kind", "transition_operation_binding"},
        {"binding_record_size", std::to_string(out.binding.size())},
        {"binding_record_sha256", digest_hex(out.binding)},
    });
    plant(trust_root, "key/bootstrap-key-v1", bootstrap_raw);
    plant(trust_root, "genesis/transition.payload", payload);
    force_mode(trust_root, "genesis/transition.payload", 0444);
    plant(trust_root, "genesis/transition.record", transition);
    plant(trust_root, "genesis/signature.bin", genesis_signature);
    plant(trust_root, "genesis/signature.index", genesis_index);
    plant(trust_root, "genesis/verification.record", genesis_verification);
    plant(trust_root, "current", out.trust_bytes);
    plant(admission_root, "trust/trust-trust-gen1.record", out.trust_bytes);
    plant(admission_root, "authority/registry-registry-gen1.record", registry);
    plant(admission_root, "authority/registry-registry-gen1.entries", list_bytes);
    plant(admission_root, "signature-records/" + registry_signature_id + ".record", envelope);
    plant(admission_root, "signature-indexes/" + registry_index + ".index", registry_index_bytes);
    plant(admission_root, "keys/admin-gen1.index", out.key_index);
    plant(evidence_root, registry_blob_path, registry_signature);
    plant(evidence_root, key_path, admin_raw);
    plant(evidence_root, "provenance/item-a/keep.bin", "keep-unrelated");
    plant(admission_root, "foreign-keep.txt", "foreign");
}

void plant_object_signature(const std::string& evidence_root, const std::string& admission_root,
                            const ChainMaterial& chain, std::string_view message,
                            std::string_view signature_id, std::string_view blob_id,
                            std::string_view index_id, std::string_view signed_object_id) {
    const std::string zero(64, '0');
    const std::string signature = signed_message(chain.admin, message);
    const std::string blob_path =
        "benchmarks/generative_phase9c/signatures/" + std::string(blob_id) + ".bin";
    const std::string admin_raw(reinterpret_cast<const char*>(chain.admin.public_key), 32);
    const std::string index = seal_fields({{"schema_version", "1"},
                                           {"record_kind", "blob_index"},
                                           {"blob_id", std::string(blob_id)},
                                           {"blob_class", "signature_bytes"},
                                           {"raw_size", "64"},
                                           {"raw_sha256", digest_hex(signature)},
                                           {"storage_path", blob_path},
                                           {"review_note_id", "none"},
                                           {"review_note_file_size", "0"},
                                           {"review_note_file_sha256", zero}});
    const std::string verification =
        seal_fields({{"schema_version", "1"},
                     {"record_kind", "signature_verification"},
                     {"signature_id", std::string(signature_id)},
                     {"signed_object_kind", "structured_file"},
                     {"signed_object_id", std::string(signed_object_id)},
                     {"signed_object_path", "none"},
                     {"signed_object_size", std::to_string(message.size())},
                     {"signed_object_sha256", digest_hex(message)},
                     {"signature_blob_id", std::string(blob_id)},
                     {"signature_index_id", std::string(index_id)},
                     {"signature_index_path", "none"},
                     {"signature_index_size", std::to_string(index.size())},
                     {"signature_index_sha256", digest_hex(index)},
                     {"signature_size", "64"},
                     {"signature_sha256", digest_hex(signature)},
                     {"signer_key_source", "blob"},
                     {"signer_key_id", kAdminId},
                     {"public_key_size", "32"},
                     {"public_key_sha256", digest_hex(admin_raw)},
                     {"key_index_id", kAdminId},
                     {"key_index_size", std::to_string(chain.key_index.size())},
                     {"key_index_sha256", digest_hex(chain.key_index)},
                     {"fingerprint_sha256", digest_hex(admin_raw)},
                     {"algorithm", "ed25519"},
                     {"verification_result", "verified"},
                     {"verification_date", kGenerationDate},
                     {"verifier_id", "inferedge-ed25519-v1"}});
    plant(evidence_root, blob_path, signature);
    plant(admission_root, "signature-indexes/" + std::string(index_id) + ".index", index);
    plant(admission_root, "signature-records/" + std::string(signature_id) + ".record",
          verification);
}

struct StaleFixture {
    ChainMaterial chain;
    std::string authorization;
    std::string path{"lock/stale-auth-stale-gen1.record"};
    std::string target{"lock/evidence.lock"};
};

void plant_stale_fixture(const std::string& trust_root, const std::string& admission_root,
                         const std::string& evidence_root, StaleFixture& out,
                         const ChainOptions& options = {}) {
    plant_generation_one(trust_root, admission_root, evidence_root, out.chain, options);
    EvidenceLockRecord lock_fields;
    lock_fields.owner_run_id = "run-old";
    lock_fields.owner_host_id = "host-old";
    lock_fields.acquired_date = "2026-03-01";
    std::string lock_bytes;
    ASSERT_EQ(serialize_evidence_lock(lock_fields, lock_bytes), EvidenceError::kOk);
    plant(evidence_root, "lock/evidence.lock", lock_bytes);
    StaleLockAuthorization authorization;
    authorization.authorization_id = "auth-stale-gen1";
    authorization.lock_path = "lock/evidence.lock";
    authorization.lock_owner_run_id = "run-old";
    authorization.lock_file_size = lock_bytes.size();
    authorization.lock_file_sha256 = digest_hex(lock_bytes);
    authorization.acquired_date = "2026-03-01";
    authorization.authorization_date = kGenerationDate;
    ASSERT_EQ(serialize_stale_lock_authorization(authorization, out.authorization),
              EvidenceError::kOk);
    plant(admission_root, out.path, out.authorization);
    plant_object_signature(evidence_root, admission_root, out.chain, out.authorization,
                           "stalelock-signature-auth-stale-gen1",
                           "stalelock-sigblob-auth-stale-gen1",
                           "stalelock-sigindex-auth-stale-gen1", "stale-auth-stale-gen1");
}

inferedge::CapabilityResult
authenticate_stale(const std::string& trust_root, const std::string& evidence_root,
                   const std::string& admission_root, const StaleFixture& fixture,
                   std::string_view trust_id = kTrustId, std::string_view date = kGenerationDate,
                   std::string_view signer = kAdminId, std::string_view from = kKeyFrom,
                   std::string_view until = kKeyUntil, std::string_view path = {},
                   std::string_view target = {}, std::string_view bytes = {}) {
    return inferedge::AuthenticatedCapability::authenticate(
        "stale-lock", bytes.empty() ? fixture.authorization : bytes,
        path.empty() ? fixture.path : path, target.empty() ? fixture.target : target, trust_root,
        evidence_root, admission_root, trust_id, date, signer, from, until);
}

void replace_bytes(const std::string& root, std::string_view relative, std::string_view bytes) {
    unlink(join_path(root, relative).c_str());
    plant(root, relative, bytes);
}

void expect_rejected(EvidenceError expected, inferedge::CapabilityResult result,
                     const std::string& evidence_root, const std::string& admission_root) {
    EXPECT_EQ(result.status(), expected) << static_cast<int>(result.status());
    EXPECT_FALSE(result.has_capability());
    EXPECT_EQ(slurp(join_path(evidence_root, "provenance/item-a/keep.bin")), "keep-unrelated");
    EXPECT_EQ(slurp(join_path(admission_root, "foreign-keep.txt")), "foreign");
    EXPECT_TRUE(path_exists(join_path(evidence_root, "lock/evidence.lock")));
}

TEST(EvidenceStore, GenerationOneAuthenticationSucceeds) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    const std::string current_before = slurp(join_path(trust.path(), "current"));
    EXPECT_EQ(current_before, fixture.chain.trust_bytes);
    EXPECT_EQ(current_before, slurp(join_path(admission.path(), "trust/trust-trust-gen1.record")));
    EXPECT_NE(fixture.chain.binding.find("record_kind=transition_operation_binding"),
              std::string::npos);
    EXPECT_NE(fixture.chain.binding.find("record_sha256="), std::string::npos);
    EXPECT_EQ(fixture.chain.trust_bytes.find("terminal_transition_file_size="), std::string::npos);
    EXPECT_EQ(fixture.chain.trust_bytes.find("terminal_transition_file_sha256="),
              std::string::npos);
    const std::string binding_size =
        "binding_record_size=" + std::to_string(fixture.chain.binding.size());
    const std::string binding_digest = "binding_record_sha256=" + digest_hex(fixture.chain.binding);
    EXPECT_NE(fixture.chain.trust_bytes.find(binding_size), std::string::npos);
    EXPECT_NE(fixture.chain.trust_bytes.find(binding_digest), std::string::npos);
    EXPECT_NE(slurp(join_path(trust.path(), "genesis/transition.record")).find(binding_digest),
              std::string::npos);
    inferedge::CapabilityResult result =
        authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture);
    ASSERT_EQ(result.status(), EvidenceError::kOk) << static_cast<int>(result.status());
    EXPECT_TRUE(result.has_capability());
    inferedge::AuthenticatedCapability capability = result.release_capability();
    EXPECT_TRUE(capability.holds_authority());
    EXPECT_FALSE(result.has_capability());
    EXPECT_EQ(capability.operation(), "stale-lock");
    EXPECT_EQ(capability.canonical_authorization_path(), fixture.path);
    EXPECT_EQ(capability.target_id(), fixture.target);
    EXPECT_EQ(capability.terminal_trust_state_id(), kTrustId);
    EXPECT_EQ(capability.terminal_effective_date(), kGenerationDate);
    EXPECT_EQ(capability.signer_key_id(), kAdminId);
    EXPECT_EQ(capability.required_role(), "trust_admin");
    EXPECT_EQ(capability.revocation_decision(), "unrevoked");
    EXPECT_EQ(capability.signature_verification_result(), "verified");
    EXPECT_EQ(capability.key_valid_from(), kKeyFrom);
    EXPECT_EQ(capability.key_valid_until(), kKeyUntil);
    EXPECT_EQ(capability.evidence_root(), evidence.path());
    EXPECT_EQ(capability.admission_root(), admission.path());
    EXPECT_GT(capability.evidence_device() + capability.evidence_inode(), 0u);
    EXPECT_GT(capability.admission_device() + capability.admission_inode(), 0u);
    EXPECT_EQ(capability.authorization_bytes(), fixture.authorization);
    EXPECT_EQ(slurp(join_path(trust.path(), "current")), current_before);
    EXPECT_EQ(slurp(join_path(evidence.path(), "provenance/item-a/keep.bin")), "keep-unrelated");
}

TEST(EvidenceStore, AcceptedStaleLockRecoveryPreservesUnrelatedBytes) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    TempRoot foreign;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok() && foreign.ok());
    plant(foreign.path(), "outside.bin", "outside");
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    inferedge::CapabilityResult result =
        authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture);
    ASSERT_EQ(result.status(), EvidenceError::kOk) << static_cast<int>(result.status());
    inferedge::AuthenticatedCapability capability = result.release_capability();
    ManualClock clock;
    clock.date = kGenerationDate;
    EvidenceLock recovered;
    bool acquired = false;
    EXPECT_EQ(recover_stale_lock(evidence.path(), admission.path(), kGenerationDate, kTrustId,
                                 kAdminId, fixture.authorization, &capability, "run-new",
                                 "host-new", &clock, recovered, acquired),
              EvidenceError::kOk);
    EXPECT_TRUE(acquired);
    EXPECT_TRUE(recovered.held());
    const std::string replacement = slurp(join_path(evidence.path(), "lock/evidence.lock"));
    EXPECT_NE(replacement.find("owner_run_id=run-new"), std::string::npos);
    EXPECT_EQ(replacement.find("owner_run_id=run-old"), std::string::npos);
    EXPECT_EQ(slurp(join_path(evidence.path(), "provenance/item-a/keep.bin")), "keep-unrelated");
    EXPECT_EQ(slurp(join_path(foreign.path(), "outside.bin")), "outside");
    EXPECT_EQ(recovered.release(), EvidenceError::kOk);
}

TEST(EvidenceStore, AcceptedReservationCleanupRetriesAndPreservesUnrelatedObjects) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    ChainMaterial chain;
    plant_generation_one(trust.path(), admission.path(), evidence.path(), chain);
    ManualClock clock;
    clock.date = kGenerationDate;
    EvidenceLock lock = hold_lock(evidence.path(), clock);
    const std::string payload = "file-a";
    plant(evidence.path(), "partial/run-a/a.bin", payload);
    plant(evidence.path(), "partial/run-b/b.bin", "file-b");
    const ReservationEntry staged = entry_for("partial/run-a/a.bin", "provenance/item-a/a.bin",
                                              payload, EvidenceObjectClass::kPublishedBlob);
    EvidenceReservation reservation;
    ASSERT_EQ(serialize_evidence_reservation("run-a", {staged}, reservation), EvidenceError::kOk);
    plant(evidence.path(), "lock/reservation-batch-a.record", reservation.bytes);
    EvidenceReservation unrelated;
    const ReservationEntry other = entry_for("partial/run-b/b.bin", "provenance/item-b/b.bin",
                                             "file-b", EvidenceObjectClass::kPublishedBlob);
    ASSERT_EQ(serialize_evidence_reservation("run-b", {other}, unrelated), EvidenceError::kOk);
    plant(evidence.path(), "lock/reservation-batch-b.record", unrelated.bytes);
    ReservationDeletionList list;
    list.list_id = "list-batch-a";
    list.reservation_id = "batch-a";
    list.owner_run_id = "run-a";
    DeletionListEntry reservation_line;
    reservation_line.relative_path = "lock/reservation-batch-a.record";
    reservation_line.deletion_entry_size = reservation.bytes.size();
    reservation_line.deletion_entry_sha256 = digest_hex(reservation.bytes);
    reservation_line.object_class = CleanupObjectClass::kEvidenceReservation;
    DeletionListEntry partial_line;
    partial_line.relative_path = "partial/run-a/a.bin";
    partial_line.deletion_entry_size = payload.size();
    partial_line.deletion_entry_sha256 = digest_hex(payload);
    partial_line.object_class = CleanupObjectClass::kPartialStaging;
    list.entries = {reservation_line, partial_line};
    list.entry_count = 2;
    ASSERT_EQ(serialize_reservation_deletion_list(list, list.bytes), EvidenceError::kOk);
    plant(admission.path(), "deletion/reservation-batch-a.files.record", list.bytes);
    ReservationDeletionAuthorization authorization;
    authorization.authorization_id = "auth-delete-gen1";
    authorization.target_id = "batch-a";
    authorization.reservation_path = "lock/reservation-batch-a.record";
    authorization.target_file_size = reservation.bytes.size();
    authorization.target_file_sha256 = digest_hex(reservation.bytes);
    authorization.owner_run_id = "run-a";
    authorization.deletion_list_id = "list-batch-a";
    authorization.deletion_list_path = "admission/deletion/reservation-batch-a.files.record";
    authorization.deletion_list_file_size = list.bytes.size();
    authorization.deletion_list_file_sha256 = digest_hex(list.bytes);
    authorization.authorization_date = kGenerationDate;
    ASSERT_EQ(serialize_reservation_deletion(authorization, authorization.bytes),
              EvidenceError::kOk);
    const std::string path = "deletion/deletion-auth-delete-gen1.record";
    plant(admission.path(), path, authorization.bytes);
    plant_object_signature(evidence.path(), admission.path(), chain, authorization.bytes,
                           "deletion-signature-auth-delete-gen1",
                           "deletion-sigblob-auth-delete-gen1",
                           "deletion-sigindex-auth-delete-gen1", "deletion-auth-delete-gen1");
    inferedge::CapabilityResult result = inferedge::AuthenticatedCapability::authenticate(
        "reservation-deletion", authorization.bytes, path, "batch-a", trust.path(), evidence.path(),
        admission.path(), kTrustId, kGenerationDate, kAdminId, kKeyFrom, kKeyUntil);
    ASSERT_EQ(result.status(), EvidenceError::kOk) << static_cast<int>(result.status());
    inferedge::AuthenticatedCapability capability = result.release_capability();
    EXPECT_EQ(cleanup_reservation(lock, admission.path(), kGenerationDate, kTrustId, kAdminId,
                                  authorization.bytes, &capability, list.bytes, &clock),
              EvidenceError::kOk);
    EXPECT_FALSE(path_exists(join_path(evidence.path(), "lock/reservation-batch-a.record")));
    EXPECT_FALSE(path_exists(join_path(evidence.path(), "partial/run-a/a.bin")));
    EXPECT_EQ(slurp(join_path(evidence.path(), "partial/run-b/b.bin")), "file-b");
    EXPECT_TRUE(path_exists(join_path(evidence.path(), "lock/reservation-batch-b.record")));
    EXPECT_EQ(slurp(join_path(evidence.path(), "provenance/item-a/keep.bin")), "keep-unrelated");
    EXPECT_EQ(cleanup_reservation(lock, admission.path(), kGenerationDate, kTrustId, kAdminId,
                                  authorization.bytes, &capability, list.bytes, &clock),
              EvidenceError::kOk);
    EXPECT_EQ(slurp(join_path(evidence.path(), "partial/run-b/b.bin")), "file-b");
    EXPECT_EQ(lock.release(), EvidenceError::kOk);
}

TEST(EvidenceStore, AcceptedQuarantineDeletionPreservesUnrelatedQuarantine) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    ChainMaterial chain;
    plant_generation_one(trust.path(), admission.path(), evidence.path(), chain);
    ManualClock clock;
    clock.date = "2026-01-15";
    EvidenceLock lock = hold_lock(evidence.path(), clock);
    ASSERT_EQ(create_quarantine(lock, "qrun-a", "9-c6", "checksum_rejected", "none", {}, &clock),
              EvidenceError::kOk);
    ASSERT_EQ(create_quarantine(lock, "qrun-b", "9-c6", "checksum_rejected", "none", {}, &clock),
              EvidenceError::kOk);
    const std::string metadata_path = "quarantine/qrun-a/quarantine.record";
    const std::string metadata = slurp(join_path(evidence.path(), metadata_path));
    ASSERT_FALSE(metadata.empty());
    QuarantineDeletionAuthorization authorization;
    authorization.authorization_id = "auth-quarantine-gen1";
    authorization.target_id = "qrun-a";
    authorization.target_file_size = metadata.size();
    authorization.target_file_sha256 = digest_hex(metadata);
    authorization.authorization_date = kGenerationDate;
    ASSERT_EQ(serialize_quarantine_deletion(authorization, authorization.bytes),
              EvidenceError::kOk);
    const std::string path = "deletion/deletion-auth-quarantine-gen1.record";
    plant(admission.path(), path, authorization.bytes);
    plant_object_signature(
        evidence.path(), admission.path(), chain, authorization.bytes,
        "deletion-signature-auth-quarantine-gen1", "deletion-sigblob-auth-quarantine-gen1",
        "deletion-sigindex-auth-quarantine-gen1", "deletion-auth-quarantine-gen1");
    inferedge::CapabilityResult result = inferedge::AuthenticatedCapability::authenticate(
        "quarantine-deletion", authorization.bytes, path, "qrun-a", trust.path(), evidence.path(),
        admission.path(), kTrustId, kGenerationDate, kAdminId, kKeyFrom, kKeyUntil);
    ASSERT_EQ(result.status(), EvidenceError::kOk) << static_cast<int>(result.status());
    inferedge::AuthenticatedCapability capability = result.release_capability();
    clock.date = kGenerationDate;
    EXPECT_EQ(delete_quarantine(lock, admission.path(), kGenerationDate, kTrustId, kAdminId,
                                authorization.bytes, &capability, &clock),
              EvidenceError::kOk);
    EXPECT_FALSE(path_exists(join_path(evidence.path(), metadata_path)));
    EXPECT_TRUE(path_exists(join_path(evidence.path(), "quarantine/qrun-b/quarantine.record")));
    EXPECT_EQ(slurp(join_path(evidence.path(), "provenance/item-a/keep.bin")), "keep-unrelated");
    EXPECT_EQ(lock.release(), EvidenceError::kOk);
}

TEST(EvidenceStore, SignatureObjectNegativesMintNothing) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    const std::string blob =
        "benchmarks/generative_phase9c/signatures/stalelock-sigblob-auth-stale-gen1.bin";
    const std::string index = "signature-indexes/stalelock-sigindex-auth-stale-gen1.index";
    const std::string record = "signature-records/stalelock-signature-auth-stale-gen1.record";
    const std::string blob_bytes = slurp(join_path(evidence.path(), blob));
    const std::string index_bytes = slurp(join_path(admission.path(), index));
    const std::string record_bytes = slurp(join_path(admission.path(), record));
    auto restore = [&] {
        replace_bytes(evidence.path(), blob, blob_bytes);
        replace_bytes(admission.path(), index, index_bytes);
        replace_bytes(admission.path(), record, record_bytes);
    };
    std::string flipped = blob_bytes;
    flipped[0] = static_cast<char>(flipped[0] ^ 0x01);
    replace_bytes(evidence.path(), blob, flipped);
    expect_rejected(EvidenceError::kDigest,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
                    evidence.path(), admission.path());
    std::string short_blob = blob_bytes.substr(0, 63);
    replace_bytes(evidence.path(), blob, short_blob);
    expect_rejected(EvidenceError::kSize,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
                    evidence.path(), admission.path());
    replace_bytes(evidence.path(), blob, blob_bytes + "x");
    expect_rejected(EvidenceError::kSize,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
                    evidence.path(), admission.path());
    unlink(join_path(evidence.path(), blob).c_str());
    expect_rejected(EvidenceError::kUnavailable,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
                    evidence.path(), admission.path());
    restore();
    unlink(join_path(admission.path(), index).c_str());
    expect_rejected(EvidenceError::kUnavailable,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
                    evidence.path(), admission.path());
    restore();
    unlink(join_path(admission.path(), record).c_str());
    expect_rejected(EvidenceError::kUnavailable,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
                    evidence.path(), admission.path());
    restore();
    const std::string blob_full = join_path(evidence.path(), blob);
    unlink(blob_full.c_str());
    ASSERT_EQ(symlink("stalelock-sigblob-auth-stale-gen1.bin", blob_full.c_str()), 0);
    expect_rejected(EvidenceError::kSymlink,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
                    evidence.path(), admission.path());
    restore();
    const std::string index_full = join_path(admission.path(), index);
    unlink(index_full.c_str());
    ASSERT_EQ(symlink("stalelock-sigindex-auth-stale-gen1.index", index_full.c_str()), 0);
    expect_rejected(EvidenceError::kSymlink,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
                    evidence.path(), admission.path());
    restore();
    const std::string record_full = join_path(admission.path(), record);
    unlink(record_full.c_str());
    ASSERT_EQ(symlink("stalelock-signature-auth-stale-gen1.record", record_full.c_str()), 0);
    expect_rejected(EvidenceError::kSymlink,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
                    evidence.path(), admission.path());
    restore();
    const std::string link_name = join_path(evidence.path(), blob + ".link");
    ASSERT_EQ(link(blob_full.c_str(), link_name.c_str()), 0);
    expect_rejected(EvidenceError::kLink,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
                    evidence.path(), admission.path());
    unlink(link_name.c_str());
    EXPECT_EQ(chmod(blob_full.c_str(), 0), 0);
    expect_rejected(EvidenceError::kUnavailable,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
                    evidence.path(), admission.path());
    EXPECT_EQ(chmod(blob_full.c_str(), 0600), 0);
    std::string wrong_kind = record_bytes;
    const std::string kind = "signed_object_kind=structured_file";
    ASSERT_NE(wrong_kind.find(kind), std::string::npos);
    wrong_kind.replace(wrong_kind.find(kind), kind.size(), "signed_object_kind=not_structured1");
    const std::size_t kind_checksum = wrong_kind.rfind("\nrecord_sha256=");
    wrong_kind = seal_record(wrong_kind.substr(0, kind_checksum + 1));
    replace_bytes(admission.path(), record, wrong_kind);
    expect_rejected(EvidenceError::kSchema,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
                    evidence.path(), admission.path());
    replace_bytes(admission.path(), record, record_bytes);
    std::string wrong_algorithm = record_bytes;
    const std::string algorithm = "algorithm=ed25519";
    wrong_algorithm.replace(wrong_algorithm.find(algorithm), algorithm.size(), "algorithm=other01");
    const std::size_t algorithm_checksum = wrong_algorithm.rfind("\nrecord_sha256=");
    wrong_algorithm = seal_record(wrong_algorithm.substr(0, algorithm_checksum + 1));
    replace_bytes(admission.path(), record, wrong_algorithm);
    expect_rejected(EvidenceError::kSchema,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
                    evidence.path(), admission.path());
    replace_bytes(admission.path(), record, record_bytes);
    std::string wrong_digest = record_bytes;
    const std::size_t digest_at = wrong_digest.find("signed_object_sha256=");
    ASSERT_NE(digest_at, std::string::npos);
    const std::size_t digest_hex_at = digest_at + std::strlen("signed_object_sha256=");
    wrong_digest[digest_hex_at] = wrong_digest[digest_hex_at] == '0' ? '1' : '0';
    const std::size_t digest_checksum = wrong_digest.rfind("\nrecord_sha256=");
    wrong_digest = seal_record(wrong_digest.substr(0, digest_checksum + 1));
    replace_bytes(admission.path(), record, wrong_digest);
    expect_rejected(EvidenceError::kDigest,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
                    evidence.path(), admission.path());
}

TEST(EvidenceStore, AuthorizationBindingNegativesMintNothing) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    std::string test_only = fixture.authorization;
    const std::string marker = "test_only=no";
    ASSERT_NE(test_only.find(marker), std::string::npos);
    test_only.replace(test_only.find(marker), marker.size(), "test_only=yes");
    const std::size_t checksum = test_only.rfind("\nrecord_sha256=");
    test_only = seal_record(test_only.substr(0, checksum + 1));
    replace_bytes(admission.path(), fixture.path, test_only);
    expect_rejected(EvidenceError::kSchema,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture,
                                       kTrustId, kGenerationDate, kAdminId, kKeyFrom, kKeyUntil, {},
                                       {}, test_only),
                    evidence.path(), admission.path());
    replace_bytes(admission.path(), fixture.path, fixture.authorization);
    std::string altered = fixture.authorization;
    altered[20] = static_cast<char>(altered[20] ^ 0x01);
    expect_rejected(EvidenceError::kChecksum,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture,
                                       kTrustId, kGenerationDate, kAdminId, kKeyFrom, kKeyUntil, {},
                                       {}, altered),
                    evidence.path(), admission.path());
    expect_rejected(EvidenceError::kAuthorization,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture,
                                       kTrustId, kGenerationDate, kAdminId, kKeyFrom, kKeyUntil,
                                       "lock/not-the-record.record"),
                    evidence.path(), admission.path());
    expect_rejected(EvidenceError::kAuthorization,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture,
                                       kTrustId, kGenerationDate, kAdminId, kKeyFrom, kKeyUntil, {},
                                       "lock/recovery.lock"),
                    evidence.path(), admission.path());
    expect_rejected(
        EvidenceError::kAuthorization,
        authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture, "trust-other"),
        evidence.path(), admission.path());
    expect_rejected(EvidenceError::kAuthorization,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture,
                                       kTrustId, "2026-03-30"),
                    evidence.path(), admission.path());
    expect_rejected(EvidenceError::kAuthorization,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture,
                                       kTrustId, kGenerationDate, "other-gen1"),
                    evidence.path(), admission.path());
    expect_rejected(EvidenceError::kAuthorization,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture,
                                       kTrustId, kGenerationDate, kAdminId, "2021-01-01",
                                       kKeyUntil),
                    evidence.path(), admission.path());
    expect_rejected(EvidenceError::kAuthorization,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture,
                                       kTrustId, kGenerationDate, kAdminId, kKeyFrom, kKeyUntil,
                                       "../lock/stale.record"),
                    evidence.path(), admission.path());
    expect_rejected(EvidenceError::kAuthorization,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture,
                                       kTrustId, kGenerationDate, kAdminId, kKeyFrom, kKeyUntil,
                                       "lock//stale-auth-stale-gen1.record"),
                    evidence.path(), admission.path());
    const std::string auth_full = join_path(admission.path(), fixture.path);
    const std::string saved = slurp(auth_full);
    unlink(auth_full.c_str());
    ASSERT_EQ(symlink("stale-auth-stale-gen1.record", auth_full.c_str()), 0);
    expect_rejected(EvidenceError::kSymlink,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
                    evidence.path(), admission.path());
    unlink(auth_full.c_str());
    plant(admission.path(), fixture.path, saved);
    inferedge::CapabilityResult wrong_operation = inferedge::AuthenticatedCapability::authenticate(
        "reservation-deletion", fixture.authorization, fixture.path, fixture.target, trust.path(),
        evidence.path(), admission.path(), kTrustId, kGenerationDate, kAdminId, kKeyFrom,
        kKeyUntil);
    EXPECT_EQ(wrong_operation.status(), EvidenceError::kSchema);
    EXPECT_FALSE(wrong_operation.has_capability());
}

TEST(EvidenceStore, RegistryAndKeyNegativesMintNothing) {
    auto check = [](const ChainOptions& options, EvidenceError expected) {
        TempRoot trust;
        TempRoot admission;
        TempRoot evidence;
        ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
        StaleFixture fixture;
        plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture, options);
        expect_rejected(
            expected, authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
            evidence.path(), admission.path());
    };
    ChainOptions role;
    role.role = "authority_admin";
    check(role, EvidenceError::kAuthorization);
    ChainOptions revoked;
    revoked.revoked = "yes";
    check(revoked, EvidenceError::kAuthorization);
    ChainOptions early;
    early.valid_from = "2026-04-01";
    check(early, EvidenceError::kDate);
    ChainOptions expired;
    expired.valid_until = "2026-03-30";
    check(expired, EvidenceError::kDate);
    ChainOptions duplicate_id;
    duplicate_id.duplicate_id = true;
    check(duplicate_id, EvidenceError::kDuplicate);
    ChainOptions duplicate_digest;
    duplicate_digest.duplicate_digest = true;
    check(duplicate_digest, EvidenceError::kDuplicate);
    ChainOptions bad_registry;
    bad_registry.flip_registry_signature = true;
    check(bad_registry, EvidenceError::kAuthorization);
    ChainOptions bad_bootstrap;
    bad_bootstrap.flip_genesis_signature = true;
    check(bad_bootstrap, EvidenceError::kAuthorization);
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    const std::string envelope_path =
        "signature-records/registry-envelope-signature-registry-gen1.record";
    std::string envelope = slurp(join_path(admission.path(), envelope_path));
    envelope[10] = static_cast<char>(envelope[10] ^ 0x01);
    replace_bytes(admission.path(), envelope_path, envelope);
    expect_rejected(EvidenceError::kChecksum,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
                    evidence.path(), admission.path());
}

TEST(EvidenceStore, RootAndDescriptorNegativesMintNothing) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    std::string changed = fixture.chain.trust_bytes;
    const std::string host = "host_id=" + std::string(kHostId);
    ASSERT_NE(changed.find(host), std::string::npos);
    changed.replace(changed.find(host), host.size(), "host_id=host-other");
    const std::size_t checksum = changed.rfind("\nrecord_sha256=");
    changed = seal_record(changed.substr(0, checksum + 1));
    replace_bytes(trust.path(), "current", changed);
    expect_rejected(EvidenceError::kRace,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
                    evidence.path(), admission.path());
    replace_bytes(trust.path(), "current", fixture.chain.trust_bytes);
    unlink(join_path(trust.path(), "current").c_str());
    plant(trust.path(), "current", "");
    expect_rejected(EvidenceError::kSize,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
                    evidence.path(), admission.path());
    replace_bytes(trust.path(), "current", fixture.chain.trust_bytes);
    const std::string genesis = join_path(trust.path(), "genesis");
    const std::string moved = genesis + "-moved";
    ASSERT_EQ(rename(genesis.c_str(), moved.c_str()), 0);
    ASSERT_EQ(symlink(moved.c_str(), genesis.c_str()), 0);
    expect_rejected(EvidenceError::kSymlink,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
                    evidence.path(), admission.path());
    unlink(genesis.c_str());
    ASSERT_EQ(rename(moved.c_str(), genesis.c_str()), 0);
    inferedge::CapabilityResult accepted =
        authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture);
    ASSERT_EQ(accepted.status(), EvidenceError::kOk) << static_cast<int>(accepted.status());
    inferedge::AuthenticatedCapability capability = accepted.release_capability();
    const std::string evidence_moved = evidence.path() + "-moved";
    ASSERT_EQ(rename(evidence.path().c_str(), evidence_moved.c_str()), 0);
    ASSERT_EQ(mkdir(evidence.path().c_str(), 0700), 0);
    ManualClock clock;
    clock.date = kGenerationDate;
    EvidenceLock recovered;
    bool acquired = false;
    EXPECT_EQ(recover_stale_lock(evidence.path(), admission.path(), kGenerationDate, kTrustId,
                                 kAdminId, fixture.authorization, &capability, "run-new",
                                 "host-new", &clock, recovered, acquired),
              EvidenceError::kRace);
    EXPECT_FALSE(acquired);
    EXPECT_TRUE(path_exists(evidence_moved + "/lock/evidence.lock"));
    EXPECT_EQ(slurp(evidence_moved + "/provenance/item-a/keep.bin"), "keep-unrelated");
    ASSERT_EQ(rmdir(evidence.path().c_str()), 0);
    ASSERT_EQ(rename(evidence_moved.c_str(), evidence.path().c_str()), 0);
}

std::string record_field_text(const std::string& record, std::string_view key) {
    const std::string needle = std::string(key) + "=";
    const std::size_t at = record.find(needle);
    EXPECT_NE(at, std::string::npos);
    const std::size_t end = record.find('\n', at);
    return record.substr(at + needle.size(), end - (at + needle.size()));
}

std::string reseal_fields(std::string record,
                          const std::vector<std::pair<std::string, std::string>>& fields) {
    const std::size_t checksum = record.rfind("\nrecord_sha256=");
    EXPECT_NE(checksum, std::string::npos);
    std::string prefix = record.substr(0, checksum + 1);
    for (const auto& field : fields) {
        const std::string needle = field.first + "=";
        const std::size_t at = prefix.find(needle);
        EXPECT_NE(at, std::string::npos);
        const std::size_t end = prefix.find('\n', at);
        prefix.replace(at, end - at, needle + field.second);
    }
    return seal_record(prefix);
}

std::string drop_record_field(std::string record, std::string_view key) {
    const std::size_t checksum = record.rfind("\nrecord_sha256=");
    std::string prefix = record.substr(0, checksum + 1);
    const std::string needle = std::string(key) + "=";
    const std::size_t at = prefix.find(needle);
    EXPECT_NE(at, std::string::npos);
    const std::size_t end = prefix.find('\n', at);
    prefix.erase(at, end - at + 1);
    return seal_record(prefix);
}

void install_transition_signature(const std::string& trust_root, std::string_view signature) {
    replace_bytes(trust_root, "genesis/signature.bin", signature);
    std::string index = reseal_fields(slurp(join_path(trust_root, "genesis/signature.index")),
                                      {{"raw_sha256", digest_hex(signature)}});
    replace_bytes(trust_root, "genesis/signature.index", index);
    std::string verification =
        reseal_fields(slurp(join_path(trust_root, "genesis/verification.record")),
                      {{"signature_sha256", digest_hex(signature)},
                       {"signature_index_sha256", index_self_prefix_digest(index)}});
    replace_bytes(trust_root, "genesis/verification.record", verification);
    const std::string transition =
        reseal_fields(slurp(join_path(trust_root, "genesis/transition.record")),
                      {{"signature_file_size", std::to_string(verification.size())},
                       {"signature_file_sha256", digest_hex(verification)}});
    replace_bytes(trust_root, "genesis/transition.record", transition);
}

void bind_signature_file(const std::string& trust, std::string_view directory) {
    const std::string verification_path = std::string(directory) + "/verification.record";
    const std::string verification = slurp(join_path(trust, verification_path));
    const std::string transition_path = std::string(directory) + "/transition.record";
    const std::string transition =
        reseal_fields(slurp(join_path(trust, transition_path)),
                      {{"signature_file_size", std::to_string(verification.size())},
                       {"signature_file_sha256", digest_hex(verification)}});
    replace_bytes(trust, transition_path, transition);
}

void resign_family(const std::string& trust, std::string_view directory, const TestKey& key) {
    const std::string payload_path = std::string(directory) + "/transition.payload";
    const std::string payload = slurp(join_path(trust, payload_path));
    const std::string signature = signed_message(key, trust_transition_message(payload));
    const std::string signature_path = std::string(directory) + "/signature.bin";
    replace_bytes(trust, signature_path, signature);
    const std::string index_path = std::string(directory) + "/signature.index";
    const std::string index =
        reseal_fields(slurp(join_path(trust, index_path)), {{"raw_sha256", digest_hex(signature)}});
    replace_bytes(trust, index_path, index);
    const std::string verification_path = std::string(directory) + "/verification.record";
    const std::string verification =
        reseal_fields(slurp(join_path(trust, verification_path)),
                      {{"signed_object_size", std::to_string(payload.size())},
                       {"signed_object_sha256", digest_hex(payload)},
                       {"signature_sha256", digest_hex(signature)},
                       {"signature_index_size", std::to_string(index.size())},
                       {"signature_index_sha256", index_self_prefix_digest(index)}});
    replace_bytes(trust, verification_path, verification);
    const std::string transition_path = std::string(directory) + "/transition.record";
    const std::string transition =
        reseal_fields(slurp(join_path(trust, transition_path)),
                      {{"signed_payload_size", std::to_string(payload.size())},
                       {"signed_payload_sha256", digest_hex(payload)},
                       {"signature_file_size", std::to_string(verification.size())},
                       {"signature_file_sha256", digest_hex(verification)}});
    replace_bytes(trust, transition_path, transition);
}

TestKey bootstrap_key() {
    unsigned char seed[32];
    std::memset(seed, 0x42, sizeof(seed));
    TestKey key;
    test_keypair(key, seed);
    return key;
}

void plant_payload_bytes(const std::string& trust_root, std::string_view bytes) {
    replace_bytes(trust_root, "genesis/transition.payload", bytes);
    force_mode(trust_root, "genesis/transition.payload", 0444);
}

struct LaterGeneration {
    std::string trust_id{"trust-gen2"};
    std::string transition_id{"transition-gen2"};
    std::string trust_bytes;
};

LaterGeneration plant_later_generation(
    const std::string& trust_root, const std::string& admission_root, ChainMaterial& chain,
    const char* generation = "2", const char* revocation_generation = "0",
    const char* revocation_id = "none", const char* revocation_digest = nullptr,
    const char* predecessor_transition_id = "genesis-v1", const char* revocation_file_size = "0") {
    const std::string zero(64, '0');
    const std::string digest = revocation_digest == nullptr ? zero : revocation_digest;
    const TestKey bootstrap = bootstrap_key();
    const std::string bootstrap_raw(reinterpret_cast<const char*>(bootstrap.public_key), 32);
    const std::string admin_raw(reinterpret_cast<const char*>(chain.admin.public_key), 32);
    const std::string previous_trust = chain.trust_bytes;
    const std::string previous_transition =
        slurp(join_path(trust_root, "genesis/transition.record"));
    const std::string transition_id = "transition-gen2";
    const std::string trust_id = "trust-gen2";
    const std::string transition_path = "transitions/" + transition_id + "/transition.record";
    const std::string payload_path = "transitions/" + transition_id + "/transition.payload";
    const std::string predecessor_path =
        std::string(predecessor_transition_id) == "genesis-v1"
            ? "genesis/transition.record"
            : "transitions/" + std::string(predecessor_transition_id) + "/transition.record";
    const std::vector<std::pair<std::string, std::string>> copied = {
        {"publication_kind", "ordinary"},
        {"publication_authorization_id", "auth-pub-gen2"},
        {"publication_authorization_path", "admission/authorization/auth-pub-gen2.record"},
        {"publication_authorization_file_size", "11"},
        {"publication_authorization_file_sha256", digest_hex("auth-bytes")},
        {"publication_authorization_preimage_sha256", digest_hex("auth-bytes")},
        {"signer_key_id", kAdminId},
        {"signer_publication_key_size", "32"},
        {"signer_publication_key_sha256", digest_hex(admin_raw)},
        {"signature_family_id", "trust_transition"},
        {"trust_id", trust_id},
        {"trust_generation", generation},
        {"effective_date", kGenerationDate},
        {"terminal_transition_id", transition_id},
        {"terminal_transition_path", transition_path},
        {"predecessor_trust_id", kTrustId},
        {"predecessor_file_size", std::to_string(previous_trust.size())},
        {"predecessor_file_sha256", digest_hex(previous_trust)},
        {"host_id", kHostId},
        {"lineage_kind", "native"},
        {"revocation_generation", revocation_generation},
        {"revocation_id", revocation_id},
        {"revocation_file_size", revocation_file_size},
        {"revocation_file_sha256", digest},
        {"config_id", "none"},
        {"configuration_file_size", "0"},
        {"configuration_file_sha256", zero},
        {"registry_id", record_field_text(previous_trust, "registry_id")},
        {"authority_registry_file_size",
         record_field_text(previous_trust, "authority_registry_file_size")},
        {"authority_registry_file_sha256",
         record_field_text(previous_trust, "authority_registry_file_sha256")},
        {"authority_envelope_id", record_field_text(previous_trust, "authority_envelope_id")},
        {"authority_envelope_file_size",
         record_field_text(previous_trust, "authority_envelope_file_size")},
        {"authority_envelope_file_sha256",
         record_field_text(previous_trust, "authority_envelope_file_sha256")},
        {"bootstrap_key_handle", "bootstrap-key-v1"},
        {"bootstrap_key_size", "32"},
        {"bootstrap_key_sha256", digest_hex(bootstrap_raw)},
        {"admission_manifest_id", "none"},
        {"admission_root_manifest_file_size", "0"},
        {"admission_root_manifest_file_sha256", zero}};
    std::string binding_prefix;
    append_field(binding_prefix, "schema_version", "1");
    append_field(binding_prefix, "record_kind", "transition_operation_binding");
    for (const auto& field : copied) {
        append_field(binding_prefix, field.first, field.second);
    }
    const std::string binding = seal_record(binding_prefix);
    const std::string trust = seal_fields({
        {"schema_version", "1"},
        {"record_kind", "trust_state"},
        {"trust_id", trust_id},
        {"trust_generation", generation},
        {"effective_date", kGenerationDate},
        {"terminal_transition_id", transition_id},
        {"terminal_transition_path", transition_path},
        {"revocation_generation", revocation_generation},
        {"revocation_id", revocation_id},
        {"revocation_file_size", revocation_file_size},
        {"revocation_file_sha256", digest},
        {"config_id", "none"},
        {"configuration_file_size", "0"},
        {"configuration_file_sha256", zero},
        {"registry_id", record_field_text(previous_trust, "registry_id")},
        {"authority_registry_file_size",
         record_field_text(previous_trust, "authority_registry_file_size")},
        {"authority_registry_file_sha256",
         record_field_text(previous_trust, "authority_registry_file_sha256")},
        {"authority_envelope_id", record_field_text(previous_trust, "authority_envelope_id")},
        {"authority_envelope_file_size",
         record_field_text(previous_trust, "authority_envelope_file_size")},
        {"authority_envelope_file_sha256",
         record_field_text(previous_trust, "authority_envelope_file_sha256")},
        {"bootstrap_key_handle", "bootstrap-key-v1"},
        {"bootstrap_key_size", "32"},
        {"bootstrap_key_sha256", digest_hex(bootstrap_raw)},
        {"admission_manifest_id", "none"},
        {"admission_root_manifest_file_size", "0"},
        {"admission_root_manifest_file_sha256", zero},
        {"host_id", kHostId},
        {"publication_kind", "ordinary"},
        {"publication_authorization_id", "auth-pub-gen2"},
        {"publication_authorization_path", "admission/authorization/auth-pub-gen2.record"},
        {"publication_authorization_file_size", "11"},
        {"publication_authorization_file_sha256", digest_hex("auth-bytes")},
        {"publication_authorization_preimage_sha256", digest_hex("auth-bytes")},
        {"signer_key_id", kAdminId},
        {"signer_publication_key_size", "32"},
        {"signer_publication_key_sha256", digest_hex(admin_raw)},
        {"signature_family_id", "trust_transition"},
        {"lineage_kind", "native"},
        {"binding_record_kind", "transition_operation_binding"},
        {"binding_record_size", std::to_string(binding.size())},
        {"binding_record_sha256", digest_hex(binding)},
        {"telemetry_mode", "disabled"},
        {"telemetry_governance_id", "none"},
        {"telemetry_governance_file_size", "0"},
        {"telemetry_governance_file_sha256", zero},
        {"predecessor_trust_id", kTrustId},
        {"predecessor_file_size", std::to_string(previous_trust.size())},
        {"predecessor_file_sha256", digest_hex(previous_trust)},
    });
    const std::string registry_size =
        record_field_text(previous_trust, "authority_registry_file_size");
    const std::string registry_digest =
        record_field_text(previous_trust, "authority_registry_file_sha256");
    const std::vector<std::pair<std::string, std::string>> shared = {
        {"schema_version", "1"},
        {"transition_id", transition_id},
        {"trust_generation", generation},
        {"transition_date", kGenerationDate},
        {"predecessor_transition_id", predecessor_transition_id},
        {"predecessor_transition_path", predecessor_path},
        {"predecessor_transition_file_size", std::to_string(previous_transition.size())},
        {"predecessor_transition_file_sha256", digest_hex(previous_transition)},
        {"predecessor_successor_trust_id", kTrustId},
        {"predecessor_successor_trust_file_size", std::to_string(previous_trust.size())},
        {"predecessor_successor_trust_file_sha256", digest_hex(previous_trust)},
        {"successor_trust_id", trust_id},
        {"successor_trust_path", "admission/trust/trust-" + trust_id + ".record"},
        {"successor_trust_file_size", std::to_string(trust.size())},
        {"successor_trust_file_sha256", digest_hex(trust)},
        {"bootstrap_key_sha256", digest_hex(bootstrap_raw)},
        {"signer_key_id", kAdminId},
        {"registry_id", kRegistryId},
        {"authority_registry_file_size", registry_size},
        {"authority_registry_file_sha256", registry_digest},
        {"publication_kind", "ordinary"},
        {"binding_record_kind", "transition_operation_binding"},
        {"binding_record_size", std::to_string(binding.size())},
        {"binding_record_sha256", digest_hex(binding)}};
    std::vector<std::pair<std::string, std::string>> payload_fields = {
        {"schema_version", "1"}, {"record_kind", "trust_transition_signed_payload"}};
    payload_fields.insert(payload_fields.end(), shared.begin() + 1, shared.end());
    const std::string payload = seal_fields(payload_fields);
    const std::string signature = signed_message(chain.admin, trust_transition_message(payload));
    const std::string blob_id = "transition-sigblob-" + transition_id;
    const std::string index_id = "transition-sigindex-" + transition_id;
    const std::string signature_id = "transition-signature-" + transition_id;
    const std::string storage = "transitions/" + transition_id + "/signature.bin";
    const std::string index = seal_fields({{"schema_version", "1"},
                                           {"record_kind", "blob_index"},
                                           {"blob_id", blob_id},
                                           {"blob_class", "signature_bytes"},
                                           {"raw_size", "64"},
                                           {"raw_sha256", digest_hex(signature)},
                                           {"storage_path", storage},
                                           {"review_note_id", "none"},
                                           {"review_note_file_size", "0"},
                                           {"review_note_file_sha256", zero}});
    const std::string verification =
        seal_fields({{"schema_version", "1"},
                     {"record_kind", "signature_verification"},
                     {"signature_id", signature_id},
                     {"signed_object_kind", "structured_file"},
                     {"signed_object_id", "transition-payload-" + transition_id},
                     {"signed_object_path", "none"},
                     {"signed_object_size", std::to_string(payload.size())},
                     {"signed_object_sha256", digest_hex(payload)},
                     {"signature_blob_id", blob_id},
                     {"signature_index_id", index_id},
                     {"signature_index_path", "none"},
                     {"signature_index_size", std::to_string(index.size())},
                     {"signature_index_sha256", index_self_prefix_digest(index)},
                     {"signature_size", "64"},
                     {"signature_sha256", digest_hex(signature)},
                     {"signer_key_source", "blob"},
                     {"signer_key_id", kAdminId},
                     {"public_key_size", "32"},
                     {"public_key_sha256", digest_hex(admin_raw)},
                     {"key_index_id", "none"},
                     {"key_index_size", "0"},
                     {"key_index_sha256", zero},
                     {"fingerprint_sha256", digest_hex(admin_raw)},
                     {"algorithm", "ed25519"},
                     {"verification_result", "verified"},
                     {"verification_date", kGenerationDate},
                     {"verifier_id", "inferedge-ed25519-v1"}});
    std::vector<std::pair<std::string, std::string>> transition_fields = {
        {"schema_version", "1"}, {"record_kind", "trust_transition"}};
    transition_fields.insert(transition_fields.end(), shared.begin() + 1, shared.end());
    const std::size_t signature_at = 21;
    transition_fields.insert(transition_fields.begin() + signature_at,
                             {{"signed_payload_id", "transition-payload-" + transition_id},
                              {"signed_payload_path", payload_path},
                              {"signed_payload_size", std::to_string(payload.size())},
                              {"signed_payload_sha256", digest_hex(payload)},
                              {"signature_id", signature_id},
                              {"signature_blob_id", blob_id},
                              {"signature_index_id", index_id},
                              {"signature_file_size", std::to_string(verification.size())},
                              {"signature_file_sha256", digest_hex(verification)}});
    const std::string transition = seal_fields(transition_fields);
    plant(admission_root, "trust/trust-" + trust_id + ".record", trust);
    plant(trust_root, payload_path, payload);
    force_mode(trust_root, payload_path, 0444);
    plant(trust_root, transition_path, transition);
    plant(trust_root, storage, signature);
    plant(trust_root, "transitions/" + transition_id + "/signature.index", index);
    plant(trust_root, "transitions/" + transition_id + "/verification.record", verification);
    replace_bytes(trust_root, "current", trust);
    LaterGeneration later;
    later.trust_bytes = trust;
    return later;
}

template <class Mutator> void reject_mutated(Mutator mutate, EvidenceError expected) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    mutate(trust.path(), admission.path(), evidence.path(), fixture);
    expect_rejected(expected,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
                    evidence.path(), admission.path());
}

TEST(EvidenceStore, LaterGenerationPayloadAuthenticationSucceeds) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    const LaterGeneration later =
        plant_later_generation(trust.path(), admission.path(), fixture.chain);
    const std::string current_before = slurp(join_path(trust.path(), "current"));
    EXPECT_EQ(current_before, later.trust_bytes);
    inferedge::CapabilityResult result = inferedge::AuthenticatedCapability::authenticate(
        "stale-lock", fixture.authorization, fixture.path, fixture.target, trust.path(),
        evidence.path(), admission.path(), later.trust_id, kGenerationDate, kAdminId, kKeyFrom,
        kKeyUntil);
    ASSERT_EQ(result.status(), EvidenceError::kOk) << static_cast<int>(result.status());
    inferedge::AuthenticatedCapability capability = result.release_capability();
    EXPECT_TRUE(capability.holds_authority());
    EXPECT_EQ(slurp(join_path(trust.path(), "current")), current_before);
    EXPECT_FALSE(result.has_capability());
}

TEST(EvidenceStore, ExactDomainSeparatedPayloadMessage) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    const std::string payload = slurp(join_path(trust.path(), "genesis/transition.payload"));
    const std::string message = trust_transition_message(payload);
    const std::string literal = "INFEREDGE-ED25519-V1 TRUST-TRANSITION-PAYLOAD 1";
    EXPECT_EQ(literal.size(), 47u);
    EXPECT_EQ(message.size(), 48u + payload.size());
    EXPECT_EQ(message.substr(0, 47), literal);
    EXPECT_EQ(message[47], '\n');
    EXPECT_EQ(message.find('\n'), 47u);
    EXPECT_EQ(message.substr(48), payload);
    EXPECT_EQ(payload.back(), '\n');
    EXPECT_NE(payload.find("\nrecord_sha256="), std::string::npos);
    EXPECT_EQ(payload.substr(payload.size() - 1), "\n");
    const std::string signature = slurp(join_path(trust.path(), "genesis/signature.bin"));
    EXPECT_EQ(signature, signed_message(bootstrap_key(), message));
    const std::string legacy_source = slurp(join_path(trust.path(), "genesis/transition.record"));
    const std::size_t checksum = legacy_source.rfind("\nrecord_sha256=");
    std::string legacy = legacy_source.substr(0, checksum + 1);
    const std::size_t digest_at = legacy.find("signature_file_sha256=") + 22;
    legacy.replace(digest_at, 64, 64, '0');
    EXPECT_NE(signature, signed_message(bootstrap_key(), legacy));
    EXPECT_NE(signature, signed_message(bootstrap_key(), legacy_source));
    EXPECT_NE(signature, signed_message(bootstrap_key(), legacy_source.substr(0, checksum + 1)));
    inferedge::CapabilityResult result =
        authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture);
    EXPECT_EQ(result.status(), EvidenceError::kOk);
}

TEST(EvidenceStore, RejectsLegacyZeroSubstitutedPrefix) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string record = slurp(join_path(trust, "genesis/transition.record"));
            const std::size_t checksum = record.rfind("\nrecord_sha256=");
            std::string legacy = record.substr(0, checksum + 1);
            const std::size_t digest_at = legacy.find("signature_file_sha256=") + 22;
            legacy.replace(digest_at, 64, 64, '0');
            install_transition_signature(trust, signed_message(bootstrap_key(), legacy));
        },
        EvidenceError::kAuthorization);
}

TEST(EvidenceStore, RejectsTransitionRecordPrefixSignature) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string record = slurp(join_path(trust, "genesis/transition.record"));
            const std::size_t checksum = record.rfind("\nrecord_sha256=");
            install_transition_signature(
                trust, signed_message(bootstrap_key(), record.substr(0, checksum + 1)));
        },
        EvidenceError::kAuthorization);
}

TEST(EvidenceStore, RejectsSignatureOverFinishedTransitionRecord) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string record = slurp(join_path(trust, "genesis/transition.record"));
            install_transition_signature(trust, signed_message(bootstrap_key(), record));
        },
        EvidenceError::kAuthorization);
}

TEST(EvidenceStore, RejectsMissingTransitionPayload) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            ASSERT_EQ(unlink(join_path(trust, "genesis/transition.payload").c_str()), 0);
        },
        EvidenceError::kUnavailable);
}

TEST(EvidenceStore, RejectsTransitionPayloadSymlink) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string path = join_path(trust, "genesis/transition.payload");
            ASSERT_EQ(unlink(path.c_str()), 0);
            ASSERT_EQ(symlink("transition.record", path.c_str()), 0);
        },
        EvidenceError::kSymlink);
}

TEST(EvidenceStore, RejectsTransitionPayloadHardLink) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string path = join_path(trust, "genesis/transition.payload");
            ASSERT_EQ(link(path.c_str(), (path + ".link").c_str()), 0);
        },
        EvidenceError::kLink);
}

TEST(EvidenceStore, RejectsTransitionPayloadWrongMode) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            force_mode(trust, "genesis/transition.payload", 0644);
        },
        EvidenceError::kForbidden);
}

TEST(EvidenceStore, RejectsTransitionPayloadOversized) {
    reject_mutated([](const std::string& trust, const std::string&, const std::string&,
                      const StaleFixture&) { plant_payload_bytes(trust, std::string(16385, 'a')); },
                   EvidenceError::kCap);
}

TEST(EvidenceStore, RejectsTransitionPayloadTruncated) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            std::string payload = slurp(join_path(trust, "genesis/transition.payload"));
            payload.resize(payload.size() - 5);
            plant_payload_bytes(trust, payload);
        },
        EvidenceError::kSize);
}

TEST(EvidenceStore, RejectsTransitionPayloadMalformedSchema) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string payload =
                reseal_fields(slurp(join_path(trust, "genesis/transition.payload")),
                              {{"record_kind", "trust_transition"}});
            plant_payload_bytes(trust, payload);
        },
        EvidenceError::kSchema);
}

TEST(EvidenceStore, RejectsDuplicateTransitionPayloadKey) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            std::string payload = slurp(join_path(trust, "genesis/transition.payload"));
            const std::size_t checksum = payload.rfind("\nrecord_sha256=");
            std::string prefix = payload.substr(0, checksum + 1);
            prefix += "schema_version=1\n";
            plant_payload_bytes(trust, seal_record(prefix));
        },
        EvidenceError::kSchema);
}

TEST(EvidenceStore, RejectsUnknownTransitionPayloadKey) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            std::string payload = slurp(join_path(trust, "genesis/transition.payload"));
            const std::size_t checksum = payload.rfind("\nrecord_sha256=");
            std::string prefix = payload.substr(0, checksum + 1);
            const std::string needle = "transition_date=";
            const std::size_t at = prefix.find(needle);
            prefix.replace(at, needle.size(), "transition_when=");
            plant_payload_bytes(trust, seal_record(prefix));
        },
        EvidenceError::kSchema);
}

TEST(EvidenceStore, RejectsWrongTransitionPayloadKeyOrder) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            std::string payload = slurp(join_path(trust, "genesis/transition.payload"));
            const std::size_t checksum = payload.rfind("\nrecord_sha256=");
            std::string prefix = payload.substr(0, checksum + 1);
            const std::size_t first_end = prefix.find('\n');
            const std::size_t second_end = prefix.find('\n', first_end + 1);
            const std::string swapped = prefix.substr(first_end + 1, second_end - first_end) +
                                        prefix.substr(0, first_end + 1) +
                                        prefix.substr(second_end + 1);
            plant_payload_bytes(trust, seal_record(swapped));
        },
        EvidenceError::kSchema);
}

TEST(EvidenceStore, RejectsTransitionPayloadChecksumMismatch) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            std::string payload = slurp(join_path(trust, "genesis/transition.payload"));
            payload[payload.size() - 2] = payload[payload.size() - 2] == '0' ? '1' : '0';
            plant_payload_bytes(trust, payload);
        },
        EvidenceError::kChecksum);
}

TEST(EvidenceStore, RejectsTransitionPayloadByteMutation) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            std::string payload = slurp(join_path(trust, "genesis/transition.payload"));
            const std::size_t at = payload.find("transition_date=");
            payload[at + 16] = '1';
            payload = seal_record(payload.substr(0, payload.rfind("\nrecord_sha256=") + 1));
            plant_payload_bytes(trust, payload);
            std::string verification =
                reseal_fields(slurp(join_path(trust, "genesis/verification.record")),
                              {{"signed_object_sha256", digest_hex(payload)}});
            replace_bytes(trust, "genesis/verification.record", verification);
            const std::string transition =
                reseal_fields(slurp(join_path(trust, "genesis/transition.record")),
                              {{"transition_date", "1026-03-31"},
                               {"signed_payload_sha256", digest_hex(payload)},
                               {"signature_file_size", std::to_string(verification.size())},
                               {"signature_file_sha256", digest_hex(verification)}});
            replace_bytes(trust, "genesis/transition.record", transition);
        },
        EvidenceError::kAuthorization);
}

TEST(EvidenceStore, RejectsSignedPayloadIdMismatch) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string transition =
                reseal_fields(slurp(join_path(trust, "genesis/transition.record")),
                              {{"signed_payload_id", "transition-payload-other-v1"}});
            replace_bytes(trust, "genesis/transition.record", transition);
        },
        EvidenceError::kSchema);
}

TEST(EvidenceStore, RejectsSignedPayloadPathMismatch) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string transition =
                reseal_fields(slurp(join_path(trust, "genesis/transition.record")),
                              {{"signed_payload_path", "genesis/other.payload"}});
            replace_bytes(trust, "genesis/transition.record", transition);
        },
        EvidenceError::kPath);
}

TEST(EvidenceStore, RejectsSignedPayloadSizeMismatch) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string transition =
                reseal_fields(slurp(join_path(trust, "genesis/transition.record")),
                              {{"signed_payload_size", "1"}});
            replace_bytes(trust, "genesis/transition.record", transition);
        },
        EvidenceError::kSize);
}

TEST(EvidenceStore, RejectsSignedPayloadDigestMismatch) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string other = digest_hex("different-payload");
            const std::string transition =
                reseal_fields(slurp(join_path(trust, "genesis/transition.record")),
                              {{"signed_payload_sha256", other}});
            replace_bytes(trust, "genesis/transition.record", transition);
        },
        EvidenceError::kDigest);
}

TEST(EvidenceStore, RejectsTransitionPayloadSemanticMismatch) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string transition =
                reseal_fields(slurp(join_path(trust, "genesis/transition.record")),
                              {{"transition_date", "2026-04-02"}});
            replace_bytes(trust, "genesis/transition.record", transition);
        },
        EvidenceError::kState);
}

TEST(EvidenceStore, RejectsMissingSignedPayloadField) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string transition = drop_record_field(
                slurp(join_path(trust, "genesis/transition.record")), "signed_payload_id");
            replace_bytes(trust, "genesis/transition.record", transition);
        },
        EvidenceError::kSchema);
}

TEST(EvidenceStore, RejectsZeroSignatureFileDigest) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string transition =
                reseal_fields(slurp(join_path(trust, "genesis/transition.record")),
                              {{"signature_file_sha256", std::string(64, '0')}});
            replace_bytes(trust, "genesis/transition.record", transition);
        },
        EvidenceError::kDigest);
}

TEST(EvidenceStore, RejectsVerificationRecordSizeMismatch) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string transition =
                reseal_fields(slurp(join_path(trust, "genesis/transition.record")),
                              {{"signature_file_size", "64"}});
            replace_bytes(trust, "genesis/transition.record", transition);
        },
        EvidenceError::kSize);
}

TEST(EvidenceStore, RejectsVerificationRecordDigestMismatch) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string transition =
                reseal_fields(slurp(join_path(trust, "genesis/transition.record")),
                              {{"signature_file_sha256", digest_hex("other-verification")}});
            replace_bytes(trust, "genesis/transition.record", transition);
        },
        EvidenceError::kDigest);
}

TEST(EvidenceStore, RejectsVerificationNamingTransitionRecord) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            std::string verification =
                reseal_fields(slurp(join_path(trust, "genesis/verification.record")),
                              {{"signed_object_id", "genesis-v1"}});
            replace_bytes(trust, "genesis/verification.record", verification);
            const std::string transition =
                reseal_fields(slurp(join_path(trust, "genesis/transition.record")),
                              {{"signature_file_size", std::to_string(verification.size())},
                               {"signature_file_sha256", digest_hex(verification)}});
            replace_bytes(trust, "genesis/transition.record", transition);
        },
        EvidenceError::kState);
}

TEST(EvidenceStore, RejectsVerificationNamingAnotherPayload) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            std::string verification =
                reseal_fields(slurp(join_path(trust, "genesis/verification.record")),
                              {{"signed_object_id", "transition-payload-other-v1"}});
            replace_bytes(trust, "genesis/verification.record", verification);
            const std::string transition =
                reseal_fields(slurp(join_path(trust, "genesis/transition.record")),
                              {{"signature_file_size", std::to_string(verification.size())},
                               {"signature_file_sha256", digest_hex(verification)}});
            replace_bytes(trust, "genesis/transition.record", transition);
        },
        EvidenceError::kState);
}

TEST(EvidenceStore, RejectsFullFileIndexDigest) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string index = slurp(join_path(trust, "genesis/signature.index"));
            std::string verification =
                reseal_fields(slurp(join_path(trust, "genesis/verification.record")),
                              {{"signature_index_sha256", digest_hex(index)}});
            replace_bytes(trust, "genesis/verification.record", verification);
            const std::string transition =
                reseal_fields(slurp(join_path(trust, "genesis/transition.record")),
                              {{"signature_file_size", std::to_string(verification.size())},
                               {"signature_file_sha256", digest_hex(verification)}});
            replace_bytes(trust, "genesis/transition.record", transition);
        },
        EvidenceError::kDigest);
}

TEST(EvidenceStore, RejectsIndexSelfPrefixMismatch) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string index = slurp(join_path(trust, "genesis/signature.index"));
            std::string prefix = index_self_prefix_digest(index);
            prefix[0] = prefix[0] == 'a' ? 'b' : 'a';
            std::string verification =
                reseal_fields(slurp(join_path(trust, "genesis/verification.record")),
                              {{"signature_index_sha256", prefix}});
            replace_bytes(trust, "genesis/verification.record", verification);
            const std::string transition =
                reseal_fields(slurp(join_path(trust, "genesis/transition.record")),
                              {{"signature_file_size", std::to_string(verification.size())},
                               {"signature_file_sha256", digest_hex(verification)}});
            replace_bytes(trust, "genesis/transition.record", transition);
        },
        EvidenceError::kDigest);
}

TEST(EvidenceStore, RejectsRawSignatureMutation) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            std::string signature = slurp(join_path(trust, "genesis/signature.bin"));
            signature[0] = static_cast<char>(signature[0] ^ 0x01);
            replace_bytes(trust, "genesis/signature.bin", signature);
        },
        EvidenceError::kDigest);
}

TEST(EvidenceStore, RejectsWrongTransitionSigningKey) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&,
           const StaleFixture& fixture) {
            const std::string payload = slurp(join_path(trust, "genesis/transition.payload"));
            install_transition_signature(
                trust, signed_message(fixture.chain.admin, trust_transition_message(payload)));
        },
        EvidenceError::kAuthorization);
}

TEST(EvidenceStore, RejectsRevokedSignerAfterPayloadVerification) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    ChainOptions options;
    options.revoked = "yes";
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture, options);
    expect_rejected(EvidenceError::kAuthorization,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
                    evidence.path(), admission.path());
}

TEST(EvidenceStore, RejectsNotYetValidSigner) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    ChainOptions options;
    options.valid_from = "2026-04-01";
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture, options);
    expect_rejected(EvidenceError::kDate,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
                    evidence.path(), admission.path());
}

TEST(EvidenceStore, RejectsExpiredSigner) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    ChainOptions options;
    options.valid_until = "2026-03-30";
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture, options);
    expect_rejected(EvidenceError::kDate,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture),
                    evidence.path(), admission.path());
}

TEST(EvidenceStore, RejectsUnrelatedTerminalTrust) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    expect_rejected(EvidenceError::kAuthorization,
                    inferedge::AuthenticatedCapability::authenticate(
                        "stale-lock", fixture.authorization, fixture.path, fixture.target,
                        trust.path(), evidence.path(), admission.path(), "trust-other",
                        kGenerationDate, kAdminId, kKeyFrom, kKeyUntil),
                    evidence.path(), admission.path());
}

TEST(EvidenceStore, RejectsWrongOperation) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    expect_rejected(EvidenceError::kAuthorization,
                    inferedge::AuthenticatedCapability::authenticate(
                        "publish", fixture.authorization, fixture.path, fixture.target,
                        trust.path(), evidence.path(), admission.path(), kTrustId, kGenerationDate,
                        kAdminId, kKeyFrom, kKeyUntil),
                    evidence.path(), admission.path());
}

TEST(EvidenceStore, RejectsWrongTarget) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    expect_rejected(EvidenceError::kAuthorization,
                    authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture,
                                       kTrustId, kGenerationDate, kAdminId, kKeyFrom, kKeyUntil,
                                       fixture.path, "lock/other.lock"),
                    evidence.path(), admission.path());
}

TEST(EvidenceStore, RejectsGenerationJump) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    const LaterGeneration later =
        plant_later_generation(trust.path(), admission.path(), fixture.chain, "3");
    expect_rejected(EvidenceError::kState,
                    inferedge::AuthenticatedCapability::authenticate(
                        "stale-lock", fixture.authorization, fixture.path, fixture.target,
                        trust.path(), evidence.path(), admission.path(), later.trust_id,
                        kGenerationDate, kAdminId, kKeyFrom, kKeyUntil),
                    evidence.path(), admission.path());
}

TEST(EvidenceStore, RejectsGenerationRollback) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    const LaterGeneration later =
        plant_later_generation(trust.path(), admission.path(), fixture.chain, "1");
    expect_rejected(EvidenceError::kState,
                    inferedge::AuthenticatedCapability::authenticate(
                        "stale-lock", fixture.authorization, fixture.path, fixture.target,
                        trust.path(), evidence.path(), admission.path(), later.trust_id,
                        kGenerationDate, kAdminId, kKeyFrom, kKeyUntil),
                    evidence.path(), admission.path());
}

TEST(EvidenceStore, RejectsRepeatedGeneration) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    plant_later_generation(trust.path(), admission.path(), fixture.chain, "2");
    const std::string transition = reseal_fields(
        slurp(join_path(trust.path(), "transitions/transition-gen2/transition.record")),
        {{"trust_generation", "1"}});
    replace_bytes(trust.path(), "transitions/transition-gen2/transition.record", transition);
    expect_rejected(EvidenceError::kState,
                    inferedge::AuthenticatedCapability::authenticate(
                        "stale-lock", fixture.authorization, fixture.path, fixture.target,
                        trust.path(), evidence.path(), admission.path(), "trust-gen2",
                        kGenerationDate, kAdminId, kKeyFrom, kKeyUntil),
                    evidence.path(), admission.path());
}

TEST(EvidenceStore, RejectsRevocationGenerationDecrease) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    ChainOptions options;
    options.revocation_generation = "2";
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture, options);
    const LaterGeneration later =
        plant_later_generation(trust.path(), admission.path(), fixture.chain, "2", "1");
    expect_rejected(EvidenceError::kAuthorization,
                    inferedge::AuthenticatedCapability::authenticate(
                        "stale-lock", fixture.authorization, fixture.path, fixture.target,
                        trust.path(), evidence.path(), admission.path(), later.trust_id,
                        kGenerationDate, kAdminId, kKeyFrom, kKeyUntil),
                    evidence.path(), admission.path());
}

TEST(EvidenceStore, RejectsRevocationIdentityChange) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    const std::string revocation_digest = digest_hex("revocation-gen2");
    const LaterGeneration later =
        plant_later_generation(trust.path(), admission.path(), fixture.chain, "2", "0",
                               "revocation-gen2", revocation_digest.c_str(), "genesis-v1", "4");
    expect_rejected(EvidenceError::kAuthorization,
                    inferedge::AuthenticatedCapability::authenticate(
                        "stale-lock", fixture.authorization, fixture.path, fixture.target,
                        trust.path(), evidence.path(), admission.path(), later.trust_id,
                        kGenerationDate, kAdminId, kKeyFrom, kKeyUntil),
                    evidence.path(), admission.path());
}

TEST(EvidenceStore, RejectsPayloadRegistryIdMismatch) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string payload =
                reseal_fields(slurp(join_path(trust, "genesis/transition.payload")),
                              {{"registry_id", "registry-other"}});
            replace_bytes(trust, "genesis/transition.payload", payload);
        },
        EvidenceError::kState);
}

TEST(EvidenceStore, RejectsPayloadTransitionIdMismatch) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string payload =
                reseal_fields(slurp(join_path(trust, "genesis/transition.payload")),
                              {{"transition_id", "transition-other"}});
            replace_bytes(trust, "genesis/transition.payload", payload);
        },
        EvidenceError::kState);
}

TEST(EvidenceStore, RejectsPayloadPublicationKindMismatch) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string payload =
                reseal_fields(slurp(join_path(trust, "genesis/transition.payload")),
                              {{"publication_kind", "ordinary"}});
            replace_bytes(trust, "genesis/transition.payload", payload);
        },
        EvidenceError::kState);
}

TEST(EvidenceStore, RejectsSignatureIdMismatch) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string verification =
                reseal_fields(slurp(join_path(trust, "genesis/verification.record")),
                              {{"signature_id", "transition-signature-other-v1"}});
            replace_bytes(trust, "genesis/verification.record", verification);
            bind_signature_file(trust, "genesis");
        },
        EvidenceError::kState);
}

TEST(EvidenceStore, RejectsSignatureBlobIdMismatch) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string index =
                reseal_fields(slurp(join_path(trust, "genesis/signature.index")),
                              {{"blob_id", "genesis-sigblob-v2"}});
            replace_bytes(trust, "genesis/signature.index", index);
            const std::string verification =
                reseal_fields(slurp(join_path(trust, "genesis/verification.record")),
                              {{"signature_index_size", std::to_string(index.size())},
                               {"signature_index_sha256", index_self_prefix_digest(index)}});
            replace_bytes(trust, "genesis/verification.record", verification);
            bind_signature_file(trust, "genesis");
        },
        EvidenceError::kState);
}

TEST(EvidenceStore, RejectsSignerKeyIdMismatch) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string verification =
                reseal_fields(slurp(join_path(trust, "genesis/verification.record")),
                              {{"signer_key_id", "admin-gen1"}});
            replace_bytes(trust, "genesis/verification.record", verification);
            bind_signature_file(trust, "genesis");
        },
        EvidenceError::kState);
}

TEST(EvidenceStore, RejectsSignatureIndexIdMismatch) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string verification =
                reseal_fields(slurp(join_path(trust, "genesis/verification.record")),
                              {{"signature_index_id", "genesis-sigindex-v2"}});
            replace_bytes(trust, "genesis/verification.record", verification);
            bind_signature_file(trust, "genesis");
        },
        EvidenceError::kState);
}

TEST(EvidenceStore, RejectsEarlierTransitionDate) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    plant_later_generation(trust.path(), admission.path(), fixture.chain, "2");
    const std::string directory = "transitions/transition-gen2";
    const std::string payload =
        reseal_fields(slurp(join_path(trust.path(), directory + "/transition.payload")),
                      {{"transition_date", "2026-03-30"}});
    replace_bytes(trust.path(), directory + "/transition.payload", payload);
    const std::string transition =
        reseal_fields(slurp(join_path(trust.path(), directory + "/transition.record")),
                      {{"transition_date", "2026-03-30"}});
    replace_bytes(trust.path(), directory + "/transition.record", transition);
    resign_family(trust.path(), directory, fixture.chain.admin);
    expect_rejected(EvidenceError::kDate,
                    inferedge::AuthenticatedCapability::authenticate(
                        "stale-lock", fixture.authorization, fixture.path, fixture.target,
                        trust.path(), evidence.path(), admission.path(), "trust-gen2",
                        kGenerationDate, kAdminId, kKeyFrom, kKeyUntil),
                    evidence.path(), admission.path());
}

TEST(EvidenceStore, RejectsFamilyRecordWrongMode) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            force_mode(trust, "genesis/transition.record", 0644);
        },
        EvidenceError::kForbidden);
}

TEST(EvidenceStore, RejectsFamilyModeBeforeLinkCount) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string path = join_path(trust, "genesis/transition.payload");
            force_mode(trust, "genesis/transition.payload", 0644);
            ASSERT_EQ(link(path.c_str(), (path + ".link").c_str()), 0);
        },
        EvidenceError::kForbidden);
}

TEST(EvidenceStore, RejectsPayloadCarriageReturn) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            std::string payload = slurp(join_path(trust, "genesis/transition.payload"));
            const std::size_t newline = payload.find('\n');
            ASSERT_NE(newline, std::string::npos);
            payload.insert(newline, 1, '\r');
            plant_payload_bytes(trust, payload);
        },
        EvidenceError::kSchema);
}

TEST(EvidenceStore, RejectsPayloadEmbeddedNul) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            std::string payload = slurp(join_path(trust, "genesis/transition.payload"));
            payload.insert(payload.find('='), 1, '\0');
            plant_payload_bytes(trust, payload);
        },
        EvidenceError::kSchema);
}

TEST(EvidenceStore, RejectsPayloadTrailingByte) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            std::string payload = slurp(join_path(trust, "genesis/transition.payload"));
            payload.push_back('X');
            plant_payload_bytes(trust, payload);
        },
        EvidenceError::kSchema);
}

void replace_payload_inode(const std::string& path, std::string_view bytes) {
    ASSERT_EQ(unlink(path.c_str()), 0);
    const int fd = open(path.c_str(), O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC | O_WRONLY, 0444);
    ASSERT_GE(fd, 0);
    ASSERT_EQ(write(fd, bytes.data(), bytes.size()), static_cast<ssize_t>(bytes.size()));
    ASSERT_EQ(close(fd), 0);
}

TEST(EvidenceStore, RejectsSameBytePayloadInodeSubstitution) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    const std::string path = join_path(trust.path(), "genesis/transition.payload");
    const std::string bytes = slurp(path);
    ArmedPayloadOpen gate("transition.payload", 2);
    inferedge::CapabilityResult result;
    std::thread worker([&] {
        result = authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture);
    });
    JoinWorker joined(worker);
#if defined(__APPLE__)
    ASSERT_GT(openat_rebind_count, 0) << openat_rebind_failure;
#endif
    ASSERT_TRUE(gate.wait_until_paused());
    EXPECT_EQ(gate.matches(), 2);
    struct stat before{};
    ASSERT_EQ(lstat(path.c_str(), &before), 0);
    const auto unrelated_began = std::chrono::steady_clock::now();
    const int extra =
        openat(AT_FDCWD, join_path(evidence.path(), "unrelated-during-pause.bin").c_str(),
               O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC | O_WRONLY, 0600);
    EXPECT_LT(std::chrono::steady_clock::now() - unrelated_began, std::chrono::seconds(1));
    EXPECT_GE(extra, 0);
    if (extra >= 0) {
        EXPECT_EQ(write(extra, "x", 1), 1);
        EXPECT_EQ(close(extra), 0);
    }
    replace_payload_inode(path, bytes);
    struct stat after{};
    ASSERT_EQ(lstat(path.c_str(), &after), 0);
    EXPECT_EQ(before.st_dev, after.st_dev);
    EXPECT_NE(before.st_ino, after.st_ino);
    EXPECT_EQ(after.st_mode & 0777, 0444);
    EXPECT_EQ(after.st_nlink, 1);
    gate.release();
    worker.join();
    EXPECT_FALSE(gate.timed_out());
    EXPECT_EQ(result.status(), EvidenceError::kRace);
    EXPECT_FALSE(result.has_capability());
    EXPECT_EQ(slurp(join_path(evidence.path(), "provenance/item-a/keep.bin")), "keep-unrelated");
    EXPECT_TRUE(path_exists(join_path(evidence.path(), "lock/evidence.lock")));
}

TEST(EvidenceStore, PayloadReopenSameInodeIsNotRace) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    ArmedPayloadOpen gate("transition.payload", 2);
    inferedge::CapabilityResult result;
    std::thread worker([&] {
        result = authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture);
    });
    JoinWorker joined(worker);
    ASSERT_TRUE(gate.wait_until_paused());
    EXPECT_EQ(gate.matches(), 2);
    gate.release();
    worker.join();
    EXPECT_FALSE(gate.timed_out());
    EXPECT_EQ(result.status(), EvidenceError::kOk) << static_cast<int>(result.status());
    EXPECT_TRUE(result.has_capability());
    EXPECT_TRUE(path_exists(join_path(evidence.path(), "lock/evidence.lock")));
}

TEST(EvidenceStore, PayloadReopenDifferentBytesGrantsNothing) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    const std::string path = join_path(trust.path(), "genesis/transition.payload");
    std::string bytes = slurp(path);
    bytes[0] = static_cast<char>(bytes[0] ^ 0x01);
    ArmedPayloadOpen gate("transition.payload", 2);
    inferedge::CapabilityResult result;
    std::thread worker([&] {
        result = authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture);
    });
    JoinWorker joined(worker);
    ASSERT_TRUE(gate.wait_until_paused());
    struct stat before{};
    ASSERT_EQ(lstat(path.c_str(), &before), 0);
    replace_payload_inode(path, bytes);
    struct stat after{};
    ASSERT_EQ(lstat(path.c_str(), &after), 0);
    EXPECT_EQ(before.st_dev, after.st_dev);
    EXPECT_NE(before.st_ino, after.st_ino);
    gate.release();
    worker.join();
    EXPECT_FALSE(gate.timed_out());
    EXPECT_EQ(result.status(), EvidenceError::kRace);
    EXPECT_FALSE(result.has_capability());
    EXPECT_TRUE(path_exists(join_path(evidence.path(), "lock/evidence.lock")));
}

int count_open_descriptors() {
    long configured = sysconf(_SC_OPEN_MAX);
    int limit = 1024;
    if (configured > 0 && configured < limit) {
        limit = static_cast<int>(configured);
    }
    int count = 0;
    for (int fd = 0; fd < limit; ++fd) {
        if (fcntl(fd, F_GETFD) != -1) {
            ++count;
        }
    }
    return count;
}

TEST(EvidenceStore, RejectsRetainedPayloadTruncation) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    const std::string path = join_path(trust.path(), "genesis/transition.payload");
    ArmedPayloadOpen gate("transition.payload", 2);
    inferedge::CapabilityResult result;
    std::thread worker([&] {
        result = authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture);
    });
    JoinWorker joined(worker);
    ASSERT_TRUE(gate.wait_until_paused());
    struct stat before{};
    ASSERT_EQ(lstat(path.c_str(), &before), 0);
    ASSERT_EQ(chmod(path.c_str(), 0644), 0);
    const int writable = open(path.c_str(), O_RDWR | O_NOFOLLOW | O_CLOEXEC);
    ASSERT_GE(writable, 0);
    ASSERT_EQ(ftruncate(writable, 0), 0);
    struct stat truncated{};
    ASSERT_EQ(fstat(writable, &truncated), 0);
    EXPECT_EQ(before.st_dev, truncated.st_dev);
    EXPECT_EQ(before.st_ino, truncated.st_ino);
    EXPECT_EQ(truncated.st_size, 0);
    ASSERT_EQ(close(writable), 0);
    gate.release();
    worker.join();
    EXPECT_FALSE(gate.timed_out());
    EXPECT_EQ(result.status(), EvidenceError::kRace);
    EXPECT_FALSE(result.has_capability());
    EXPECT_EQ(slurp(join_path(evidence.path(), "provenance/item-a/keep.bin")), "keep-unrelated");
    EXPECT_TRUE(path_exists(join_path(evidence.path(), "lock/evidence.lock")));
}

TEST(EvidenceStore, PayloadDescriptorClosesAfterDecision) {
    const int baseline = count_open_descriptors();
    for (int repeat = 0; repeat < 4; ++repeat) {
        TempRoot trust;
        TempRoot admission;
        TempRoot evidence;
        ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
        StaleFixture fixture;
        plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
        const inferedge::CapabilityResult result =
            authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture);
        EXPECT_EQ(result.status(), EvidenceError::kOk);
        EXPECT_TRUE(result.has_capability());
    }
    for (int repeat = 0; repeat < 4; ++repeat) {
        TempRoot trust;
        TempRoot admission;
        TempRoot evidence;
        ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
        StaleFixture fixture;
        plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
        std::string payload = slurp(join_path(trust.path(), "genesis/transition.payload"));
        const std::size_t newline = payload.find('\n');
        ASSERT_NE(newline, std::string::npos);
        payload.insert(newline, 1, '\r');
        plant_payload_bytes(trust.path(), payload);
        const inferedge::CapabilityResult result =
            authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture);
        EXPECT_EQ(result.status(), EvidenceError::kSchema);
        EXPECT_FALSE(result.has_capability());
    }
    {
        TempRoot trust;
        TempRoot admission;
        TempRoot evidence;
        ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
        StaleFixture fixture;
        plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
        const std::string path = join_path(trust.path(), "genesis/transition.payload");
        const std::string bytes = slurp(path);
        ArmedPayloadOpen gate("transition.payload", 2);
        inferedge::CapabilityResult result;
        std::thread worker([&] {
            result = authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture);
        });
        JoinWorker joined(worker);
        ASSERT_TRUE(gate.wait_until_paused());
        replace_payload_inode(path, bytes);
        gate.release();
        worker.join();
        EXPECT_EQ(result.status(), EvidenceError::kRace);
        EXPECT_FALSE(result.has_capability());
    }
    EXPECT_EQ(count_open_descriptors(), baseline);
}

TEST(EvidenceStore, PayloadOpenGateIgnoresOtherPaths) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    ArmedPayloadOpen gate("not-a-leaf", 1);
    inferedge::CapabilityResult result;
    std::thread worker([&] {
        result = authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture);
    });
    JoinWorker joined(worker);
    worker.join();
    EXPECT_FALSE(gate.paused());
    EXPECT_EQ(gate.matches(), 0);
    EXPECT_FALSE(gate.timed_out());
    EXPECT_EQ(result.status(), EvidenceError::kOk) << static_cast<int>(result.status());
}

TEST(EvidenceStore, PayloadOpenGateTimesOut) {
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    ArmedPayloadOpen gate("transition.payload", 2);
    inferedge::CapabilityResult result;
    std::thread worker([&] {
        result = authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture);
    });
    JoinWorker joined(worker);
    ASSERT_TRUE(gate.wait_until_paused());
    worker.join();
    EXPECT_TRUE(gate.timed_out());
    EXPECT_EQ(result.status(), EvidenceError::kOk) << static_cast<int>(result.status());
    EXPECT_TRUE(result.has_capability());
}

TEST(EvidenceStore, InactivePayloadOpenGatePreservesAcceptance) {
    reset_payload_open_gate();
    TempRoot trust;
    TempRoot admission;
    TempRoot evidence;
    ASSERT_TRUE(trust.ok() && admission.ok() && evidence.ok());
    StaleFixture fixture;
    plant_stale_fixture(trust.path(), admission.path(), evidence.path(), fixture);
    inferedge::CapabilityResult result =
        authenticate_stale(trust.path(), evidence.path(), admission.path(), fixture);
    EXPECT_EQ(result.status(), EvidenceError::kOk) << static_cast<int>(result.status());
    EXPECT_TRUE(result.has_capability());
    EXPECT_TRUE(path_exists(join_path(evidence.path(), "lock/evidence.lock")));
}

TEST(EvidenceStore, RejectsSecondMessageFallback) {
    reject_mutated(
        [](const std::string& trust, const std::string&, const std::string&, const StaleFixture&) {
            const std::string record = slurp(join_path(trust, "genesis/transition.record"));
            const std::size_t checksum = record.rfind("\nrecord_sha256=");
            std::string legacy = record.substr(0, checksum + 1);
            const std::size_t digest_at = legacy.find("signature_file_sha256=") + 22;
            legacy.replace(digest_at, 64, 64, '0');
            const std::string legacy_signature = signed_message(bootstrap_key(), legacy);
            const std::string payload = slurp(join_path(trust, "genesis/transition.payload"));
            EXPECT_NE(legacy_signature,
                      signed_message(bootstrap_key(), trust_transition_message(payload)));
            install_transition_signature(trust, legacy_signature);
        },
        EvidenceError::kAuthorization);
}
