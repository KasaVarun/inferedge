#if defined(__linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#endif

#include "inferedge/evidence_store.hpp"

#include "inferedge/ed25519_verify.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <fcntl.h>
#include <memory>
#include <set>
#include <span>
#include <stdio.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <utility>

namespace inferedge {
namespace {

struct CapabilityStamp {};

// Address is visible only in this translation unit. Another unit cannot form it.
const CapabilityStamp kCapabilityStamp{};

class Fd {
public:
    Fd() = default;
    explicit Fd(int fd) : fd_(fd) {}
    ~Fd() { reset(); }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    Fd(Fd&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
    Fd& operator=(Fd&& other) noexcept {
        if (this != &other) {
            reset();
            fd_ = other.fd_;
            other.fd_ = -1;
        }
        return *this;
    }
    int get() const { return fd_; }
    int release() {
        const int value = fd_;
        fd_ = -1;
        return value;
    }
    void reset() {
        if (fd_ >= 0) {
            close(fd_);
        }
        fd_ = -1;
    }
    explicit operator bool() const { return fd_ >= 0; }

private:
    int fd_{-1};
};

class SystemTime final : public EvidenceTime {
public:
    EvidenceError civil_date(std::string& out) const override {
        const std::time_t now = std::time(nullptr);
        if (now == static_cast<std::time_t>(-1)) {
            return EvidenceError::kDate;
        }
        std::tm parts{};
        if (gmtime_r(&now, &parts) == nullptr) {
            return EvidenceError::kDate;
        }
        char buffer[16];
        const int wrote = snprintf(buffer, sizeof(buffer), "%04d-%02d-%02d", parts.tm_year + 1900,
                                   parts.tm_mon + 1, parts.tm_mday);
        if (wrote != 10 || validate_date(buffer) != AdmissionError::kOk) {
            return EvidenceError::kDate;
        }
        out = buffer;
        return EvidenceError::kOk;
    }
    std::chrono::steady_clock::time_point now() const override {
        return std::chrono::steady_clock::now();
    }
    void sleep_for(std::chrono::milliseconds duration) override {
        std::this_thread::sleep_for(duration);
    }
};

EvidenceTime& time_of(EvidenceTime* clock) {
    static SystemTime time;
    return clock == nullptr ? static_cast<EvidenceTime&>(time) : *clock;
}

EvidenceError from_admission(AdmissionError error) {
    switch (error) {
    case AdmissionError::kOk:
        return EvidenceError::kOk;
    case AdmissionError::kOverflow:
        return EvidenceError::kOverflow;
    case AdmissionError::kInvalidDate:
        return EvidenceError::kDate;
    case AdmissionError::kChecksumMismatch:
        return EvidenceError::kChecksum;
    case AdmissionError::kSize:
        return EvidenceError::kSize;
    case AdmissionError::kLexical:
    case AdmissionError::kChecksumFormat:
    case AdmissionError::kMalformed:
        return EvidenceError::kLexical;
    default:
        return EvidenceError::kSchema;
    }
}

EvidenceError from_errno(int error) {
    if (error == ELOOP || error == EMLINK) {
        return EvidenceError::kSymlink;
    }
    if (error == ENOTDIR) {
        return EvidenceError::kType;
    }
    if (error == EEXIST) {
        return EvidenceError::kPresent;
    }
    if (error == ENOENT) {
        return EvidenceError::kAbsent;
    }
    return EvidenceError::kIo;
}

std::string text_u64(std::uint64_t value) {
    return std::to_string(value);
}

int decimal_width(std::uint64_t value) {
    int width = 1;
    while (value >= 10) {
        value /= 10;
        ++width;
    }
    return width;
}

bool contains_forbidden_byte(std::string_view text) {
    return text.find('\\') != std::string_view::npos || text.find('\0') != std::string_view::npos ||
           text.find('\r') != std::string_view::npos || text.find('\t') != std::string_view::npos;
}

std::vector<std::string_view> split_segments(std::string_view path) {
    std::vector<std::string_view> parts;
    std::size_t start = 0;
    while (start <= path.size()) {
        const std::size_t slash = path.find('/', start);
        if (slash == std::string_view::npos) {
            parts.push_back(path.substr(start));
            break;
        }
        parts.push_back(path.substr(start, slash - start));
        start = slash + 1;
    }
    return parts;
}

EvidenceError check_segments(const std::vector<std::string_view>& parts) {
    if (parts.empty()) {
        return EvidenceError::kPath;
    }
    for (std::string_view part : parts) {
        if (part.empty() || part == "." || part == ".." || part.size() > 255) {
            return EvidenceError::kPath;
        }
    }
    return EvidenceError::kOk;
}

bool ends_with(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}

EvidenceError require_id_bin(std::string_view filename) {
    if (!ends_with(filename, ".bin") || filename.size() == 4) {
        return EvidenceError::kPath;
    }
    const std::string_view ident = filename.substr(0, filename.size() - 4);
    if (const EvidenceError status = from_admission(validate_project_id(ident));
        status != EvidenceError::kOk) {
        return status;
    }
    return from_admission(validate_filename(filename));
}

enum class FilePolicy { kLegacy, kTrustFamily };

EvidenceError classify_file(const struct stat& status, FilePolicy policy = FilePolicy::kLegacy) {
    if (S_ISLNK(status.st_mode)) {
        return EvidenceError::kSymlink;
    }
    if (S_ISDIR(status.st_mode)) {
        return EvidenceError::kType;
    }
    if (!S_ISREG(status.st_mode)) {
        return EvidenceError::kSpecial;
    }
    if (policy == FilePolicy::kTrustFamily && (status.st_mode & 0777) != 0444) {
        return EvidenceError::kForbidden;
    }
    if (status.st_nlink != 1) {
        return EvidenceError::kLink;
    }
    return EvidenceError::kOk;
}

EvidenceError seal_prefix(const std::string& prefix, std::string& out) {
    if (prefix.empty() || prefix.back() != '\n') {
        return EvidenceError::kInternal;
    }
    if (prefix.size() + 79 > kMaxCanonicalRecordBytes) {
        return EvidenceError::kSize;
    }
    out = prefix;
    out += "record_sha256=";
    out += digest_hex(prefix);
    out += '\n';
    return EvidenceError::kOk;
}

void append_kv(std::string& prefix, std::string_view key, std::string_view value) {
    prefix.append(key);
    prefix.push_back('=');
    prefix.append(value);
    prefix.push_back('\n');
}

EvidenceError split_canonical(std::string_view bytes, std::vector<std::string>& lines) {
    if (bytes.empty() || bytes.size() > kMaxCanonicalRecordBytes || bytes.back() != '\n') {
        return EvidenceError::kSize;
    }
    if (contains_forbidden_byte(bytes)) {
        return EvidenceError::kLexical;
    }
    lines.clear();
    std::size_t start = 0;
    while (start < bytes.size()) {
        const std::size_t end = bytes.find('\n', start);
        if (end == std::string_view::npos || end == start) {
            return EvidenceError::kLexical;
        }
        lines.emplace_back(bytes.substr(start, end - start));
        start = end + 1;
    }
    if (lines.empty()) {
        return EvidenceError::kLexical;
    }
    constexpr std::string_view checksum_prefix = "record_sha256=";
    const std::string& last = lines.back();
    if (last.size() != checksum_prefix.size() + 64 ||
        last.compare(0, checksum_prefix.size(), checksum_prefix) != 0) {
        return EvidenceError::kChecksum;
    }
    const std::string_view checksum = std::string_view(last).substr(checksum_prefix.size());
    if (from_admission(validate_sha256(checksum)) != EvidenceError::kOk) {
        return EvidenceError::kChecksum;
    }
    const std::string_view prefix = bytes.substr(0, bytes.size() - last.size() - 1);
    if (digest_hex(prefix) != checksum) {
        return EvidenceError::kChecksum;
    }
    lines.pop_back();
    return EvidenceError::kOk;
}

EvidenceError take_value(const std::string& line, std::string_view key, std::string& value) {
    const std::string expected = std::string(key) + "=";
    if (line.compare(0, expected.size(), expected) != 0) {
        return EvidenceError::kSchema;
    }
    value = line.substr(expected.size());
    if (value.empty() || value.find(' ') != std::string::npos) {
        return EvidenceError::kLexical;
    }
    return EvidenceError::kOk;
}

EvidenceError require_exact(const std::vector<std::string>& lines, std::size_t index,
                            std::string_view key, std::string_view expected) {
    if (index >= lines.size()) {
        return EvidenceError::kSchema;
    }
    std::string value;
    if (const EvidenceError status = take_value(lines[index], key, value);
        status != EvidenceError::kOk) {
        return status;
    }
    return value == expected ? EvidenceError::kOk : EvidenceError::kSchema;
}

EvidenceError require_field(const std::vector<std::string>& lines, std::size_t index,
                            std::string_view key, std::string& value) {
    if (index >= lines.size()) {
        return EvidenceError::kSchema;
    }
    return take_value(lines[index], key, value);
}

EvidenceError require_id(std::string_view value) {
    return from_admission(validate_project_id(value));
}

EvidenceError require_date(std::string_view value) {
    return from_admission(validate_date(value));
}

EvidenceError require_sha(std::string_view value, bool zero_eligible) {
    return from_admission(validate_digest_or_zero(value, zero_eligible));
}

EvidenceError require_u64(std::string_view value, std::uint64_t& out) {
    return from_admission(parse_uint64(value, out));
}

const char* class_name(EvidenceObjectClass object_class) {
    switch (object_class) {
    case EvidenceObjectClass::kPublishedBlob:
        return "published_blob";
    case EvidenceObjectClass::kRetainedMetadata:
        return "retained_metadata";
    case EvidenceObjectClass::kTemporaryMetadata:
        return "temporary_metadata";
    }
    return nullptr;
}

const char* cleanup_name(CleanupObjectClass object_class) {
    switch (object_class) {
    case CleanupObjectClass::kEvidenceReservation:
        return "evidence_reservation";
    case CleanupObjectClass::kPartialStaging:
        return "partial_staging";
    case CleanupObjectClass::kTemporaryMetadata:
        return "temporary_metadata";
    }
    return nullptr;
}

EvidenceError parse_object_class(std::string_view text, EvidenceObjectClass& out) {
    if (text == "published_blob") {
        out = EvidenceObjectClass::kPublishedBlob;
        return EvidenceError::kOk;
    }
    if (text == "retained_metadata") {
        out = EvidenceObjectClass::kRetainedMetadata;
        return EvidenceError::kOk;
    }
    if (text == "temporary_metadata") {
        out = EvidenceObjectClass::kTemporaryMetadata;
        return EvidenceError::kOk;
    }
    return EvidenceError::kSchema;
}

EvidenceError parse_cleanup_class(std::string_view text, CleanupObjectClass& out) {
    if (text == "evidence_reservation") {
        out = CleanupObjectClass::kEvidenceReservation;
        return EvidenceError::kOk;
    }
    if (text == "partial_staging") {
        out = CleanupObjectClass::kPartialStaging;
        return EvidenceError::kOk;
    }
    if (text == "temporary_metadata") {
        out = CleanupObjectClass::kTemporaryMetadata;
        return EvidenceError::kOk;
    }
    return EvidenceError::kSchema;
}

bool known_reason(std::string_view reason) {
    return reason == "checksum_rejected" || reason == "reproduction_mismatch" ||
           reason == "recipe_echo_rejected" || reason == "cap_rejected" ||
           reason == "symlink_rejected" || reason == "signature_failed" ||
           reason == "invariant_mismatch" || reason == "other_rejected";
}

EvidenceError open_root(std::string_view path, Fd& out) {
    if (path.empty() || path.front() != '/' || path.find('\0') != std::string_view::npos ||
        path.find('\\') != std::string_view::npos || path.find("//") != std::string_view::npos) {
        return EvidenceError::kPath;
    }
    const std::vector<std::string_view> parts = split_segments(path.substr(1));
    if (const EvidenceError status = check_segments(parts); status != EvidenceError::kOk) {
        return status;
    }
    const int slash = open("/", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (slash < 0) {
        return EvidenceError::kIo;
    }
    Fd current(slash);
    for (std::string_view part : parts) {
        const std::string name(part);
        struct stat status{};
        if (fstatat(current.get(), name.c_str(), &status, AT_SYMLINK_NOFOLLOW) != 0) {
            return from_errno(errno);
        }
        if (S_ISLNK(status.st_mode)) {
            return EvidenceError::kSymlink;
        }
        if (!S_ISDIR(status.st_mode)) {
            return EvidenceError::kType;
        }
        const int next =
            openat(current.get(), name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (next < 0) {
            return from_errno(errno);
        }
        struct stat opened{};
        if (fstat(next, &opened) != 0) {
            close(next);
            return EvidenceError::kIo;
        }
        if (S_ISLNK(opened.st_mode)) {
            close(next);
            return EvidenceError::kSymlink;
        }
        if (!S_ISDIR(opened.st_mode) || opened.st_dev != status.st_dev ||
            opened.st_ino != status.st_ino) {
            close(next);
            return EvidenceError::kRace;
        }
        current = Fd(next);
    }
    out = std::move(current);
    return EvidenceError::kOk;
}

EvidenceError directory_identity(int fd, std::uint64_t& device, std::uint64_t& inode,
                                 std::uint32_t& kind) {
    struct stat status{};
    if (fstat(fd, &status) != 0) {
        return EvidenceError::kIo;
    }
    if (!S_ISDIR(status.st_mode)) {
        return EvidenceError::kType;
    }
    device = static_cast<std::uint64_t>(status.st_dev);
    inode = static_cast<std::uint64_t>(status.st_ino);
    kind = static_cast<std::uint32_t>(status.st_mode & S_IFMT);
    return EvidenceError::kOk;
}

EvidenceError capture_directory(std::string_view path, std::uint64_t& device, std::uint64_t& inode,
                                std::uint32_t& kind) {
    Fd opened;
    if (const EvidenceError status = open_root(path, opened); status != EvidenceError::kOk) {
        return status;
    }
    return directory_identity(opened.get(), device, inode, kind);
}

EvidenceError require_bound_root(const AuthenticatedCapability& capability, bool evidence,
                                 std::string_view path, int held_fd) {
    std::uint64_t device = 0;
    std::uint64_t inode = 0;
    std::uint32_t kind = 0;
    if (const EvidenceError status = capture_directory(path, device, inode, kind);
        status != EvidenceError::kOk) {
        return status;
    }
    const std::uint64_t bound_device =
        evidence ? capability.evidence_device() : capability.admission_device();
    const std::uint64_t bound_inode =
        evidence ? capability.evidence_inode() : capability.admission_inode();
    const std::uint32_t bound_kind =
        evidence ? capability.evidence_kind() : capability.admission_kind();
    if (device != bound_device || inode != bound_inode || kind != bound_kind) {
        return EvidenceError::kRace;
    }
    if (held_fd >= 0) {
        std::uint64_t held_device = 0;
        std::uint64_t held_inode = 0;
        std::uint32_t held_kind = 0;
        if (const EvidenceError status =
                directory_identity(held_fd, held_device, held_inode, held_kind);
            status != EvidenceError::kOk) {
            return status;
        }
        if (held_device != bound_device || held_inode != bound_inode || held_kind != bound_kind) {
            return EvidenceError::kRace;
        }
    }
    return EvidenceError::kOk;
}

EvidenceError walk_parent(int root_fd, std::string_view relative, bool create, Fd& parent,
                          std::string& leaf) {
    if (const EvidenceError status = validate_evidence_relative_path(relative);
        status != EvidenceError::kOk) {
        return status;
    }
    const std::vector<std::string_view> parts = split_segments(relative);
    Fd current(dup(root_fd));
    if (!current) {
        return EvidenceError::kIo;
    }
    for (std::size_t index = 0; index + 1 < parts.size(); ++index) {
        const std::string name(parts[index]);
        struct stat status{};
        if (fstatat(current.get(), name.c_str(), &status, AT_SYMLINK_NOFOLLOW) != 0) {
            if (errno != ENOENT || !create) {
                return from_errno(errno);
            }
            if (mkdirat(current.get(), name.c_str(), 0700) != 0) {
                return from_errno(errno);
            }
        }
        const int next =
            openat(current.get(), name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (next < 0) {
            return from_errno(errno);
        }
        if (fchmod(next, 0700) != 0) {
            close(next);
            return EvidenceError::kIo;
        }
        struct stat opened{};
        if (fstat(next, &opened) != 0) {
            close(next);
            return EvidenceError::kIo;
        }
        if (!S_ISDIR(opened.st_mode)) {
            close(next);
            return EvidenceError::kType;
        }
        current = Fd(next);
    }
    parent = std::move(current);
    leaf = std::string(parts.back());
    return EvidenceError::kOk;
}

EvidenceError stat_leaf(int root_fd, std::string_view relative, struct stat& out, bool& present) {
    Fd parent;
    std::string leaf;
    const EvidenceError walked = walk_parent(root_fd, relative, false, parent, leaf);
    if (walked == EvidenceError::kAbsent) {
        present = false;
        return EvidenceError::kOk;
    }
    if (walked != EvidenceError::kOk) {
        return walked;
    }
    if (fstatat(parent.get(), leaf.c_str(), &out, AT_SYMLINK_NOFOLLOW) != 0) {
        if (errno == ENOENT) {
            present = false;
            return EvidenceError::kOk;
        }
        return from_errno(errno);
    }
    present = true;
    return EvidenceError::kOk;
}

EvidenceError require_absent_path(int root_fd, std::string_view relative) {
    struct stat status{};
    bool present = false;
    const EvidenceError checked = stat_leaf(root_fd, relative, status, present);
    if (checked != EvidenceError::kOk) {
        return checked;
    }
    if (!present) {
        return EvidenceError::kOk;
    }
    if (S_ISLNK(status.st_mode)) {
        return EvidenceError::kSymlink;
    }
    return EvidenceError::kPresent;
}

EvidenceError read_fd(int fd, std::uint64_t size, std::string& out) {
    if (size > kMaxCanonicalRecordBytes) {
        return EvidenceError::kSize;
    }
    std::string data(static_cast<std::size_t>(size), '\0');
    std::size_t got = 0;
    while (got < data.size()) {
        const ssize_t count =
            pread(fd, data.data() + got, data.size() - got, static_cast<off_t>(got));
        if (count < 0) {
            return EvidenceError::kIo;
        }
        if (count == 0) {
            return EvidenceError::kSize;
        }
        got += static_cast<std::size_t>(count);
    }
    std::uint8_t extra = 0;
    const ssize_t more = pread(fd, &extra, 1, static_cast<off_t>(size));
    if (more < 0) {
        return EvidenceError::kIo;
    }
    if (more != 0) {
        return EvidenceError::kSize;
    }
    out = std::move(data);
    return EvidenceError::kOk;
}

EvidenceError read_limited(int fd, std::uint64_t size, std::uint64_t limit, std::string& out) {
    if (size > limit) {
        return EvidenceError::kSize;
    }
    std::string data(static_cast<std::size_t>(size), '\0');
    std::size_t got = 0;
    while (got < data.size()) {
        const ssize_t count =
            pread(fd, data.data() + got, data.size() - got, static_cast<off_t>(got));
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            return EvidenceError::kIo;
        }
        if (count == 0) {
            return EvidenceError::kSize;
        }
        got += static_cast<std::size_t>(count);
    }
    out = std::move(data);
    return EvidenceError::kOk;
}

EvidenceError write_all(int fd, std::string_view bytes) {
    std::size_t wrote = 0;
    while (wrote < bytes.size()) {
        const ssize_t count =
            pwrite(fd, bytes.data() + wrote, bytes.size() - wrote, static_cast<off_t>(wrote));
        if (count < 0) {
            return EvidenceError::kIo;
        }
        wrote += static_cast<std::size_t>(count);
    }
    if (fsync(fd) != 0) {
        return EvidenceError::kIo;
    }
    return EvidenceError::kOk;
}

EvidenceError create_file(int root_fd, std::string_view relative, std::string_view bytes, Fd& file,
                          Fd& parent, std::uint64_t& device, std::uint64_t& inode) {
    std::string leaf;
    if (const EvidenceError status = walk_parent(root_fd, relative, true, parent, leaf);
        status != EvidenceError::kOk) {
        return status;
    }
    const int fd = openat(parent.get(), leaf.c_str(),
                          O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC | O_RDWR, 0600);
    if (fd < 0) {
        return from_errno(errno);
    }
    file = Fd(fd);
    const auto abandon = [&]() {
        unlinkat(parent.get(), leaf.c_str(), 0);
        fsync(parent.get());
        file.reset();
    };
    if (fchmod(file.get(), 0600) != 0) {
        abandon();
        return EvidenceError::kIo;
    }
    if (const EvidenceError status = write_all(file.get(), bytes); status != EvidenceError::kOk) {
        abandon();
        return status;
    }
    struct stat status{};
    if (fstat(file.get(), &status) != 0) {
        abandon();
        return EvidenceError::kIo;
    }
    if (const EvidenceError kind = classify_file(status); kind != EvidenceError::kOk) {
        abandon();
        return kind;
    }
    if (static_cast<std::uint64_t>(status.st_size) != bytes.size() ||
        (status.st_mode & 0777) != 0600) {
        abandon();
        return EvidenceError::kSize;
    }
    device = static_cast<std::uint64_t>(status.st_dev);
    inode = static_cast<std::uint64_t>(status.st_ino);
    if (fsync(parent.get()) != 0) {
        abandon();
        return EvidenceError::kIo;
    }
    return EvidenceError::kOk;
}

EvidenceError exclusive_rename(int from_dir, const char* from, int to_dir, const char* to) {
#if defined(__APPLE__)
    if (renameatx_np(from_dir, from, to_dir, to, RENAME_EXCL) != 0) {
        return from_errno(errno);
    }
#elif defined(__linux__)
    if (renameat2(from_dir, from, to_dir, to, RENAME_NOREPLACE) != 0) {
        return from_errno(errno);
    }
#else
    return EvidenceError::kInternal;
#endif
    return EvidenceError::kOk;
}

struct TrackedFile {
    std::string relative;
    std::uint64_t device{0};
    std::uint64_t inode{0};
};

EvidenceError unlink_tracked(int root_fd, const TrackedFile& file) {
    Fd parent;
    std::string leaf;
    if (const EvidenceError status = walk_parent(root_fd, file.relative, false, parent, leaf);
        status != EvidenceError::kOk) {
        return status;
    }
    struct stat path_status{};
    if (fstatat(parent.get(), leaf.c_str(), &path_status, AT_SYMLINK_NOFOLLOW) != 0) {
        return from_errno(errno);
    }
    if (const EvidenceError kind = classify_file(path_status); kind != EvidenceError::kOk) {
        return kind;
    }
    if (static_cast<std::uint64_t>(path_status.st_dev) != file.device ||
        static_cast<std::uint64_t>(path_status.st_ino) != file.inode) {
        return EvidenceError::kRace;
    }
    if (unlinkat(parent.get(), leaf.c_str(), 0) != 0) {
        return from_errno(errno);
    }
    if (fsync(parent.get()) != 0) {
        return EvidenceError::kIo;
    }
    return EvidenceError::kOk;
}

// Cooperative lock: a compliant writer mutates the evidence root only while holding
// lock/evidence.lock, and this scan runs while that lock is held. Descriptor-relative
// openat rejects a symlink and fails the scan when an observed entry vanishes, changes
// type, or changes device or inode before its size is added. A non-cooperating writer can
// still create a name after readdir has passed the insertion point. That race is outside
// the cooperative protocol. An inconsistent entry that this scan does observe is never
// treated as a successful sum.
EvidenceError sum_fd(int dir_fd, std::uint64_t& total) {
    const int duplicated = dup(dir_fd);
    if (duplicated < 0) {
        return EvidenceError::kIo;
    }
    DIR* directory = fdopendir(duplicated);
    if (directory == nullptr) {
        close(duplicated);
        return EvidenceError::kIo;
    }
    while (true) {
        errno = 0;
        const dirent* entry = readdir(directory);
        if (entry == nullptr) {
            const int error = errno;
            closedir(directory);
            return error == 0 ? EvidenceError::kOk : EvidenceError::kIo;
        }
        const char* name = entry->d_name;
        if (std::strcmp(name, ".") == 0 || std::strcmp(name, "..") == 0) {
            continue;
        }
        if (std::strchr(name, '/') != nullptr || std::strchr(name, '\\') != nullptr) {
            closedir(directory);
            return EvidenceError::kPath;
        }
        struct stat status{};
        if (fstatat(dir_fd, name, &status, AT_SYMLINK_NOFOLLOW) != 0) {
            closedir(directory);
            return from_errno(errno);
        }
        if (S_ISLNK(status.st_mode)) {
            closedir(directory);
            return EvidenceError::kSymlink;
        }
        if (S_ISDIR(status.st_mode)) {
            const int child = openat(dir_fd, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
            if (child < 0) {
                closedir(directory);
                return errno == ENOENT ? EvidenceError::kRace : from_errno(errno);
            }
            struct stat opened{};
            if (fstat(child, &opened) != 0) {
                close(child);
                closedir(directory);
                return EvidenceError::kIo;
            }
            if (!S_ISDIR(opened.st_mode) || opened.st_dev != status.st_dev ||
                opened.st_ino != status.st_ino) {
                close(child);
                closedir(directory);
                return EvidenceError::kRace;
            }
            const EvidenceError nested = sum_fd(child, total);
            close(child);
            if (nested != EvidenceError::kOk) {
                closedir(directory);
                return nested;
            }
            continue;
        }
        if (const EvidenceError kind = classify_file(status); kind != EvidenceError::kOk) {
            closedir(directory);
            return kind;
        }
        const int file = openat(dir_fd, name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
        if (file < 0) {
            closedir(directory);
            return errno == ENOENT ? EvidenceError::kRace : from_errno(errno);
        }
        struct stat opened{};
        if (fstat(file, &opened) != 0) {
            close(file);
            closedir(directory);
            return EvidenceError::kIo;
        }
        close(file);
        if (const EvidenceError kind = classify_file(opened); kind != EvidenceError::kOk) {
            closedir(directory);
            return kind;
        }
        if (opened.st_dev != status.st_dev || opened.st_ino != status.st_ino ||
            opened.st_size != status.st_size) {
            closedir(directory);
            return EvidenceError::kRace;
        }
        std::uint64_t sum = 0;
        if (add_uint64(total, static_cast<std::uint64_t>(opened.st_size), sum) !=
            AdmissionError::kOk) {
            closedir(directory);
            return EvidenceError::kOverflow;
        }
        total = sum;
    }
}

EvidenceError ensure_lock_dir(int root_fd, Fd& out) {
    struct stat status{};
    if (fstatat(root_fd, "lock", &status, AT_SYMLINK_NOFOLLOW) != 0) {
        if (errno != ENOENT) {
            return from_errno(errno);
        }
        if (mkdirat(root_fd, "lock", 0700) != 0) {
            return from_errno(errno);
        }
    }
    const int fd = openat(root_fd, "lock", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) {
        return from_errno(errno);
    }
    if (fchmod(fd, 0700) != 0) {
        close(fd);
        return EvidenceError::kIo;
    }
    struct stat opened{};
    if (fstat(fd, &opened) != 0) {
        close(fd);
        return EvidenceError::kIo;
    }
    if (!S_ISDIR(opened.st_mode)) {
        close(fd);
        return EvidenceError::kType;
    }
    out = Fd(fd);
    return EvidenceError::kOk;
}

EvidenceError date_on_or_after(std::string_view date, std::string_view minimum) {
    if (const EvidenceError left = require_date(date); left != EvidenceError::kOk) {
        return left;
    }
    if (const EvidenceError right = require_date(minimum); right != EvidenceError::kOk) {
        return right;
    }
    return date < minimum ? EvidenceError::kDate : EvidenceError::kOk;
}

EvidenceError at_least_two_days(std::string_view acquired, std::string_view authorized) {
    std::string minimum;
    if (const EvidenceError status = from_admission(add_calendar_days(acquired, 2, minimum));
        status != EvidenceError::kOk) {
        return status;
    }
    return date_on_or_after(authorized, minimum);
}

EvidenceError read_relative(int root_fd, std::string_view relative, std::uint64_t limit,
                            std::string& out, struct stat& status) {
    Fd parent;
    std::string leaf;
    if (const EvidenceError walked = walk_parent(root_fd, relative, false, parent, leaf);
        walked != EvidenceError::kOk) {
        return walked;
    }
    const int fd = openat(parent.get(), leaf.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) {
        return from_errno(errno);
    }
    Fd file(fd);
    if (fstat(file.get(), &status) != 0) {
        return EvidenceError::kIo;
    }
    if (const EvidenceError kind = classify_file(status); kind != EvidenceError::kOk) {
        return kind;
    }
    return read_limited(file.get(), static_cast<std::uint64_t>(status.st_size), limit, out);
}

EvidenceError read_record_file(int root_fd, std::string_view relative, std::string& out) {
    struct stat status{};
    return read_relative(root_fd, relative, kMaxCanonicalRecordBytes, out, status);
}

std::string reservation_filename(std::string_view reservation_id) {
    return "lock/reservation-" + std::string(reservation_id) + ".record";
}

std::string stale_authorization_path(std::string_view authorization_id) {
    return "lock/stale-" + std::string(authorization_id) + ".record";
}

std::string deletion_authorization_path(std::string_view authorization_id) {
    return "deletion/deletion-" + std::string(authorization_id) + ".record";
}

std::string deletion_list_path(std::string_view reservation_id) {
    return "deletion/reservation-" + std::string(reservation_id) + ".files.record";
}

std::string pending_path(std::string_view authorization_id) {
    return "deletion/deletion-" + std::string(authorization_id) + ".pending.record";
}

std::string completion_path(std::string_view authorization_id) {
    return "deletion/deletion-" + std::string(authorization_id) + ".complete.record";
}

EvidenceError parse_entry_line(const std::string& line, ReservationEntry& entry) {
    if (line.compare(0, 6, "entry ") != 0) {
        return EvidenceError::kSchema;
    }
    std::vector<std::string> fields;
    std::size_t start = 0;
    const std::string body = line;
    while (start <= body.size()) {
        const std::size_t space = body.find(' ', start);
        if (space == std::string::npos) {
            fields.push_back(body.substr(start));
            break;
        }
        if (space == start) {
            return EvidenceError::kLexical;
        }
        fields.push_back(body.substr(start, space - start));
        start = space + 1;
    }
    if (fields.size() != 6 || fields[0] != "entry") {
        return EvidenceError::kSchema;
    }
    entry.staging_relative_path = fields[1];
    entry.final_relative_path = fields[2];
    if (const EvidenceError status = require_u64(fields[3], entry.reserved_object_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_sha(fields[4], false); status != EvidenceError::kOk) {
        return status;
    }
    entry.reserved_object_sha256 = fields[4];
    return parse_object_class(fields[5], entry.object_class);
}

EvidenceError validate_staging(std::string_view path, std::string_view owner) {
    if (const EvidenceError status = validate_evidence_relative_path(path);
        status != EvidenceError::kOk) {
        return status;
    }
    const std::vector<std::string_view> parts = split_segments(path);
    if (parts.size() != 3 || parts[0] != "partial" || parts[1] != owner) {
        return EvidenceError::kPath;
    }
    if (const EvidenceError status = require_id(owner); status != EvidenceError::kOk) {
        return status;
    }
    if (std::string(parts[2]) == "recipe-echo.bin") {
        return EvidenceError::kForbidden;
    }
    return from_admission(validate_filename(parts[2]));
}

EvidenceError validate_published_final(std::string_view path, std::uint64_t size) {
    if (path == "none" || path == "recipe-echo.bin") {
        return EvidenceError::kPath;
    }
    if (const EvidenceError status = validate_evidence_relative_path(path);
        status != EvidenceError::kOk) {
        return status;
    }
    const std::vector<std::string_view> parts = split_segments(path);
    if (parts[0] == "partial" || parts[0] == "lock" || parts[0] == "quarantine") {
        return EvidenceError::kForbidden;
    }
    const auto filename_ok = [&](std::string_view filename) {
        if (filename == "recipe-echo.bin") {
            return EvidenceError::kForbidden;
        }
        return from_admission(validate_filename(filename));
    };
    EvidenceError status = EvidenceError::kPath;
    if (parts.size() == 3 && (parts[0] == "provenance" || parts[0] == "classb" ||
                              parts[0] == "runs" || parts[0] == "golden")) {
        status = require_id(parts[1]);
        if (status == EvidenceError::kOk) {
            status = filename_ok(parts[2]);
        }
    } else if (parts.size() == 4 && parts[0] == "candidates" && parts[2] == "gguf") {
        status = require_id(parts[1]);
        if (status == EvidenceError::kOk) {
            status = filename_ok(parts[3]);
        }
        if (status == EvidenceError::kOk && size > 2147483648ull) {
            return EvidenceError::kSize;
        }
    } else if (parts.size() == 2 &&
               (parts[0] == "license" || parts[0] == "signatures" || parts[0] == "keys")) {
        status = require_id_bin(parts[1]);
    }
    return status;
}

EvidenceError validate_reservation_entries(std::string_view owner,
                                           const std::vector<ReservationEntry>& entries) {
    if (const EvidenceError status = require_id(owner); status != EvidenceError::kOk) {
        return status;
    }
    std::vector<std::string> staging;
    std::vector<std::string> finals;
    staging.reserve(entries.size());
    for (const ReservationEntry& entry : entries) {
        if (const EvidenceError status = validate_staging(entry.staging_relative_path, owner);
            status != EvidenceError::kOk) {
            return status;
        }
        if (const EvidenceError status = require_sha(entry.reserved_object_sha256, false);
            status != EvidenceError::kOk) {
            return status;
        }
        if (class_name(entry.object_class) == nullptr) {
            return EvidenceError::kSchema;
        }
        if (entry.object_class == EvidenceObjectClass::kTemporaryMetadata) {
            if (entry.final_relative_path != "none") {
                return EvidenceError::kPath;
            }
        } else if (const EvidenceError status = validate_published_final(
                       entry.final_relative_path, entry.reserved_object_size);
                   status != EvidenceError::kOk) {
            return status;
        }
        if (entry.final_relative_path == entry.staging_relative_path) {
            return EvidenceError::kPath;
        }
        staging.push_back(entry.staging_relative_path);
        finals.push_back(entry.final_relative_path);
    }
    std::vector<std::string> staging_sorted = staging;
    std::sort(staging_sorted.begin(), staging_sorted.end());
    if (std::adjacent_find(staging_sorted.begin(), staging_sorted.end()) != staging_sorted.end()) {
        return EvidenceError::kDuplicate;
    }
    std::vector<std::string> final_sorted = finals;
    std::sort(final_sorted.begin(), final_sorted.end());
    if (std::adjacent_find(final_sorted.begin(), final_sorted.end()) != final_sorted.end()) {
        return EvidenceError::kDuplicate;
    }
    for (const std::string& final_path : finals) {
        if (std::find(staging.begin(), staging.end(), final_path) != staging.end()) {
            return EvidenceError::kPath;
        }
    }
    return EvidenceError::kOk;
}

std::string entry_line(const ReservationEntry& entry) {
    std::string line = "entry ";
    line += entry.staging_relative_path;
    line.push_back(' ');
    line += entry.final_relative_path;
    line.push_back(' ');
    line += text_u64(entry.reserved_object_size);
    line.push_back(' ');
    line += entry.reserved_object_sha256;
    line.push_back(' ');
    line += class_name(entry.object_class);
    line.push_back('\n');
    return line;
}

std::string deletion_entry_line(const DeletionListEntry& entry) {
    std::string line = "entry ";
    line += entry.relative_path;
    line.push_back(' ');
    line += text_u64(entry.deletion_entry_size);
    line.push_back(' ');
    line += entry.deletion_entry_sha256;
    line.push_back(' ');
    line += cleanup_name(entry.object_class);
    line.push_back('\n');
    return line;
}

EvidenceError planned_bytes(const std::vector<ReservationEntry>& entries, std::uint64_t& total) {
    std::uint64_t sum = 0;
    for (const ReservationEntry& entry : entries) {
        std::uint64_t next = 0;
        if (add_uint64(sum, entry.reserved_object_size, next) != AdmissionError::kOk) {
            return EvidenceError::kOverflow;
        }
        sum = next;
    }
    total = sum;
    return EvidenceError::kOk;
}

EvidenceError cleanup_tracked(int root_fd, const std::vector<TrackedFile>& files) {
    EvidenceError result = EvidenceError::kOk;
    for (const TrackedFile& file : files) {
        struct stat status{};
        bool present = false;
        const EvidenceError checked = stat_leaf(root_fd, file.relative, status, present);
        if (checked != EvidenceError::kOk) {
            return checked;
        }
        if (!present) {
            continue;
        }
        if (static_cast<std::uint64_t>(status.st_dev) != file.device ||
            static_cast<std::uint64_t>(status.st_ino) != file.inode) {
            result = EvidenceError::kRace;
            continue;
        }
        if (const EvidenceError unlinked = unlink_tracked(root_fd, file);
            unlinked != EvidenceError::kOk) {
            return unlinked;
        }
    }
    return result;
}

EvidenceError open_created_lock(int lock_dir, std::string_view owner_run,
                                std::string_view owner_host, std::string_view acquired,
                                bool recovery, Fd& file, std::uint64_t& device,
                                std::uint64_t& inode) {
    const char* name = recovery ? "recovery.lock" : "evidence.lock";
    const int fd = openat(lock_dir, name, O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC | O_RDWR, 0600);
    if (fd < 0) {
        return from_errno(errno);
    }
    file = Fd(fd);
    if (fchmod(file.get(), 0600) != 0) {
        return EvidenceError::kIo;
    }
    std::string bytes;
    EvidenceError sealed = EvidenceError::kOk;
    if (recovery) {
        std::string prefix;
        append_kv(prefix, "schema_version", "1");
        append_kv(prefix, "record_kind", "recovery_lock");
        append_kv(prefix, "owner_run_id", owner_run);
        append_kv(prefix, "acquired_date", acquired);
        sealed = seal_prefix(prefix, bytes);
    } else {
        EvidenceLockRecord fields;
        fields.owner_run_id = std::string(owner_run);
        fields.owner_host_id = std::string(owner_host);
        fields.acquired_date = std::string(acquired);
        sealed = serialize_evidence_lock(fields, bytes);
    }
    if (sealed != EvidenceError::kOk) {
        return sealed;
    }
    if (const EvidenceError status = write_all(file.get(), bytes); status != EvidenceError::kOk) {
        return status;
    }
    struct stat status{};
    if (fstat(file.get(), &status) != 0) {
        return EvidenceError::kIo;
    }
    if (const EvidenceError kind = classify_file(status); kind != EvidenceError::kOk) {
        return kind;
    }
    if ((status.st_mode & 0777) != 0600) {
        return EvidenceError::kIo;
    }
    device = static_cast<std::uint64_t>(status.st_dev);
    inode = static_cast<std::uint64_t>(status.st_ino);
    if (fsync(lock_dir) != 0) {
        return EvidenceError::kIo;
    }
    return EvidenceError::kOk;
}

EvidenceError hash_descriptor_twice(int fd, std::uint64_t size, std::string& digest) {
    std::string first;
    if (const EvidenceError status = read_fd(fd, size, first); status != EvidenceError::kOk) {
        return status;
    }
    struct stat again{};
    if (fstat(fd, &again) != 0) {
        return EvidenceError::kIo;
    }
    if (static_cast<std::uint64_t>(again.st_size) != size) {
        return EvidenceError::kRace;
    }
    std::string second;
    if (const EvidenceError status = read_fd(fd, size, second); status != EvidenceError::kOk) {
        return status;
    }
    if (first != second) {
        return EvidenceError::kRace;
    }
    digest = digest_hex(first);
    first.swap(second);
    return EvidenceError::kOk;
}

EvidenceError unlink_descriptor_path(int dir_fd, const char* name, int fd, std::uint64_t device,
                                     std::uint64_t inode) {
    struct stat descriptor{};
    if (fstat(fd, &descriptor) != 0) {
        return EvidenceError::kIo;
    }
    if (static_cast<std::uint64_t>(descriptor.st_dev) != device ||
        static_cast<std::uint64_t>(descriptor.st_ino) != inode) {
        return EvidenceError::kRace;
    }
    struct stat path_status{};
    if (fstatat(dir_fd, name, &path_status, AT_SYMLINK_NOFOLLOW) != 0) {
        return from_errno(errno);
    }
    if (const EvidenceError kind = classify_file(path_status); kind != EvidenceError::kOk) {
        return kind;
    }
    if (static_cast<std::uint64_t>(path_status.st_dev) != device ||
        static_cast<std::uint64_t>(path_status.st_ino) != inode ||
        path_status.st_size != descriptor.st_size) {
        return EvidenceError::kRace;
    }
    if (unlinkat(dir_fd, name, 0) != 0) {
        return from_errno(errno);
    }
    if (fstat(fd, &descriptor) != 0) {
        return EvidenceError::kIo;
    }
    if (static_cast<std::uint64_t>(descriptor.st_dev) != device ||
        static_cast<std::uint64_t>(descriptor.st_ino) != inode || descriptor.st_nlink != 0) {
        return EvidenceError::kRace;
    }
    if (fsync(dir_fd) != 0) {
        return EvidenceError::kIo;
    }
    return EvidenceError::kOk;
}

} // namespace

const char* evidence_error_name(EvidenceError error) noexcept {
    switch (error) {
    case EvidenceError::kOk:
        return "ok";
    case EvidenceError::kLexical:
        return "lexical";
    case EvidenceError::kPath:
        return "path";
    case EvidenceError::kSymlink:
        return "symlink";
    case EvidenceError::kType:
        return "type";
    case EvidenceError::kLink:
        return "link";
    case EvidenceError::kSpecial:
        return "special";
    case EvidenceError::kCap:
        return "cap";
    case EvidenceError::kOverflow:
        return "overflow";
    case EvidenceError::kLock:
        return "lock";
    case EvidenceError::kOwner:
        return "owner";
    case EvidenceError::kSize:
        return "size";
    case EvidenceError::kDigest:
        return "digest";
    case EvidenceError::kChecksum:
        return "checksum";
    case EvidenceError::kSchema:
        return "schema";
    case EvidenceError::kDuplicate:
        return "duplicate";
    case EvidenceError::kAuthorization:
        return "authorization";
    case EvidenceError::kUnavailable:
        return "unavailable";
    case EvidenceError::kDate:
        return "date";
    case EvidenceError::kRace:
        return "race";
    case EvidenceError::kIo:
        return "io";
    case EvidenceError::kPresent:
        return "present";
    case EvidenceError::kAbsent:
        return "absent";
    case EvidenceError::kState:
        return "state";
    case EvidenceError::kForbidden:
        return "forbidden";
    case EvidenceError::kInternal:
        return "internal";
    }
    return "internal";
}

EvidenceError BatchProbe::after_staging_written(std::string_view) {
    return EvidenceError::kOk;
}

EvidenceLock::EvidenceLock(EvidenceLock&& other) noexcept {
    *this = std::move(other);
}

EvidenceLock& EvidenceLock::operator=(EvidenceLock&& other) noexcept {
    if (this != &other) {
        reset();
        root_fd_ = other.root_fd_;
        lock_fd_ = other.lock_fd_;
        lock_dir_fd_ = other.lock_dir_fd_;
        device_ = other.device_;
        inode_ = other.inode_;
        owner_run_id_ = std::move(other.owner_run_id_);
        owner_host_id_ = std::move(other.owner_host_id_);
        evidence_root_ = std::move(other.evidence_root_);
        held_ = other.held_;
        other.root_fd_ = -1;
        other.lock_fd_ = -1;
        other.lock_dir_fd_ = -1;
        other.device_ = 0;
        other.inode_ = 0;
        other.held_ = false;
    }
    return *this;
}

EvidenceLock::~EvidenceLock() {
    reset();
}

void EvidenceLock::adopt(int root_fd, int lock_dir_fd, int lock_fd, std::uint64_t device,
                         std::uint64_t inode, std::string owner_run_id, std::string owner_host_id,
                         std::string evidence_root) {
    reset();
    root_fd_ = root_fd;
    lock_dir_fd_ = lock_dir_fd;
    lock_fd_ = lock_fd;
    device_ = device;
    inode_ = inode;
    owner_run_id_ = std::move(owner_run_id);
    owner_host_id_ = std::move(owner_host_id);
    evidence_root_ = std::move(evidence_root);
    held_ = true;
}

void EvidenceLock::reset() noexcept {
    if (lock_fd_ >= 0) {
        close(lock_fd_);
    }
    if (lock_dir_fd_ >= 0) {
        close(lock_dir_fd_);
    }
    if (root_fd_ >= 0) {
        close(root_fd_);
    }
    lock_fd_ = -1;
    lock_dir_fd_ = -1;
    root_fd_ = -1;
    device_ = 0;
    inode_ = 0;
    owner_run_id_.clear();
    owner_host_id_.clear();
    evidence_root_.clear();
    held_ = false;
}

EvidenceError EvidenceLock::release() {
    if (!held_ || lock_fd_ < 0 || lock_dir_fd_ < 0) {
        return EvidenceError::kLock;
    }
    struct stat descriptor{};
    if (fstat(lock_fd_, &descriptor) != 0) {
        return EvidenceError::kIo;
    }
    if (static_cast<std::uint64_t>(descriptor.st_dev) != device_ ||
        static_cast<std::uint64_t>(descriptor.st_ino) != inode_) {
        return EvidenceError::kRace;
    }
    struct stat path_status{};
    if (fstatat(lock_dir_fd_, "evidence.lock", &path_status, AT_SYMLINK_NOFOLLOW) != 0) {
        return from_errno(errno);
    }
    if (static_cast<std::uint64_t>(path_status.st_dev) != device_ ||
        static_cast<std::uint64_t>(path_status.st_ino) != inode_) {
        return EvidenceError::kRace;
    }
    if (const EvidenceError kind = classify_file(descriptor); kind != EvidenceError::kOk) {
        return kind;
    }
    std::string bytes;
    if (const EvidenceError status =
            read_fd(lock_fd_, static_cast<std::uint64_t>(descriptor.st_size), bytes);
        status != EvidenceError::kOk) {
        return status;
    }
    EvidenceLockRecord record;
    if (const EvidenceError status = parse_evidence_lock(bytes, record);
        status != EvidenceError::kOk) {
        return status;
    }
    if (record.owner_run_id != owner_run_id_ || record.owner_host_id != owner_host_id_) {
        return EvidenceError::kOwner;
    }
    if (const EvidenceError status =
            unlink_descriptor_path(lock_dir_fd_, "evidence.lock", lock_fd_, device_, inode_);
        status != EvidenceError::kOk) {
        return status;
    }
    reset();
    return EvidenceError::kOk;
}

EvidenceError validate_evidence_relative_path(std::string_view path) {
    if (path.empty() || path.front() == '/' || path.size() > 4096 ||
        contains_forbidden_byte(path)) {
        return EvidenceError::kPath;
    }
    return check_segments(split_segments(path));
}

EvidenceError reservation_fits_cap(std::uint64_t sum_before, std::uint64_t reservation_record_size,
                                   std::uint64_t planned_batch_bytes) {
    std::uint64_t batch = 0;
    if (add_uint64(reservation_record_size, planned_batch_bytes, batch) != AdmissionError::kOk) {
        return EvidenceError::kOverflow;
    }
    std::uint64_t total = 0;
    if (add_uint64(sum_before, batch, total) != AdmissionError::kOk) {
        return EvidenceError::kOverflow;
    }
    return total > kEvidenceCapBytes ? EvidenceError::kCap : EvidenceError::kOk;
}

EvidenceError sum_evidence_root(std::string_view evidence_root, std::uint64_t& total) {
    Fd root;
    if (const EvidenceError status = open_root(evidence_root, root); status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t sum = 0;
    if (const EvidenceError status = sum_fd(root.get(), sum); status != EvidenceError::kOk) {
        return status;
    }
    total = sum;
    return EvidenceError::kOk;
}

EvidenceError serialize_evidence_lock(const EvidenceLockRecord& fields, std::string& out) {
    if (const EvidenceError status = require_id(fields.owner_run_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(fields.owner_host_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_date(fields.acquired_date);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string prefix;
    append_kv(prefix, "schema_version", "1");
    append_kv(prefix, "record_kind", "evidence_lock");
    append_kv(prefix, "owner_run_id", fields.owner_run_id);
    append_kv(prefix, "owner_host_id", fields.owner_host_id);
    append_kv(prefix, "acquired_date", fields.acquired_date);
    return seal_prefix(prefix, out);
}

EvidenceError parse_evidence_lock(std::string_view bytes, EvidenceLockRecord& out) {
    std::vector<std::string> lines;
    if (const EvidenceError status = split_canonical(bytes, lines); status != EvidenceError::kOk) {
        return status;
    }
    if (lines.size() != 5) {
        return EvidenceError::kSchema;
    }
    EvidenceLockRecord record;
    if (const EvidenceError status = require_exact(lines, 0, "schema_version", "1");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_exact(lines, 1, "record_kind", "evidence_lock");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 2, "owner_run_id", record.owner_run_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 3, "owner_host_id", record.owner_host_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 4, "acquired_date", record.acquired_date);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(record.owner_run_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(record.owner_host_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_date(record.acquired_date);
        status != EvidenceError::kOk) {
        return status;
    }
    record.bytes = std::string(bytes);
    out = std::move(record);
    return EvidenceError::kOk;
}

EvidenceError parse_recovery_lock(std::string_view bytes, RecoveryLockRecord& out) {
    std::vector<std::string> lines;
    if (const EvidenceError status = split_canonical(bytes, lines); status != EvidenceError::kOk) {
        return status;
    }
    if (lines.size() != 4) {
        return EvidenceError::kSchema;
    }
    RecoveryLockRecord record;
    if (const EvidenceError status = require_exact(lines, 0, "schema_version", "1");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_exact(lines, 1, "record_kind", "recovery_lock");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 2, "owner_run_id", record.owner_run_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 3, "acquired_date", record.acquired_date);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(record.owner_run_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_date(record.acquired_date);
        status != EvidenceError::kOk) {
        return status;
    }
    record.bytes = std::string(bytes);
    out = std::move(record);
    return EvidenceError::kOk;
}

EvidenceError serialize_evidence_reservation(std::string_view owner_run_id,
                                             const std::vector<ReservationEntry>& entries,
                                             EvidenceReservation& out) {
    if (const EvidenceError status = validate_reservation_entries(owner_run_id, entries);
        status != EvidenceError::kOk) {
        return status;
    }
    std::vector<ReservationEntry> ordered = entries;
    std::sort(ordered.begin(), ordered.end(),
              [](const ReservationEntry& left, const ReservationEntry& right) {
                  return left.staging_relative_path < right.staging_relative_path;
              });
    std::uint64_t planned = 0;
    if (const EvidenceError status = planned_bytes(ordered, planned);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string entries_blob;
    for (const ReservationEntry& entry : ordered) {
        entries_blob += entry_line(entry);
    }
    const std::string head = "schema_version=1\nrecord_kind=evidence_reservation\nowner_run_id=" +
                             std::string(owner_run_id) + "\n";
    const std::string count = "path_count=" + text_u64(ordered.size()) + "\n";
    bool found = false;
    std::string bytes;
    std::uint64_t reserved = 0;
    for (int width = 1; width <= 20; ++width) {
        const std::uint64_t prefix = static_cast<std::uint64_t>(
            head.size() + std::strlen("reserved_bytes=") + static_cast<std::size_t>(width) + 1 +
            count.size() + entries_blob.size());
        std::uint64_t record_size = 0;
        if (add_uint64(prefix, 79, record_size) != AdmissionError::kOk) {
            return EvidenceError::kOverflow;
        }
        std::uint64_t candidate = 0;
        if (add_uint64(record_size, planned, candidate) != AdmissionError::kOk) {
            continue;
        }
        if (decimal_width(candidate) != width) {
            continue;
        }
        std::string prefix_bytes = head;
        append_kv(prefix_bytes, "reserved_bytes", text_u64(candidate));
        prefix_bytes += count;
        prefix_bytes += entries_blob;
        std::string sealed;
        if (const EvidenceError status = seal_prefix(prefix_bytes, sealed);
            status != EvidenceError::kOk) {
            return status;
        }
        if (sealed.size() != record_size) {
            return EvidenceError::kInternal;
        }
        if (found) {
            return EvidenceError::kInternal;
        }
        found = true;
        bytes = std::move(sealed);
        reserved = candidate;
    }
    if (!found) {
        return EvidenceError::kSize;
    }
    EvidenceReservation reservation;
    if (const EvidenceError status = parse_evidence_reservation(bytes, reservation);
        status != EvidenceError::kOk) {
        return status;
    }
    if (reservation.reserved_bytes != reserved) {
        return EvidenceError::kInternal;
    }
    out = std::move(reservation);
    return EvidenceError::kOk;
}

EvidenceError parse_evidence_reservation(std::string_view bytes, EvidenceReservation& out) {
    std::vector<std::string> lines;
    if (const EvidenceError status = split_canonical(bytes, lines); status != EvidenceError::kOk) {
        return status;
    }
    if (lines.size() < 5) {
        return EvidenceError::kSchema;
    }
    EvidenceReservation reservation;
    if (const EvidenceError status = require_exact(lines, 0, "schema_version", "1");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_exact(lines, 1, "record_kind", "evidence_reservation");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 2, "owner_run_id", reservation.owner_run_id);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string reserved_text;
    std::string count_text;
    if (const EvidenceError status = require_field(lines, 3, "reserved_bytes", reserved_text);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 4, "path_count", count_text);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_u64(reserved_text, reservation.reserved_bytes);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_u64(count_text, reservation.path_count);
        status != EvidenceError::kOk) {
        return status;
    }
    for (std::size_t index = 5; index < lines.size(); ++index) {
        ReservationEntry entry;
        if (const EvidenceError status = parse_entry_line(lines[index], entry);
            status != EvidenceError::kOk) {
            return status;
        }
        if (!reservation.entries.empty() &&
            entry.staging_relative_path <= reservation.entries.back().staging_relative_path) {
            return entry.staging_relative_path == reservation.entries.back().staging_relative_path
                       ? EvidenceError::kDuplicate
                       : EvidenceError::kSchema;
        }
        reservation.entries.push_back(std::move(entry));
    }
    if (reservation.path_count != reservation.entries.size()) {
        return EvidenceError::kSchema;
    }
    if (const EvidenceError status =
            validate_reservation_entries(reservation.owner_run_id, reservation.entries);
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t planned = 0;
    if (const EvidenceError status = planned_bytes(reservation.entries, planned);
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t expected = 0;
    if (add_uint64(bytes.size(), planned, expected) != AdmissionError::kOk) {
        return EvidenceError::kOverflow;
    }
    if (reservation.reserved_bytes != expected) {
        return EvidenceError::kSize;
    }
    reservation.bytes = std::string(bytes);
    out = std::move(reservation);
    return EvidenceError::kOk;
}

EvidenceError serialize_stale_lock_authorization(const StaleLockAuthorization& fields,
                                                 std::string& out) {
    if (const EvidenceError status = require_id(fields.authorization_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (fields.lock_path != "lock/evidence.lock" && fields.lock_path != "lock/recovery.lock") {
        return EvidenceError::kPath;
    }
    if (const EvidenceError status = require_id(fields.lock_owner_run_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_sha(fields.lock_file_sha256, false);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_date(fields.acquired_date);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_date(fields.authorization_date);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string prefix;
    append_kv(prefix, "schema_version", "1");
    append_kv(prefix, "record_kind", "stale_lock_authorization");
    append_kv(prefix, "authorization_id", fields.authorization_id);
    append_kv(prefix, "lock_path", fields.lock_path);
    append_kv(prefix, "lock_owner_run_id", fields.lock_owner_run_id);
    append_kv(prefix, "lock_file_size", text_u64(fields.lock_file_size));
    append_kv(prefix, "lock_file_sha256", fields.lock_file_sha256);
    append_kv(prefix, "acquired_date", fields.acquired_date);
    append_kv(prefix, "authorization_date", fields.authorization_date);
    append_kv(prefix, "test_only", "no");
    return seal_prefix(prefix, out);
}

EvidenceError parse_stale_lock_authorization(std::string_view bytes, StaleLockAuthorization& out) {
    std::vector<std::string> lines;
    if (const EvidenceError status = split_canonical(bytes, lines); status != EvidenceError::kOk) {
        return status;
    }
    if (lines.size() != 10) {
        return EvidenceError::kSchema;
    }
    StaleLockAuthorization record;
    if (const EvidenceError status = require_exact(lines, 0, "schema_version", "1");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_exact(lines, 1, "record_kind", "stale_lock_authorization");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 2, "authorization_id", record.authorization_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 3, "lock_path", record.lock_path);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 4, "lock_owner_run_id", record.lock_owner_run_id);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string size_text;
    if (const EvidenceError status = require_field(lines, 5, "lock_file_size", size_text);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 6, "lock_file_sha256", record.lock_file_sha256);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 7, "acquired_date", record.acquired_date);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 8, "authorization_date", record.authorization_date);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_exact(lines, 9, "test_only", "no");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(record.authorization_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (record.lock_path != "lock/evidence.lock" && record.lock_path != "lock/recovery.lock") {
        return EvidenceError::kPath;
    }
    if (const EvidenceError status = require_id(record.lock_owner_run_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_u64(size_text, record.lock_file_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_sha(record.lock_file_sha256, false);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_date(record.acquired_date);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_date(record.authorization_date);
        status != EvidenceError::kOk) {
        return status;
    }
    record.bytes = std::string(bytes);
    out = std::move(record);
    return EvidenceError::kOk;
}

EvidenceError serialize_reservation_deletion(const ReservationDeletionAuthorization& fields,
                                             std::string& out) {
    if (const EvidenceError status = require_id(fields.authorization_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(fields.target_id); status != EvidenceError::kOk) {
        return status;
    }
    if (fields.reservation_path != reservation_filename(fields.target_id)) {
        return EvidenceError::kPath;
    }
    if (const EvidenceError status = require_sha(fields.target_file_sha256, false);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(fields.owner_run_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(fields.deletion_list_id);
        status != EvidenceError::kOk) {
        return status;
    }
    const std::string expected_list = "admission/" + deletion_list_path(fields.target_id);
    if (fields.deletion_list_path != expected_list) {
        return EvidenceError::kPath;
    }
    if (const EvidenceError status = require_sha(fields.deletion_list_file_sha256, false);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_date(fields.authorization_date);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string prefix;
    append_kv(prefix, "schema_version", "1");
    append_kv(prefix, "record_kind", "deletion_authorization");
    append_kv(prefix, "authorization_id", fields.authorization_id);
    append_kv(prefix, "target_kind", "evidence_reservation");
    append_kv(prefix, "target_id", fields.target_id);
    append_kv(prefix, "reservation_path", fields.reservation_path);
    append_kv(prefix, "target_file_size", text_u64(fields.target_file_size));
    append_kv(prefix, "target_file_sha256", fields.target_file_sha256);
    append_kv(prefix, "owner_run_id", fields.owner_run_id);
    append_kv(prefix, "deletion_list_id", fields.deletion_list_id);
    append_kv(prefix, "deletion_list_path", fields.deletion_list_path);
    append_kv(prefix, "deletion_list_file_size", text_u64(fields.deletion_list_file_size));
    append_kv(prefix, "deletion_list_file_sha256", fields.deletion_list_file_sha256);
    append_kv(prefix, "authorization_date", fields.authorization_date);
    append_kv(prefix, "test_only", "no");
    return seal_prefix(prefix, out);
}

EvidenceError parse_reservation_deletion(std::string_view bytes,
                                         ReservationDeletionAuthorization& out) {
    std::vector<std::string> lines;
    if (const EvidenceError status = split_canonical(bytes, lines); status != EvidenceError::kOk) {
        return status;
    }
    if (lines.size() != 15) {
        return EvidenceError::kSchema;
    }
    ReservationDeletionAuthorization record;
    if (const EvidenceError status = require_exact(lines, 0, "schema_version", "1");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_exact(lines, 1, "record_kind", "deletion_authorization");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 2, "authorization_id", record.authorization_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_exact(lines, 3, "target_kind", "evidence_reservation");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 4, "target_id", record.target_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 5, "reservation_path", record.reservation_path);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string target_size;
    if (const EvidenceError status = require_field(lines, 6, "target_file_size", target_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 7, "target_file_sha256", record.target_file_sha256);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 8, "owner_run_id", record.owner_run_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 9, "deletion_list_id", record.deletion_list_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 10, "deletion_list_path", record.deletion_list_path);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string list_size;
    if (const EvidenceError status = require_field(lines, 11, "deletion_list_file_size", list_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 12, "deletion_list_file_sha256", record.deletion_list_file_sha256);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 13, "authorization_date", record.authorization_date);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_exact(lines, 14, "test_only", "no");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_u64(target_size, record.target_file_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_u64(list_size, record.deletion_list_file_size);
        status != EvidenceError::kOk) {
        return status;
    }
    record.bytes = std::string(bytes);
    std::string ignored;
    if (const EvidenceError status = serialize_reservation_deletion(record, ignored);
        status != EvidenceError::kOk) {
        return status;
    }
    if (ignored != bytes) {
        return EvidenceError::kSchema;
    }
    out = std::move(record);
    return EvidenceError::kOk;
}

EvidenceError serialize_quarantine_deletion(const QuarantineDeletionAuthorization& fields,
                                            std::string& out) {
    if (const EvidenceError status = require_id(fields.authorization_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(fields.target_id); status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_sha(fields.target_file_sha256, false);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_date(fields.authorization_date);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string prefix;
    append_kv(prefix, "schema_version", "1");
    append_kv(prefix, "record_kind", "deletion_authorization");
    append_kv(prefix, "authorization_id", fields.authorization_id);
    append_kv(prefix, "target_kind", "quarantine_metadata");
    append_kv(prefix, "target_id", fields.target_id);
    append_kv(prefix, "target_file_size", text_u64(fields.target_file_size));
    append_kv(prefix, "target_file_sha256", fields.target_file_sha256);
    append_kv(prefix, "authorization_date", fields.authorization_date);
    append_kv(prefix, "test_only", "no");
    return seal_prefix(prefix, out);
}

EvidenceError parse_quarantine_deletion(std::string_view bytes,
                                        QuarantineDeletionAuthorization& out) {
    std::vector<std::string> lines;
    if (const EvidenceError status = split_canonical(bytes, lines); status != EvidenceError::kOk) {
        return status;
    }
    if (lines.size() != 9) {
        return EvidenceError::kSchema;
    }
    QuarantineDeletionAuthorization record;
    if (const EvidenceError status = require_exact(lines, 0, "schema_version", "1");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_exact(lines, 1, "record_kind", "deletion_authorization");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 2, "authorization_id", record.authorization_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_exact(lines, 3, "target_kind", "quarantine_metadata");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 4, "target_id", record.target_id);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string size_text;
    if (const EvidenceError status = require_field(lines, 5, "target_file_size", size_text);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 6, "target_file_sha256", record.target_file_sha256);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 7, "authorization_date", record.authorization_date);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_exact(lines, 8, "test_only", "no");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_u64(size_text, record.target_file_size);
        status != EvidenceError::kOk) {
        return status;
    }
    record.bytes = std::string(bytes);
    std::string ignored;
    if (const EvidenceError status = serialize_quarantine_deletion(record, ignored);
        status != EvidenceError::kOk) {
        return status;
    }
    if (ignored != bytes) {
        return EvidenceError::kSchema;
    }
    out = std::move(record);
    return EvidenceError::kOk;
}

EvidenceError serialize_reservation_deletion_list(const ReservationDeletionList& fields,
                                                  std::string& out) {
    if (const EvidenceError status = require_id(fields.list_id); status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(fields.reservation_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(fields.owner_run_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (fields.entry_count != fields.entries.size()) {
        return EvidenceError::kSchema;
    }
    std::vector<DeletionListEntry> ordered = fields.entries;
    std::sort(ordered.begin(), ordered.end(),
              [](const DeletionListEntry& left, const DeletionListEntry& right) {
                  return left.relative_path < right.relative_path;
              });
    std::string prefix;
    append_kv(prefix, "schema_version", "1");
    append_kv(prefix, "record_kind", "reservation_deletion_list");
    append_kv(prefix, "list_id", fields.list_id);
    append_kv(prefix, "reservation_id", fields.reservation_id);
    append_kv(prefix, "owner_run_id", fields.owner_run_id);
    append_kv(prefix, "entry_count", text_u64(ordered.size()));
    std::string previous;
    for (const DeletionListEntry& entry : ordered) {
        if (entry.relative_path == previous) {
            return EvidenceError::kDuplicate;
        }
        if (cleanup_name(entry.object_class) == nullptr) {
            return EvidenceError::kSchema;
        }
        if (const EvidenceError status = require_sha(entry.deletion_entry_sha256, false);
            status != EvidenceError::kOk) {
            return status;
        }
        previous = entry.relative_path;
        prefix += deletion_entry_line(entry);
    }
    return seal_prefix(prefix, out);
}

EvidenceError parse_reservation_deletion_list(std::string_view bytes,
                                              ReservationDeletionList& out) {
    std::vector<std::string> lines;
    if (const EvidenceError status = split_canonical(bytes, lines); status != EvidenceError::kOk) {
        return status;
    }
    if (lines.size() < 6) {
        return EvidenceError::kSchema;
    }
    ReservationDeletionList list;
    if (const EvidenceError status = require_exact(lines, 0, "schema_version", "1");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_exact(lines, 1, "record_kind", "reservation_deletion_list");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 2, "list_id", list.list_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 3, "reservation_id", list.reservation_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 4, "owner_run_id", list.owner_run_id);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string count_text;
    if (const EvidenceError status = require_field(lines, 5, "entry_count", count_text);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_u64(count_text, list.entry_count);
        status != EvidenceError::kOk) {
        return status;
    }
    for (std::size_t index = 6; index < lines.size(); ++index) {
        std::vector<std::string> fields;
        std::size_t start = 0;
        while (start <= lines[index].size()) {
            const std::size_t space = lines[index].find(' ', start);
            if (space == std::string::npos) {
                fields.push_back(lines[index].substr(start));
                break;
            }
            if (space == start) {
                return EvidenceError::kLexical;
            }
            fields.push_back(lines[index].substr(start, space - start));
            start = space + 1;
        }
        if (fields.size() != 5 || fields[0] != "entry") {
            return EvidenceError::kSchema;
        }
        DeletionListEntry entry;
        entry.relative_path = fields[1];
        if (const EvidenceError status = require_u64(fields[2], entry.deletion_entry_size);
            status != EvidenceError::kOk) {
            return status;
        }
        if (const EvidenceError status = require_sha(fields[3], false);
            status != EvidenceError::kOk) {
            return status;
        }
        entry.deletion_entry_sha256 = fields[3];
        if (const EvidenceError status = parse_cleanup_class(fields[4], entry.object_class);
            status != EvidenceError::kOk) {
            return status;
        }
        if (!list.entries.empty() && entry.relative_path <= list.entries.back().relative_path) {
            return entry.relative_path == list.entries.back().relative_path
                       ? EvidenceError::kDuplicate
                       : EvidenceError::kSchema;
        }
        list.entries.push_back(std::move(entry));
    }
    if (list.entry_count != list.entries.size()) {
        return EvidenceError::kSchema;
    }
    list.bytes = std::string(bytes);
    std::string ignored;
    if (const EvidenceError status = serialize_reservation_deletion_list(list, ignored);
        status != EvidenceError::kOk) {
        return status;
    }
    if (ignored != bytes) {
        return EvidenceError::kSchema;
    }
    out = std::move(list);
    return EvidenceError::kOk;
}

EvidenceError serialize_cleanup_pending(const CleanupPendingRecord& fields, std::string& out) {
    if (const EvidenceError status = require_id(fields.authorization_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_sha(fields.authorization_file_sha256, false);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(fields.deletion_list_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_sha(fields.deletion_list_file_sha256, false);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(fields.owner_run_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_date(fields.creation_date);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string prefix;
    append_kv(prefix, "schema_version", "1");
    append_kv(prefix, "record_kind", "cleanup_pending");
    append_kv(prefix, "authorization_id", fields.authorization_id);
    append_kv(prefix, "authorization_file_size", text_u64(fields.authorization_file_size));
    append_kv(prefix, "authorization_file_sha256", fields.authorization_file_sha256);
    append_kv(prefix, "deletion_list_id", fields.deletion_list_id);
    append_kv(prefix, "deletion_list_file_size", text_u64(fields.deletion_list_file_size));
    append_kv(prefix, "deletion_list_file_sha256", fields.deletion_list_file_sha256);
    append_kv(prefix, "owner_run_id", fields.owner_run_id);
    append_kv(prefix, "creation_date", fields.creation_date);
    append_kv(prefix, "state", "deletion_pending");
    return seal_prefix(prefix, out);
}

EvidenceError parse_cleanup_pending(std::string_view bytes, CleanupPendingRecord& out) {
    std::vector<std::string> lines;
    if (const EvidenceError status = split_canonical(bytes, lines); status != EvidenceError::kOk) {
        return status;
    }
    if (lines.size() != 11) {
        return EvidenceError::kSchema;
    }
    CleanupPendingRecord record;
    if (const EvidenceError status = require_exact(lines, 0, "schema_version", "1");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_exact(lines, 1, "record_kind", "cleanup_pending");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 2, "authorization_id", record.authorization_id);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string authorization_size;
    if (const EvidenceError status =
            require_field(lines, 3, "authorization_file_size", authorization_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 4, "authorization_file_sha256", record.authorization_file_sha256);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 5, "deletion_list_id", record.deletion_list_id);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string list_size;
    if (const EvidenceError status = require_field(lines, 6, "deletion_list_file_size", list_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 7, "deletion_list_file_sha256", record.deletion_list_file_sha256);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 8, "owner_run_id", record.owner_run_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 9, "creation_date", record.creation_date);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_exact(lines, 10, "state", "deletion_pending");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_u64(authorization_size, record.authorization_file_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_u64(list_size, record.deletion_list_file_size);
        status != EvidenceError::kOk) {
        return status;
    }
    record.bytes = std::string(bytes);
    std::string ignored;
    if (const EvidenceError status = serialize_cleanup_pending(record, ignored);
        status != EvidenceError::kOk) {
        return status;
    }
    if (ignored != bytes) {
        return EvidenceError::kSchema;
    }
    out = std::move(record);
    return EvidenceError::kOk;
}

EvidenceError serialize_cleanup_completion(const CleanupCompletionRecord& fields,
                                           std::string& out) {
    if (const EvidenceError status = require_id(fields.authorization_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_sha(fields.pending_file_sha256, false);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_sha(fields.authorization_file_sha256, false);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(fields.deletion_list_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_sha(fields.deletion_list_file_sha256, false);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(fields.owner_run_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_date(fields.completion_date);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string prefix;
    append_kv(prefix, "schema_version", "1");
    append_kv(prefix, "record_kind", "cleanup_completion");
    append_kv(prefix, "authorization_id", fields.authorization_id);
    append_kv(prefix, "pending_file_size", text_u64(fields.pending_file_size));
    append_kv(prefix, "pending_file_sha256", fields.pending_file_sha256);
    append_kv(prefix, "authorization_file_size", text_u64(fields.authorization_file_size));
    append_kv(prefix, "authorization_file_sha256", fields.authorization_file_sha256);
    append_kv(prefix, "deletion_list_id", fields.deletion_list_id);
    append_kv(prefix, "deletion_list_file_size", text_u64(fields.deletion_list_file_size));
    append_kv(prefix, "deletion_list_file_sha256", fields.deletion_list_file_sha256);
    append_kv(prefix, "owner_run_id", fields.owner_run_id);
    append_kv(prefix, "completion_date", fields.completion_date);
    append_kv(prefix, "state", "deletion_complete");
    return seal_prefix(prefix, out);
}

EvidenceError parse_cleanup_completion(std::string_view bytes, CleanupCompletionRecord& out) {
    std::vector<std::string> lines;
    if (const EvidenceError status = split_canonical(bytes, lines); status != EvidenceError::kOk) {
        return status;
    }
    if (lines.size() != 13) {
        return EvidenceError::kSchema;
    }
    CleanupCompletionRecord record;
    if (const EvidenceError status = require_exact(lines, 0, "schema_version", "1");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_exact(lines, 1, "record_kind", "cleanup_completion");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 2, "authorization_id", record.authorization_id);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string pending_size;
    if (const EvidenceError status = require_field(lines, 3, "pending_file_size", pending_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 4, "pending_file_sha256", record.pending_file_sha256);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string authorization_size;
    if (const EvidenceError status =
            require_field(lines, 5, "authorization_file_size", authorization_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 6, "authorization_file_sha256", record.authorization_file_sha256);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 7, "deletion_list_id", record.deletion_list_id);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string list_size;
    if (const EvidenceError status = require_field(lines, 8, "deletion_list_file_size", list_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 9, "deletion_list_file_sha256", record.deletion_list_file_sha256);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 10, "owner_run_id", record.owner_run_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 11, "completion_date", record.completion_date);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_exact(lines, 12, "state", "deletion_complete");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_u64(pending_size, record.pending_file_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_u64(authorization_size, record.authorization_file_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_u64(list_size, record.deletion_list_file_size);
        status != EvidenceError::kOk) {
        return status;
    }
    record.bytes = std::string(bytes);
    std::string ignored;
    if (const EvidenceError status = serialize_cleanup_completion(record, ignored);
        status != EvidenceError::kOk) {
        return status;
    }
    if (ignored != bytes) {
        return EvidenceError::kSchema;
    }
    out = std::move(record);
    return EvidenceError::kOk;
}

EvidenceError serialize_quarantine_metadata(const QuarantineMetadata& fields, std::string& out) {
    if (const EvidenceError status = require_id(fields.quarantine_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (fields.associated_run_id != fields.quarantine_id ||
        fields.directory_id != "quarantine/" + fields.quarantine_id) {
        return EvidenceError::kSchema;
    }
    if (const EvidenceError status = from_admission(validate_phase_id(fields.associated_phase_id));
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_date(fields.creation_date);
        status != EvidenceError::kOk) {
        return status;
    }
    if (!known_reason(fields.reason_code)) {
        return EvidenceError::kLexical;
    }
    std::string deadline;
    if (const EvidenceError status =
            from_admission(add_calendar_days(fields.creation_date, 30, deadline));
        status != EvidenceError::kOk) {
        return status;
    }
    if (fields.retention_start != fields.creation_date || fields.retention_deadline != deadline) {
        return EvidenceError::kDate;
    }
    if (fields.accepted_run_manifest_path == "none") {
        if (fields.accepted_run_manifest_file_size != 0 ||
            !is_zero_digest(fields.accepted_run_manifest_file_sha256)) {
            return EvidenceError::kLexical;
        }
    } else {
        if (const EvidenceError status = validate_published_final(
                fields.accepted_run_manifest_path, fields.accepted_run_manifest_file_size);
            status != EvidenceError::kOk) {
            return status;
        }
        if (is_zero_digest(fields.accepted_run_manifest_file_sha256)) {
            return EvidenceError::kLexical;
        }
        if (const EvidenceError status =
                require_sha(fields.accepted_run_manifest_file_sha256, false);
            status != EvidenceError::kOk) {
            return status;
        }
    }
    std::string prefix;
    append_kv(prefix, "schema_version", "1");
    append_kv(prefix, "record_kind", "quarantine_metadata");
    append_kv(prefix, "quarantine_id", fields.quarantine_id);
    append_kv(prefix, "associated_phase_id", fields.associated_phase_id);
    append_kv(prefix, "associated_run_id", fields.associated_run_id);
    append_kv(prefix, "directory_id", fields.directory_id);
    append_kv(prefix, "creation_date", fields.creation_date);
    append_kv(prefix, "reason_code", fields.reason_code);
    append_kv(prefix, "accepted_run_manifest_path", fields.accepted_run_manifest_path);
    append_kv(prefix, "accepted_run_manifest_file_size",
              text_u64(fields.accepted_run_manifest_file_size));
    append_kv(prefix, "accepted_run_manifest_file_sha256",
              fields.accepted_run_manifest_file_sha256);
    append_kv(prefix, "retention_start", fields.retention_start);
    append_kv(prefix, "retention_deadline", fields.retention_deadline);
    append_kv(prefix, "deletion_role", "trust_admin");
    append_kv(prefix, "state", "open");
    return seal_prefix(prefix, out);
}

EvidenceError parse_quarantine_metadata(std::string_view bytes, QuarantineMetadata& out) {
    std::vector<std::string> lines;
    if (const EvidenceError status = split_canonical(bytes, lines); status != EvidenceError::kOk) {
        return status;
    }
    if (lines.size() != 15) {
        return EvidenceError::kSchema;
    }
    QuarantineMetadata record;
    if (const EvidenceError status = require_exact(lines, 0, "schema_version", "1");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_exact(lines, 1, "record_kind", "quarantine_metadata");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 2, "quarantine_id", record.quarantine_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 3, "associated_phase_id", record.associated_phase_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 4, "associated_run_id", record.associated_run_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 5, "directory_id", record.directory_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 6, "creation_date", record.creation_date);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 7, "reason_code", record.reason_code);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 8, "accepted_run_manifest_path",
                                                   record.accepted_run_manifest_path);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string manifest_size;
    if (const EvidenceError status =
            require_field(lines, 9, "accepted_run_manifest_file_size", manifest_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 10, "accepted_run_manifest_file_sha256",
                                                   record.accepted_run_manifest_file_sha256);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 11, "retention_start", record.retention_start);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 12, "retention_deadline", record.retention_deadline);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_exact(lines, 13, "deletion_role", "trust_admin");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_exact(lines, 14, "state", "open");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_u64(manifest_size, record.accepted_run_manifest_file_size);
        status != EvidenceError::kOk) {
        return status;
    }
    record.bytes = std::string(bytes);
    std::string ignored;
    if (const EvidenceError status = serialize_quarantine_metadata(record, ignored);
        status != EvidenceError::kOk) {
        return status;
    }
    if (ignored != bytes) {
        return EvidenceError::kSchema;
    }
    out = std::move(record);
    return EvidenceError::kOk;
}

EvidenceError serialize_deletion_journal(const DeletionJournal& fields, std::string& out) {
    if (const EvidenceError status = require_id(fields.quarantine_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(fields.authorization_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_sha(fields.authorization_file_sha256, false);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string prefix;
    append_kv(prefix, "schema_version", "1");
    append_kv(prefix, "record_kind", "deletion_journal");
    append_kv(prefix, "quarantine_id", fields.quarantine_id);
    append_kv(prefix, "authorization_id", fields.authorization_id);
    append_kv(prefix, "authorization_file_size", text_u64(fields.authorization_file_size));
    append_kv(prefix, "authorization_file_sha256", fields.authorization_file_sha256);
    append_kv(prefix, "state", "deletion_pending");
    return seal_prefix(prefix, out);
}

EvidenceError parse_deletion_journal(std::string_view bytes, DeletionJournal& out) {
    std::vector<std::string> lines;
    if (const EvidenceError status = split_canonical(bytes, lines); status != EvidenceError::kOk) {
        return status;
    }
    if (lines.size() != 7) {
        return EvidenceError::kSchema;
    }
    DeletionJournal record;
    if (const EvidenceError status = require_exact(lines, 0, "schema_version", "1");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_exact(lines, 1, "record_kind", "deletion_journal");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 2, "quarantine_id", record.quarantine_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 3, "authorization_id", record.authorization_id);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string size_text;
    if (const EvidenceError status = require_field(lines, 4, "authorization_file_size", size_text);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_field(lines, 5, "authorization_file_sha256", record.authorization_file_sha256);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_exact(lines, 6, "state", "deletion_pending");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_u64(size_text, record.authorization_file_size);
        status != EvidenceError::kOk) {
        return status;
    }
    record.bytes = std::string(bytes);
    std::string ignored;
    if (const EvidenceError status = serialize_deletion_journal(record, ignored);
        status != EvidenceError::kOk) {
        return status;
    }
    if (ignored != bytes) {
        return EvidenceError::kSchema;
    }
    out = std::move(record);
    return EvidenceError::kOk;
}

EvidenceError acquire_evidence_lock(std::string_view evidence_root, std::string_view owner_run_id,
                                    std::string_view owner_host_id, std::string_view acquired_date,
                                    EvidenceTime* clock, EvidenceLock& out) {
    if (out.held()) {
        return EvidenceError::kLock;
    }
    if (const EvidenceError status = require_id(owner_run_id); status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(owner_host_id); status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_date(acquired_date); status != EvidenceError::kOk) {
        return status;
    }
    Fd root;
    if (const EvidenceError status = open_root(evidence_root, root); status != EvidenceError::kOk) {
        return status;
    }
    Fd lock_dir;
    if (const EvidenceError status = ensure_lock_dir(root.get(), lock_dir);
        status != EvidenceError::kOk) {
        return status;
    }
    // The budget is the sum of requested sleeps. Each request is at most the remainder,
    // and the remainder starts at 600000 ms. A scheduler may oversleep one request.
    // That oversleep is not added back into the budget and cannot extend the next request.
    EvidenceTime& timer = time_of(clock);
    std::uint64_t remaining_ms = kEvidenceLockWaitSeconds * 1000ull;
    auto observed = timer.now();
    while (true) {
        Fd file;
        std::uint64_t device = 0;
        std::uint64_t inode = 0;
        const EvidenceError created = open_created_lock(lock_dir.get(), owner_run_id, owner_host_id,
                                                        acquired_date, false, file, device, inode);
        if (created == EvidenceError::kOk) {
            out.adopt(root.release(), lock_dir.release(), file.release(), device, inode,
                      std::string(owner_run_id), std::string(owner_host_id),
                      std::string(evidence_root));
            return EvidenceError::kOk;
        }
        if (file) {
            unlinkat(lock_dir.get(), "evidence.lock", 0);
            fsync(lock_dir.get());
            file.reset();
        }
        if (created != EvidenceError::kPresent) {
            return created;
        }
        const auto current = timer.now();
        if (current < observed) {
            return EvidenceError::kInternal;
        }
        observed = current;
        if (remaining_ms == 0) {
            return EvidenceError::kLock;
        }
        const std::uint64_t slice_ms = remaining_ms < 50 ? remaining_ms : 50;
        remaining_ms -= slice_ms;
        timer.sleep_for(std::chrono::milliseconds(slice_ms));
    }
}

EvidenceError publish_batch(EvidenceLock& lock, std::string_view reservation_id,
                            const std::vector<ReservationEntry>& entries,
                            const std::vector<std::string>& payloads, BatchProbe* probe) {
    if (!lock.held()) {
        return EvidenceError::kLock;
    }
    if (const EvidenceError status = require_id(reservation_id); status != EvidenceError::kOk) {
        return status;
    }
    if (entries.size() != payloads.size()) {
        return EvidenceError::kSchema;
    }
    struct Item {
        ReservationEntry entry;
        std::string bytes;
    };
    std::vector<Item> items;
    items.reserve(entries.size());
    for (std::size_t index = 0; index < entries.size(); ++index) {
        items.push_back(Item{entries[index], payloads[index]});
    }
    std::sort(items.begin(), items.end(), [](const Item& left, const Item& right) {
        return left.entry.staging_relative_path < right.entry.staging_relative_path;
    });
    std::vector<ReservationEntry> ordered;
    ordered.reserve(items.size());
    for (const Item& item : items) {
        ordered.push_back(item.entry);
    }
    EvidenceReservation reservation;
    if (const EvidenceError status =
            serialize_evidence_reservation(lock.owner_run_id_, ordered, reservation);
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t sum_before = 0;
    if (const EvidenceError status = sum_fd(lock.root_fd_, sum_before);
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t planned = 0;
    if (const EvidenceError status = planned_bytes(reservation.entries, planned);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            reservation_fits_cap(sum_before, reservation.bytes.size(), planned);
        status != EvidenceError::kOk) {
        return status;
    }
    for (const ReservationEntry& entry : reservation.entries) {
        if (const EvidenceError status =
                require_absent_path(lock.root_fd_, entry.staging_relative_path);
            status != EvidenceError::kOk) {
            return status;
        }
        if (entry.final_relative_path != "none") {
            if (const EvidenceError status =
                    require_absent_path(lock.root_fd_, entry.final_relative_path);
                status != EvidenceError::kOk) {
                return status;
            }
        }
    }
    std::vector<TrackedFile> rollback;
    const auto fail = [&](EvidenceError status) {
        const EvidenceError cleaned = cleanup_tracked(lock.root_fd_, rollback);
        return cleaned == EvidenceError::kOk ? status : cleaned;
    };
    const std::string reservation_path = reservation_filename(reservation_id);
    Fd reservation_parent;
    Fd reservation_file;
    std::uint64_t reservation_device = 0;
    std::uint64_t reservation_inode = 0;
    if (const EvidenceError status =
            create_file(lock.root_fd_, reservation_path, reservation.bytes, reservation_file,
                        reservation_parent, reservation_device, reservation_inode);
        status != EvidenceError::kOk) {
        return status;
    }
    rollback.push_back(TrackedFile{reservation_path, reservation_device, reservation_inode});
    std::uint64_t consumed = reservation.bytes.size();
    for (std::size_t index = 0; index < reservation.entries.size(); ++index) {
        const ReservationEntry& entry = reservation.entries[index];
        const std::string& payload = items[index].bytes;
        if (payload.size() != entry.reserved_object_size) {
            return fail(EvidenceError::kSize);
        }
        std::uint64_t next = 0;
        if (add_uint64(consumed, payload.size(), next) != AdmissionError::kOk ||
            next > reservation.reserved_bytes) {
            return fail(EvidenceError::kCap);
        }
        Fd parent;
        Fd file;
        std::uint64_t device = 0;
        std::uint64_t inode = 0;
        if (const EvidenceError status = create_file(lock.root_fd_, entry.staging_relative_path,
                                                     payload, file, parent, device, inode);
            status != EvidenceError::kOk) {
            return fail(status);
        }
        std::string read_back;
        if (const EvidenceError status =
                read_limited(file.get(), payload.size(), payload.size(), read_back);
            status != EvidenceError::kOk) {
            rollback.push_back(TrackedFile{entry.staging_relative_path, device, inode});
            return fail(status);
        }
        if (read_back != payload || digest_hex(read_back) != entry.reserved_object_sha256) {
            rollback.push_back(TrackedFile{entry.staging_relative_path, device, inode});
            return fail(EvidenceError::kDigest);
        }
        rollback.push_back(TrackedFile{entry.staging_relative_path, device, inode});
        consumed = next;
        if (probe != nullptr) {
            if (const EvidenceError status =
                    probe->after_staging_written(entry.staging_relative_path);
                status != EvidenceError::kOk) {
                return fail(status);
            }
        }
        if (entry.object_class == EvidenceObjectClass::kTemporaryMetadata) {
            continue;
        }
        Fd final_parent;
        std::string final_leaf;
        if (const EvidenceError status = walk_parent(lock.root_fd_, entry.final_relative_path, true,
                                                     final_parent, final_leaf);
            status != EvidenceError::kOk) {
            return fail(status);
        }
        std::string staging_leaf;
        Fd staging_parent;
        if (const EvidenceError status = walk_parent(lock.root_fd_, entry.staging_relative_path,
                                                     false, staging_parent, staging_leaf);
            status != EvidenceError::kOk) {
            return fail(status);
        }
        if (const EvidenceError status = exclusive_rename(
                staging_parent.get(), staging_leaf.c_str(), final_parent.get(), final_leaf.c_str());
            status != EvidenceError::kOk) {
            return fail(status);
        }
        if (fsync(staging_parent.get()) != 0 || fsync(final_parent.get()) != 0) {
            return fail(EvidenceError::kIo);
        }
        struct stat renamed{};
        if (fstatat(final_parent.get(), final_leaf.c_str(), &renamed, AT_SYMLINK_NOFOLLOW) != 0) {
            return fail(from_errno(errno));
        }
        if (static_cast<std::uint64_t>(renamed.st_dev) != device ||
            static_cast<std::uint64_t>(renamed.st_ino) != inode) {
            return fail(EvidenceError::kRace);
        }
        bool staging_present = false;
        struct stat ignored{};
        if (const EvidenceError status =
                stat_leaf(lock.root_fd_, entry.staging_relative_path, ignored, staging_present);
            status != EvidenceError::kOk) {
            return fail(status);
        }
        if (staging_present) {
            return fail(EvidenceError::kPresent);
        }
        rollback.back().relative = entry.final_relative_path;
    }
    std::vector<TrackedFile> success_removals;
    success_removals.push_back(rollback.front());
    for (std::size_t index = 0; index < reservation.entries.size(); ++index) {
        if (reservation.entries[index].object_class == EvidenceObjectClass::kTemporaryMetadata) {
            success_removals.push_back(rollback[index + 1]);
        }
    }
    if (const EvidenceError status = cleanup_tracked(lock.root_fd_, success_removals);
        status != EvidenceError::kOk) {
        return fail(status);
    }
    std::uint64_t after = 0;
    if (const EvidenceError status = sum_fd(lock.root_fd_, after); status != EvidenceError::kOk) {
        return fail(status);
    }
    if (after > kEvidenceCapBytes) {
        return fail(EvidenceError::kCap);
    }
    return EvidenceError::kOk;
}

EvidenceError screen_capability(const AuthenticatedCapability* capability,
                                std::string_view operation, std::string_view evidence_root,
                                std::string_view admission_root, std::string_view trust_state,
                                std::string_view effective_date, std::string_view signer_key_id,
                                std::string_view authorization_bytes, int evidence_fd) {
    if (capability == nullptr || !capability->holds_authority()) {
        return EvidenceError::kUnavailable;
    }
    if (const EvidenceError status = require_date(effective_date); status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(trust_state); status != EvidenceError::kOk) {
        return EvidenceError::kAuthorization;
    }
    if (const EvidenceError status = require_id(signer_key_id); status != EvidenceError::kOk) {
        return EvidenceError::kAuthorization;
    }
    if (const EvidenceError status = require_date(capability->key_valid_from());
        status != EvidenceError::kOk) {
        return EvidenceError::kAuthorization;
    }
    if (const EvidenceError status = require_date(capability->key_valid_until());
        status != EvidenceError::kOk) {
        return EvidenceError::kAuthorization;
    }
    if (capability->operation() != operation ||
        capability->authorization_bytes() != authorization_bytes ||
        capability->evidence_root() != evidence_root ||
        capability->admission_root() != admission_root ||
        capability->terminal_trust_state_id() != trust_state ||
        capability->terminal_effective_date() != effective_date ||
        capability->signer_key_id() != signer_key_id ||
        capability->required_role() != "trust_admin" ||
        capability->revocation_decision() != "unrevoked" ||
        capability->signature_verification_result() != "verified" ||
        capability->key_valid_from() > effective_date ||
        capability->key_valid_until() < effective_date) {
        return EvidenceError::kAuthorization;
    }
    if (const EvidenceError status =
            require_bound_root(*capability, true, evidence_root, evidence_fd);
        status != EvidenceError::kOk) {
        return status;
    }
    return require_bound_root(*capability, false, admission_root, -1);
}

EvidenceError
recover_stale_lock(std::string_view evidence_root, std::string_view admission_root,
                   std::string_view terminal_effective_date,
                   std::string_view terminal_trust_state_id, std::string_view signer_key_id,
                   std::string_view authorization_bytes, const AuthenticatedCapability* capability,
                   std::string_view new_owner_run_id, std::string_view new_owner_host_id,
                   EvidenceTime* clock, EvidenceLock& out, bool& evidence_lock_acquired) {
    if (out.held()) {
        return EvidenceError::kLock;
    }
    if (const EvidenceError status = require_id(new_owner_run_id); status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(new_owner_host_id); status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = screen_capability(
            capability, "stale-lock", evidence_root, admission_root, terminal_trust_state_id,
            terminal_effective_date, signer_key_id, authorization_bytes, -1);
        status != EvidenceError::kOk) {
        return status;
    }
    StaleLockAuthorization parsed;
    if (const EvidenceError status = parse_stale_lock_authorization(authorization_bytes, parsed);
        status != EvidenceError::kOk) {
        return status;
    }
    if (capability->target_id() != parsed.lock_path ||
        capability->canonical_authorization_path() !=
            stale_authorization_path(parsed.authorization_id)) {
        return EvidenceError::kAuthorization;
    }
    if (const EvidenceError status =
            at_least_two_days(parsed.acquired_date, parsed.authorization_date);
        status != EvidenceError::kOk) {
        return status;
    }
    Fd admission;
    if (const EvidenceError status = open_root(admission_root, admission);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string stored;
    if (const EvidenceError status = read_record_file(
            admission.get(), stale_authorization_path(parsed.authorization_id), stored);
        status != EvidenceError::kOk) {
        return status;
    }
    if (stored != authorization_bytes) {
        return EvidenceError::kDigest;
    }
    Fd root;
    if (const EvidenceError status = open_root(evidence_root, root); status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_bound_root(*capability, true, evidence_root, root.get());
        status != EvidenceError::kOk) {
        return status;
    }
    Fd lock_dir;
    if (const EvidenceError status = ensure_lock_dir(root.get(), lock_dir);
        status != EvidenceError::kOk) {
        return status;
    }
    const bool evidence = parsed.lock_path == "lock/evidence.lock";
    Fd recovery_file;
    std::uint64_t recovery_device = 0;
    std::uint64_t recovery_inode = 0;
    std::string today;
    if (const EvidenceError status = time_of(clock).civil_date(today);
        status != EvidenceError::kOk) {
        return status;
    }
    const auto drop_recovery = [&]() -> EvidenceError {
        if (!recovery_file) {
            return EvidenceError::kOk;
        }
        const EvidenceError removed = unlink_descriptor_path(
            lock_dir.get(), "recovery.lock", recovery_file.get(), recovery_device, recovery_inode);
        recovery_file.reset();
        return removed;
    };
    if (evidence) {
        const EvidenceError created =
            open_created_lock(lock_dir.get(), new_owner_run_id, new_owner_host_id, today, true,
                              recovery_file, recovery_device, recovery_inode);
        if (created != EvidenceError::kOk) {
            if (recovery_file) {
                unlinkat(lock_dir.get(), "recovery.lock", 0);
                fsync(lock_dir.get());
                recovery_file.reset();
            }
            return created;
        }
    }
    const char* leaf = evidence ? "evidence.lock" : "recovery.lock";
    const int opened = openat(lock_dir.get(), leaf, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (opened < 0) {
        const EvidenceError status = from_errno(errno);
        drop_recovery();
        return status;
    }
    Fd target(opened);
    struct stat descriptor{};
    if (fstat(target.get(), &descriptor) != 0) {
        drop_recovery();
        return EvidenceError::kIo;
    }
    if (const EvidenceError kind = classify_file(descriptor); kind != EvidenceError::kOk) {
        drop_recovery();
        return kind;
    }
    const auto device = static_cast<std::uint64_t>(descriptor.st_dev);
    const auto inode = static_cast<std::uint64_t>(descriptor.st_ino);
    const auto size = static_cast<std::uint64_t>(descriptor.st_size);
    if (size != parsed.lock_file_size) {
        drop_recovery();
        return EvidenceError::kSize;
    }
    std::string digest;
    if (const EvidenceError status = hash_descriptor_twice(target.get(), size, digest);
        status != EvidenceError::kOk) {
        drop_recovery();
        return status;
    }
    if (digest != parsed.lock_file_sha256) {
        drop_recovery();
        return EvidenceError::kDigest;
    }
    std::string contents;
    if (const EvidenceError status = read_fd(target.get(), size, contents);
        status != EvidenceError::kOk) {
        drop_recovery();
        return status;
    }
    if (evidence) {
        EvidenceLockRecord record;
        if (const EvidenceError status = parse_evidence_lock(contents, record);
            status != EvidenceError::kOk) {
            drop_recovery();
            return status;
        }
        if (record.owner_run_id != parsed.lock_owner_run_id) {
            drop_recovery();
            return EvidenceError::kOwner;
        }
        if (record.acquired_date != parsed.acquired_date) {
            drop_recovery();
            return EvidenceError::kDate;
        }
    } else {
        RecoveryLockRecord record;
        if (const EvidenceError status = parse_recovery_lock(contents, record);
            status != EvidenceError::kOk) {
            drop_recovery();
            return status;
        }
        if (record.owner_run_id != parsed.lock_owner_run_id) {
            drop_recovery();
            return EvidenceError::kOwner;
        }
        if (record.acquired_date != parsed.acquired_date) {
            drop_recovery();
            return EvidenceError::kDate;
        }
    }
    struct stat again{};
    if (fstat(target.get(), &again) != 0 || static_cast<std::uint64_t>(again.st_dev) != device ||
        static_cast<std::uint64_t>(again.st_ino) != inode ||
        static_cast<std::uint64_t>(again.st_size) != size) {
        drop_recovery();
        return EvidenceError::kRace;
    }
    struct stat path_status{};
    if (fstatat(lock_dir.get(), leaf, &path_status, AT_SYMLINK_NOFOLLOW) != 0) {
        const EvidenceError status = from_errno(errno);
        drop_recovery();
        return status;
    }
    if (static_cast<std::uint64_t>(path_status.st_dev) != device ||
        static_cast<std::uint64_t>(path_status.st_ino) != inode ||
        static_cast<std::uint64_t>(path_status.st_size) != size) {
        drop_recovery();
        return EvidenceError::kRace;
    }
    if (unlinkat(lock_dir.get(), leaf, 0) != 0) {
        const EvidenceError status = from_errno(errno);
        drop_recovery();
        return status;
    }
    if (fstat(target.get(), &again) != 0 || static_cast<std::uint64_t>(again.st_dev) != device ||
        static_cast<std::uint64_t>(again.st_ino) != inode || again.st_nlink != 0) {
        drop_recovery();
        return EvidenceError::kRace;
    }
    if (fsync(lock_dir.get()) != 0) {
        drop_recovery();
        return EvidenceError::kIo;
    }
    target.reset();
    if (!evidence) {
        evidence_lock_acquired = false;
        return EvidenceError::kOk;
    }
    Fd created_lock;
    std::uint64_t created_device = 0;
    std::uint64_t created_inode = 0;
    const EvidenceError created =
        open_created_lock(lock_dir.get(), new_owner_run_id, new_owner_host_id, today, false,
                          created_lock, created_device, created_inode);
    if (created != EvidenceError::kOk) {
        if (created_lock) {
            unlinkat(lock_dir.get(), "evidence.lock", 0);
            fsync(lock_dir.get());
        }
        drop_recovery();
        return created;
    }
    if (const EvidenceError status = drop_recovery(); status != EvidenceError::kOk) {
        unlink_descriptor_path(lock_dir.get(), "evidence.lock", created_lock.get(), created_device,
                               created_inode);
        return status;
    }
    out.adopt(root.release(), lock_dir.release(), created_lock.release(), created_device,
              created_inode, std::string(new_owner_run_id), std::string(new_owner_host_id),
              std::string(evidence_root));
    evidence_lock_acquired = true;
    return EvidenceError::kOk;
}

namespace {

bool forbidden_cleanup_root(std::string_view path) {
    const std::vector<std::string_view> parts = split_segments(path);
    if (parts.empty()) {
        return true;
    }
    const std::string_view top = parts[0];
    return top == "candidates" || top == "provenance" || top == "classb" || top == "golden" ||
           top == "keys" || top == "signatures" || top == "license" || top == "quarantine" ||
           top == "runs";
}

EvidenceError unlink_matching_record(int root_fd, const DeletionListEntry& entry) {
    Fd parent;
    std::string leaf;
    const EvidenceError walked = walk_parent(root_fd, entry.relative_path, false, parent, leaf);
    if (walked == EvidenceError::kAbsent) {
        return EvidenceError::kOk;
    }
    if (walked != EvidenceError::kOk) {
        return walked;
    }
    const int opened = openat(parent.get(), leaf.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (opened < 0) {
        return errno == ENOENT ? EvidenceError::kOk : from_errno(errno);
    }
    Fd file(opened);
    struct stat descriptor{};
    if (fstat(file.get(), &descriptor) != 0) {
        return EvidenceError::kIo;
    }
    if (const EvidenceError kind = classify_file(descriptor); kind != EvidenceError::kOk) {
        return kind;
    }
    if (static_cast<std::uint64_t>(descriptor.st_size) != entry.deletion_entry_size) {
        return EvidenceError::kSize;
    }
    std::string bytes;
    if (const EvidenceError status =
            read_limited(file.get(), entry.deletion_entry_size, entry.deletion_entry_size, bytes);
        status != EvidenceError::kOk) {
        return status;
    }
    if (digest_hex(bytes) != entry.deletion_entry_sha256) {
        return EvidenceError::kDigest;
    }
    struct stat path_status{};
    if (fstatat(parent.get(), leaf.c_str(), &path_status, AT_SYMLINK_NOFOLLOW) != 0) {
        return from_errno(errno);
    }
    if (path_status.st_dev != descriptor.st_dev || path_status.st_ino != descriptor.st_ino) {
        return EvidenceError::kRace;
    }
    if (unlinkat(parent.get(), leaf.c_str(), 0) != 0) {
        return from_errno(errno);
    }
    if (fstat(file.get(), &descriptor) != 0 || descriptor.st_nlink != 0) {
        return EvidenceError::kRace;
    }
    if (fsync(parent.get()) != 0) {
        return EvidenceError::kIo;
    }
    return EvidenceError::kOk;
}

EvidenceError list_names(int directory, std::vector<std::string>& names) {
    const int duplicated = dup(directory);
    if (duplicated < 0) {
        return EvidenceError::kIo;
    }
    if (lseek(duplicated, 0, SEEK_SET) < 0) {
        close(duplicated);
        return EvidenceError::kIo;
    }
    DIR* stream = fdopendir(duplicated);
    if (stream == nullptr) {
        close(duplicated);
        return EvidenceError::kIo;
    }
    while (true) {
        errno = 0;
        const dirent* entry = readdir(stream);
        if (entry == nullptr) {
            const int error = errno;
            closedir(stream);
            if (error != 0) {
                return EvidenceError::kIo;
            }
            std::sort(names.begin(), names.end());
            return EvidenceError::kOk;
        }
        if (std::strcmp(entry->d_name, ".") == 0 || std::strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        names.emplace_back(entry->d_name);
    }
}

EvidenceError unavailable_input(EvidenceError status) {
    if (status == EvidenceError::kAbsent || status == EvidenceError::kIo) {
        return EvidenceError::kUnavailable;
    }
    return status;
}

EvidenceError open_readonly(int root_fd, std::string_view relative, Fd& out, struct stat& status,
                            FilePolicy policy = FilePolicy::kLegacy) {
    if (const EvidenceError checked = validate_evidence_relative_path(relative);
        checked != EvidenceError::kOk) {
        return checked;
    }
    const std::vector<std::string_view> parts = split_segments(relative);
    Fd current(dup(root_fd));
    if (!current) {
        return EvidenceError::kIo;
    }
    for (std::size_t index = 0; index + 1 < parts.size(); ++index) {
        const std::string name(parts[index]);
        struct stat before{};
        if (fstatat(current.get(), name.c_str(), &before, AT_SYMLINK_NOFOLLOW) != 0) {
            return from_errno(errno);
        }
        if (S_ISLNK(before.st_mode)) {
            return EvidenceError::kSymlink;
        }
        if (!S_ISDIR(before.st_mode)) {
            return EvidenceError::kType;
        }
        const int next =
            openat(current.get(), name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (next < 0) {
            return from_errno(errno);
        }
        struct stat opened{};
        if (fstat(next, &opened) != 0) {
            close(next);
            return EvidenceError::kIo;
        }
        if (S_ISLNK(opened.st_mode) || !S_ISDIR(opened.st_mode) || opened.st_dev != before.st_dev ||
            opened.st_ino != before.st_ino) {
            close(next);
            return S_ISLNK(opened.st_mode) ? EvidenceError::kSymlink : EvidenceError::kRace;
        }
        current = Fd(next);
    }
    const std::string leaf(parts.back());
    const int fd = openat(current.get(), leaf.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) {
        return from_errno(errno);
    }
    if (fstat(fd, &status) != 0) {
        close(fd);
        return EvidenceError::kIo;
    }
    if (const EvidenceError kind = classify_file(status, policy); kind != EvidenceError::kOk) {
        close(fd);
        return kind;
    }
    out = Fd(fd);
    return EvidenceError::kOk;
}

EvidenceError read_opened(int fd, std::uint64_t size, std::uint64_t limit, std::string& out) {
    return read_limited(fd, size, limit, out);
}

EvidenceError read_tree_file(int root_fd, std::string_view relative, std::uint64_t limit,
                             std::string& out, FilePolicy policy = FilePolicy::kLegacy) {
    Fd file;
    struct stat status{};
    if (const EvidenceError opened = open_readonly(root_fd, relative, file, status, policy);
        opened != EvidenceError::kOk) {
        return unavailable_input(opened);
    }
    return unavailable_input(
        read_opened(file.get(), static_cast<std::uint64_t>(status.st_size), limit, out));
}

using ClosedRecord = std::vector<std::pair<std::string, std::string>>;

EvidenceError parse_closed(std::string_view bytes, std::span<const std::string_view> keys,
                           ClosedRecord& out) {
    std::vector<std::string> lines;
    if (const EvidenceError status = split_canonical(bytes, lines); status != EvidenceError::kOk) {
        return status;
    }
    if (lines.size() != keys.size()) {
        return EvidenceError::kSchema;
    }
    out.clear();
    std::size_t index = 0;
    for (std::string_view key : keys) {
        std::string value;
        if (const EvidenceError status = require_field(lines, index, key, value);
            status != EvidenceError::kOk) {
            return status;
        }
        out.emplace_back(std::string(key), std::move(value));
        ++index;
    }
    return EvidenceError::kOk;
}

const std::string* record_field(const ClosedRecord& record, std::string_view key) {
    for (const auto& item : record) {
        if (item.first == key) {
            return &item.second;
        }
    }
    return nullptr;
}

EvidenceError field_is(const ClosedRecord& record, std::string_view key,
                       std::string_view expected) {
    const std::string* value = record_field(record, key);
    if (value == nullptr) {
        return EvidenceError::kSchema;
    }
    return *value == expected ? EvidenceError::kOk : EvidenceError::kSchema;
}

EvidenceError field_binds(const ClosedRecord& record, std::string_view key,
                          std::string_view expected, EvidenceError mismatch) {
    const std::string* value = record_field(record, key);
    if (value == nullptr) {
        return EvidenceError::kSchema;
    }
    return *value == expected ? EvidenceError::kOk : mismatch;
}

EvidenceError field_text(const ClosedRecord& record, std::string_view key, std::string& out) {
    const std::string* value = record_field(record, key);
    if (value == nullptr) {
        return EvidenceError::kSchema;
    }
    out = *value;
    return EvidenceError::kOk;
}

EvidenceError field_u64(const ClosedRecord& record, std::string_view key, std::uint64_t& out) {
    const std::string* value = record_field(record, key);
    if (value == nullptr) {
        return EvidenceError::kSchema;
    }
    return require_u64(*value, out);
}

bool date_in_interval(std::string_view day, std::string_view from, std::string_view until) {
    return from <= day && day <= until;
}

constexpr std::string_view kTrustTransitionDomain =
    "INFEREDGE-ED25519-V1 TRUST-TRANSITION-PAYLOAD 1\n";

constexpr std::string_view kSignedPayloadKeys[] = {"schema_version",
                                                   "record_kind",
                                                   "transition_id",
                                                   "trust_generation",
                                                   "transition_date",
                                                   "predecessor_transition_id",
                                                   "predecessor_transition_path",
                                                   "predecessor_transition_file_size",
                                                   "predecessor_transition_file_sha256",
                                                   "predecessor_successor_trust_id",
                                                   "predecessor_successor_trust_file_size",
                                                   "predecessor_successor_trust_file_sha256",
                                                   "successor_trust_id",
                                                   "successor_trust_path",
                                                   "successor_trust_file_size",
                                                   "successor_trust_file_sha256",
                                                   "bootstrap_key_sha256",
                                                   "signer_key_id",
                                                   "registry_id",
                                                   "authority_registry_file_size",
                                                   "authority_registry_file_sha256",
                                                   "publication_kind",
                                                   "binding_record_kind",
                                                   "binding_record_size",
                                                   "binding_record_sha256"};

constexpr std::string_view kPayloadSharedKeys[] = {"transition_id",
                                                   "trust_generation",
                                                   "transition_date",
                                                   "predecessor_transition_id",
                                                   "predecessor_transition_path",
                                                   "predecessor_transition_file_size",
                                                   "predecessor_transition_file_sha256",
                                                   "predecessor_successor_trust_id",
                                                   "predecessor_successor_trust_file_size",
                                                   "predecessor_successor_trust_file_sha256",
                                                   "successor_trust_id",
                                                   "successor_trust_path",
                                                   "successor_trust_file_size",
                                                   "successor_trust_file_sha256",
                                                   "bootstrap_key_sha256",
                                                   "signer_key_id",
                                                   "registry_id",
                                                   "authority_registry_file_size",
                                                   "authority_registry_file_sha256",
                                                   "publication_kind",
                                                   "binding_record_kind",
                                                   "binding_record_size",
                                                   "binding_record_sha256"};

std::string trust_transition_message(std::string_view payload) {
    std::string message;
    message.reserve(kTrustTransitionDomain.size() + payload.size());
    message.assign(kTrustTransitionDomain);
    message.append(payload);
    return message;
}

std::string self_prefix_digest(std::string_view bytes) {
    const std::size_t checksum = bytes.rfind("\nrecord_sha256=");
    if (checksum == std::string_view::npos) {
        return {};
    }
    return digest_hex(bytes.substr(0, checksum + 1));
}

EvidenceError check_ed25519_preimage(std::string_view message, std::string_view signature,
                                     std::string_view public_key) {
    if (signature.empty() || public_key.empty()) {
        return EvidenceError::kUnavailable;
    }
    if (signature.size() != 64 || public_key.size() != 32) {
        return EvidenceError::kSchema;
    }
    VerificationRequest request;
    request.message = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(message.data()), message.size());
    request.signature = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(signature.data()), signature.size());
    request.public_key = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(public_key.data()), public_key.size());
    request.authorization_digest = "ed25519-preimage";
    request.signature_family_id = "internal-signature";
    request.terminal_trust_state_id = "internal-trust";
    request.terminal_effective_date = "2020-01-01";
    request.registry_entry_id = "internal-entry";
    request.signer_key_id = "internal-key";
    request.key_valid_from = "2020-01-01";
    request.key_valid_until = "2100-12-31";
    request.role = "trust_admin";
    request.revocation_result = "unrevoked";
    request.verification_date = "2020-01-01";
    request.operation = "internal-check";
    request.canonical_authorization_path = "internal/path";
    request.target_id = "internal-target";
    const VerificationDecision decision = verify_ed25519(request);
    if (decision.status == VerificationStatus::kUnavailable) {
        return EvidenceError::kUnavailable;
    }
    if (decision.status == VerificationStatus::kMalformed || !decision.accepted()) {
        return decision.status == VerificationStatus::kRejected ? EvidenceError::kAuthorization
                                                                : EvidenceError::kSchema;
    }
    return EvidenceError::kOk;
}

struct AuthorityEntry {
    std::string entry_id;
    std::string role;
    std::string public_key_blob_id;
    std::uint64_t public_key_size{0};
    std::string public_key_sha256;
    std::string key_index_id;
    std::uint64_t key_index_size{0};
    std::string key_index_sha256;
    std::string valid_from;
    std::string valid_until;
    std::string revoked;
    std::string public_key;
};

struct LoadedRegistry {
    std::string registry_id;
    std::string bytes;
    std::string digest;
    std::vector<AuthorityEntry> entries;
};

bool known_authority_role(std::string_view role) {
    return role == "provenance_reviewer" || role == "golden_reviewer" ||
           role == "license_reviewer" || role == "registry_reviewer" || role == "policy_reviewer" ||
           role == "telemetry_reviewer" || role == "authority_admin" || role == "trust_admin";
}

EvidenceError parse_authority_entries(std::string_view bytes, std::string& list_id,
                                      std::vector<AuthorityEntry>& entries) {
    std::vector<std::string> lines;
    if (const EvidenceError status = split_canonical(bytes, lines); status != EvidenceError::kOk) {
        return status;
    }
    if (lines.size() < 4) {
        return EvidenceError::kSchema;
    }
    if (const EvidenceError status = require_exact(lines, 0, "schema_version", "1");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_exact(lines, 1, "record_kind", "authority_entry_list");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_field(lines, 2, "list_id", list_id);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string count_text;
    if (const EvidenceError status = require_field(lines, 3, "entry_count", count_text);
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t count = 0;
    if (const EvidenceError status = require_u64(count_text, count); status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(list_id); status != EvidenceError::kOk) {
        return status;
    }
    entries.clear();
    std::set<std::string> ids;
    std::set<std::string> digests;
    for (std::size_t index = 4; index < lines.size(); ++index) {
        std::vector<std::string> tokens;
        std::size_t start = 0;
        const std::string& line = lines[index];
        if (line.find('=') != std::string::npos) {
            return EvidenceError::kSchema;
        }
        while (start <= line.size()) {
            const std::size_t space = line.find(' ', start);
            if (space == std::string::npos) {
                tokens.push_back(line.substr(start));
                break;
            }
            if (space == start) {
                return EvidenceError::kLexical;
            }
            tokens.push_back(line.substr(start, space - start));
            start = space + 1;
        }
        if (tokens.size() != 11) {
            return EvidenceError::kSchema;
        }
        AuthorityEntry entry;
        entry.entry_id = tokens[0];
        entry.role = tokens[1];
        entry.public_key_blob_id = tokens[2];
        if (const EvidenceError status = require_u64(tokens[3], entry.public_key_size);
            status != EvidenceError::kOk) {
            return status;
        }
        entry.public_key_sha256 = tokens[4];
        entry.key_index_id = tokens[5];
        if (const EvidenceError status = require_u64(tokens[6], entry.key_index_size);
            status != EvidenceError::kOk) {
            return status;
        }
        entry.key_index_sha256 = tokens[7];
        entry.valid_from = tokens[8];
        entry.valid_until = tokens[9];
        entry.revoked = tokens[10];
        if (const EvidenceError status = require_id(entry.entry_id); status != EvidenceError::kOk) {
            return status;
        }
        if (!known_authority_role(entry.role)) {
            return EvidenceError::kSchema;
        }
        if (entry.public_key_blob_id != entry.entry_id || entry.key_index_id != entry.entry_id) {
            return EvidenceError::kSchema;
        }
        if (entry.public_key_size != 32) {
            return EvidenceError::kSize;
        }
        if (const EvidenceError status = require_sha(entry.public_key_sha256, false);
            status != EvidenceError::kOk) {
            return status;
        }
        if (const EvidenceError status = require_sha(entry.key_index_sha256, false);
            status != EvidenceError::kOk) {
            return status;
        }
        if (const EvidenceError status = require_date(entry.valid_from);
            status != EvidenceError::kOk) {
            return status;
        }
        if (const EvidenceError status = require_date(entry.valid_until);
            status != EvidenceError::kOk) {
            return status;
        }
        if (entry.valid_from > entry.valid_until) {
            return EvidenceError::kDate;
        }
        if (from_admission(validate_yes_no(entry.revoked)) != EvidenceError::kOk) {
            return EvidenceError::kSchema;
        }
        if (!entries.empty() && entry.entry_id <= entries.back().entry_id) {
            return EvidenceError::kDuplicate;
        }
        if (!ids.insert(entry.entry_id).second || !digests.insert(entry.public_key_sha256).second) {
            return EvidenceError::kDuplicate;
        }
        entries.push_back(std::move(entry));
    }
    if (count != entries.size()) {
        return EvidenceError::kSchema;
    }
    return EvidenceError::kOk;
}

constexpr std::string_view kVerificationKeys[] = {
    "schema_version",         "record_kind",        "signature_id",         "signed_object_kind",
    "signed_object_id",       "signed_object_path", "signed_object_size",   "signed_object_sha256",
    "signature_blob_id",      "signature_index_id", "signature_index_path", "signature_index_size",
    "signature_index_sha256", "signature_size",     "signature_sha256",     "signer_key_source",
    "signer_key_id",          "public_key_size",    "public_key_sha256",    "key_index_id",
    "key_index_size",         "key_index_sha256",   "fingerprint_sha256",   "algorithm",
    "verification_result",    "verification_date",  "verifier_id"};

constexpr std::string_view kBlobIndexKeys[] = {"schema_version",
                                               "record_kind",
                                               "blob_id",
                                               "blob_class",
                                               "raw_size",
                                               "raw_sha256",
                                               "storage_path",
                                               "review_note_id",
                                               "review_note_file_size",
                                               "review_note_file_sha256"};

EvidenceError check_verification_common(const ClosedRecord& record, std::string_view signature_id,
                                        std::string_view signed_object_id,
                                        std::uint64_t object_size, std::string_view object_digest,
                                        std::string_view blob_id, std::string_view index_id,
                                        std::uint64_t index_size, std::string_view index_digest,
                                        std::string_view signature_digest,
                                        std::string_view public_digest,
                                        EvidenceError identity_mismatch = EvidenceError::kSchema) {
    if (const EvidenceError status = field_is(record, "schema_version", "1");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(record, "record_kind", "signature_verification");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            field_binds(record, "signature_id", signature_id, identity_mismatch);
        status != EvidenceError::kOk) {
        return status;
    }
    if (signature_id == signed_object_id) {
        return EvidenceError::kAuthorization;
    }
    if (const EvidenceError status = field_is(record, "signed_object_kind", "structured_file");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            field_binds(record, "signed_object_id", signed_object_id, identity_mismatch);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(record, "signed_object_path", "none");
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t signed_size = 0;
    if (const EvidenceError status = field_u64(record, "signed_object_size", signed_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (signed_size != object_size) {
        return EvidenceError::kSize;
    }
    if (const EvidenceError status = field_is(record, "signed_object_sha256", object_digest);
        status != EvidenceError::kOk) {
        return EvidenceError::kDigest;
    }
    if (const EvidenceError status =
            field_binds(record, "signature_blob_id", blob_id, identity_mismatch);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            field_binds(record, "signature_index_id", index_id, identity_mismatch);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(record, "signature_index_path", "none");
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t recorded_index_size = 0;
    if (const EvidenceError status = field_u64(record, "signature_index_size", recorded_index_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (recorded_index_size != index_size) {
        return EvidenceError::kSize;
    }
    if (const EvidenceError status = field_is(record, "signature_index_sha256", index_digest);
        status != EvidenceError::kOk) {
        return EvidenceError::kDigest;
    }
    std::uint64_t signature_size = 0;
    if (const EvidenceError status = field_u64(record, "signature_size", signature_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (signature_size != 64) {
        return EvidenceError::kSize;
    }
    if (const EvidenceError status = field_is(record, "signature_sha256", signature_digest);
        status != EvidenceError::kOk) {
        return EvidenceError::kDigest;
    }
    std::uint64_t public_size = 0;
    if (const EvidenceError status = field_u64(record, "public_key_size", public_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (public_size != 32) {
        return EvidenceError::kSize;
    }
    if (const EvidenceError status = field_is(record, "public_key_sha256", public_digest);
        status != EvidenceError::kOk) {
        return EvidenceError::kDigest;
    }
    if (const EvidenceError status = field_is(record, "fingerprint_sha256", public_digest);
        status != EvidenceError::kOk) {
        return EvidenceError::kDigest;
    }
    if (const EvidenceError status = field_is(record, "algorithm", "ed25519");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(record, "verification_result", "verified");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(record, "verifier_id", kEd25519VerifierId);
        status != EvidenceError::kOk) {
        return status;
    }
    return EvidenceError::kOk;
}

EvidenceError check_blob_index(const ClosedRecord& index, std::string_view blob_id,
                               std::string_view blob_class, std::uint64_t raw_size,
                               std::string_view raw_digest, std::string_view storage_path,
                               EvidenceError identity_mismatch = EvidenceError::kSchema) {
    if (const EvidenceError status = field_is(index, "schema_version", "1");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(index, "record_kind", "blob_index");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_binds(index, "blob_id", blob_id, identity_mismatch);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(index, "blob_class", blob_class);
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t size = 0;
    if (const EvidenceError status = field_u64(index, "raw_size", size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (size != raw_size) {
        return EvidenceError::kSize;
    }
    if (const EvidenceError status = field_is(index, "raw_sha256", raw_digest);
        status != EvidenceError::kOk) {
        return EvidenceError::kDigest;
    }
    if (const EvidenceError status = field_is(index, "storage_path", storage_path);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(index, "review_note_id", "none");
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t review_size = 0;
    if (const EvidenceError status = field_u64(index, "review_note_file_size", review_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (review_size != 0) {
        return EvidenceError::kSize;
    }
    if (const EvidenceError status = field_is(index, "review_note_file_sha256", kZeroDigest);
        status != EvidenceError::kOk) {
        return EvidenceError::kDigest;
    }
    return EvidenceError::kOk;
}

struct PreparedCapability {
    Fd current_fd;
    std::uint64_t current_device{0};
    std::uint64_t current_inode{0};
    std::uint64_t current_size{0};
    std::string current_bytes;
    std::string successor_bytes;
    std::string operation;
    std::string authorization_bytes;
    std::string path;
    std::string target_id;
    std::string evidence_root;
    std::string admission_root;
    std::uint64_t evidence_device{0};
    std::uint64_t evidence_inode{0};
    std::uint32_t evidence_kind{0};
    std::uint64_t admission_device{0};
    std::uint64_t admission_inode{0};
    std::uint32_t admission_kind{0};
    std::string trust_state_id;
    std::string effective_date;
    std::string signer_key_id;
    std::string required_role;
    std::string key_valid_from;
    std::string key_valid_until;
    std::string revocation_decision;
    std::string signature_result;
    Fd trust_fd;
    Fd payload_fd;
    std::string payload_path;
    std::uint64_t payload_device{0};
    std::uint64_t payload_inode{0};
    std::uint32_t payload_mode{0};
    std::uint64_t payload_nlink{0};
    std::uint64_t payload_size{0};
    std::string payload_bytes;
};

EvidenceError add_chain_bytes(std::uint64_t& total, std::uint64_t more) {
    std::uint64_t sum = 0;
    if (add_uint64(total, more, sum) != AdmissionError::kOk || sum > 4194304) {
        return EvidenceError::kSize;
    }
    total = sum;
    return EvidenceError::kOk;
}

EvidenceError require_absent_ref(const ClosedRecord& record, std::string_view id_key,
                                 std::string_view size_key, std::string_view digest_key) {
    if (const EvidenceError status = field_is(record, id_key, "none");
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t size = 0;
    if (const EvidenceError status = field_u64(record, size_key, size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (size != 0) {
        return EvidenceError::kSize;
    }
    return field_is(record, digest_key, kZeroDigest);
}

EvidenceError require_lexical_ref(const ClosedRecord& record, std::string_view id_key,
                                  std::string_view size_key, std::string_view digest_key) {
    std::string ident;
    if (const EvidenceError status = field_text(record, id_key, ident);
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t size = 0;
    if (const EvidenceError status = field_u64(record, size_key, size);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string digest;
    if (const EvidenceError status = field_text(record, digest_key, digest);
        status != EvidenceError::kOk) {
        return status;
    }
    if (ident == "none") {
        return size == 0 && digest == kZeroDigest ? EvidenceError::kOk : EvidenceError::kDigest;
    }
    if (const EvidenceError status = require_id(ident); status != EvidenceError::kOk) {
        return status;
    }
    if (size == 0 || size > kMaxCanonicalRecordBytes) {
        return EvidenceError::kSize;
    }
    return require_sha(digest, false);
}

struct ChainState {
    ClosedRecord trust;
    std::string trust_bytes;
    LoadedRegistry registry;
    std::string transition_date;
    std::uint64_t revocation_generation{0};
    std::string revocation_id;
    std::string revocation_digest;
    std::string payload_path;
    Fd payload_fd;
    std::uint64_t payload_device{0};
    std::uint64_t payload_inode{0};
    std::uint32_t payload_mode{0};
    std::uint64_t payload_nlink{0};
    std::uint64_t payload_size{0};
    std::string payload_bytes;
};

EvidenceError load_signature_parts(int blob_root, std::string_view blob_path, int index_root,
                                   std::string_view index_path, int record_root,
                                   std::string_view record_path, std::string_view blob_id,
                                   std::string_view index_id, std::string_view storage_path,
                                   std::string& signature, ClosedRecord& index,
                                   std::string& index_bytes, ClosedRecord& verification,
                                   std::string& verification_bytes, std::uint64_t& chain_bytes,
                                   FilePolicy policy = FilePolicy::kLegacy,
                                   EvidenceError identity_mismatch = EvidenceError::kSchema) {
    Fd blob;
    struct stat blob_status{};
    if (const EvidenceError opened = open_readonly(blob_root, blob_path, blob, blob_status, policy);
        opened != EvidenceError::kOk) {
        return unavailable_input(opened);
    }
    if (blob_status.st_size != 64) {
        return EvidenceError::kSize;
    }
    if (const EvidenceError status = read_opened(blob.get(), 64, 64, signature);
        status != EvidenceError::kOk) {
        return unavailable_input(status);
    }
    if (const EvidenceError status = add_chain_bytes(chain_bytes, 64);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            read_tree_file(index_root, index_path, kMaxCanonicalRecordBytes, index_bytes, policy);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = add_chain_bytes(chain_bytes, index_bytes.size());
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = parse_closed(index_bytes, kBlobIndexKeys, index);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(index_id); status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            check_blob_index(index, blob_id, "signature_bytes", 64, digest_hex(signature),
                             storage_path, identity_mismatch);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = read_tree_file(
            record_root, record_path, kMaxCanonicalRecordBytes, verification_bytes, policy);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = add_chain_bytes(chain_bytes, verification_bytes.size());
        status != EvidenceError::kOk) {
        return status;
    }
    return parse_closed(verification_bytes, kVerificationKeys, verification);
}

const AuthorityEntry* find_authority(const LoadedRegistry& registry, std::string_view entry_id) {
    const AuthorityEntry* found = nullptr;
    for (const AuthorityEntry& entry : registry.entries) {
        if (entry.entry_id == entry_id) {
            if (found != nullptr) {
                return nullptr;
            }
            found = &entry;
        }
    }
    return found;
}

EvidenceError load_registry(int admission_fd, int evidence_fd, std::string_view registry_id,
                            std::string_view context_date, const std::string& bootstrap_key,
                            const LoadedRegistry* predecessor, LoadedRegistry& out,
                            std::uint64_t& chain_bytes) {
    if (const EvidenceError status = require_id(registry_id); status != EvidenceError::kOk) {
        return status;
    }
    const std::string record_path = "authority/registry-" + std::string(registry_id) + ".record";
    const std::string list_path = "authority/registry-" + std::string(registry_id) + ".entries";
    std::string registry_bytes;
    if (const EvidenceError status =
            read_tree_file(admission_fd, record_path, kMaxCanonicalRecordBytes, registry_bytes);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = add_chain_bytes(chain_bytes, registry_bytes.size());
        status != EvidenceError::kOk) {
        return status;
    }
    static constexpr std::string_view kRegistryKeys[] = {"schema_version",
                                                         "record_kind",
                                                         "registry_id",
                                                         "predecessor_registry_id",
                                                         "predecessor_registry_file_size",
                                                         "predecessor_registry_file_sha256",
                                                         "entry_list_id",
                                                         "entry_list_file_size",
                                                         "entry_list_file_sha256",
                                                         "entry_count"};
    ClosedRecord registry;
    if (const EvidenceError status = parse_closed(registry_bytes, kRegistryKeys, registry);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(registry, "schema_version", "1");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(registry, "record_kind", "authority_registry");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(registry, "registry_id", registry_id);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string list_bytes;
    if (const EvidenceError status =
            read_tree_file(admission_fd, list_path, kMaxCanonicalRecordBytes, list_bytes);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = add_chain_bytes(chain_bytes, list_bytes.size());
        status != EvidenceError::kOk) {
        return status;
    }
    std::string list_id;
    std::vector<AuthorityEntry> entries;
    if (const EvidenceError status = parse_authority_entries(list_bytes, list_id, entries);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(registry, "entry_list_id", list_id);
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t list_size = 0;
    if (const EvidenceError status = field_u64(registry, "entry_list_file_size", list_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (list_size != list_bytes.size()) {
        return EvidenceError::kSize;
    }
    if (const EvidenceError status =
            field_is(registry, "entry_list_file_sha256", digest_hex(list_bytes));
        status != EvidenceError::kOk) {
        return EvidenceError::kDigest;
    }
    std::uint64_t entry_count = 0;
    if (const EvidenceError status = field_u64(registry, "entry_count", entry_count);
        status != EvidenceError::kOk) {
        return status;
    }
    if (entry_count != entries.size()) {
        return EvidenceError::kSchema;
    }
    const std::string signature_id = "registry-envelope-signature-" + std::string(registry_id);
    const std::string blob_id = "registry-envelope-sigblob-" + std::string(registry_id);
    const std::string index_id = "registry-envelope-sigindex-" + std::string(registry_id);
    const std::string blob_path = "benchmarks/generative_phase9c/signatures/" + blob_id + ".bin";
    const std::string index_path = "signature-indexes/" + index_id + ".index";
    const std::string verification_path = "signature-records/" + signature_id + ".record";
    std::string signature;
    std::string index_bytes;
    std::string verification_bytes;
    ClosedRecord index;
    ClosedRecord verification;
    if (const EvidenceError status =
            load_signature_parts(evidence_fd, blob_path, admission_fd, index_path, admission_fd,
                                 verification_path, blob_id, index_id, blob_path, signature, index,
                                 index_bytes, verification, verification_bytes, chain_bytes);
        status != EvidenceError::kOk) {
        return status;
    }
    const bool first = predecessor == nullptr;
    if (first) {
        if (const EvidenceError status = require_absent_ref(registry, "predecessor_registry_id",
                                                            "predecessor_registry_file_size",
                                                            "predecessor_registry_file_sha256");
            status != EvidenceError::kOk) {
            return status;
        }
        if (const EvidenceError status =
                check_ed25519_preimage(registry_bytes, signature, bootstrap_key);
            status != EvidenceError::kOk) {
            return status;
        }
        if (const EvidenceError status = check_verification_common(
                verification, signature_id, "registry-" + std::string(registry_id),
                registry_bytes.size(), digest_hex(registry_bytes), blob_id, index_id,
                index_bytes.size(), digest_hex(index_bytes), digest_hex(signature),
                digest_hex(bootstrap_key));
            status != EvidenceError::kOk) {
            return status;
        }
        if (const EvidenceError status =
                field_is(verification, "signer_key_source", "bootstrap_handle");
            status != EvidenceError::kOk) {
            return status;
        }
        if (const EvidenceError status = field_is(verification, "signer_key_id", "none");
            status != EvidenceError::kOk) {
            return status;
        }
        if (const EvidenceError status = field_is(verification, "key_index_id", "none");
            status != EvidenceError::kOk) {
            return status;
        }
        std::uint64_t key_index_size = 0;
        if (const EvidenceError status = field_u64(verification, "key_index_size", key_index_size);
            status != EvidenceError::kOk) {
            return status;
        }
        if (key_index_size != 0) {
            return EvidenceError::kSize;
        }
        if (const EvidenceError status = field_is(verification, "key_index_sha256", kZeroDigest);
            status != EvidenceError::kOk) {
            return EvidenceError::kDigest;
        }
    } else {
        std::uint64_t predecessor_size = 0;
        if (const EvidenceError status =
                field_u64(registry, "predecessor_registry_file_size", predecessor_size);
            status != EvidenceError::kOk) {
            return status;
        }
        if (predecessor_size != predecessor->bytes.size() ||
            field_is(registry, "predecessor_registry_id", predecessor->registry_id) !=
                EvidenceError::kOk ||
            field_is(registry, "predecessor_registry_file_sha256", predecessor->digest) !=
                EvidenceError::kOk) {
            return EvidenceError::kDigest;
        }
        std::string signer_id;
        if (const EvidenceError status = field_text(verification, "signer_key_id", signer_id);
            status != EvidenceError::kOk) {
            return status;
        }
        const AuthorityEntry* admin = find_authority(*predecessor, signer_id);
        if (admin == nullptr || admin->role != "authority_admin" || admin->revoked != "no" ||
            !date_in_interval(context_date, admin->valid_from, admin->valid_until)) {
            return EvidenceError::kAuthorization;
        }
        if (const EvidenceError status =
                check_ed25519_preimage(registry_bytes, signature, admin->public_key);
            status != EvidenceError::kOk) {
            return status;
        }
        if (const EvidenceError status = check_verification_common(
                verification, signature_id, "registry-" + std::string(registry_id),
                registry_bytes.size(), digest_hex(registry_bytes), blob_id, index_id,
                index_bytes.size(), digest_hex(index_bytes), digest_hex(signature),
                digest_hex(admin->public_key));
            status != EvidenceError::kOk) {
            return status;
        }
        if (const EvidenceError status = field_is(verification, "signer_key_source", "blob");
            status != EvidenceError::kOk) {
            return status;
        }
        if (const EvidenceError status = field_is(verification, "key_index_id", admin->entry_id);
            status != EvidenceError::kOk) {
            return status;
        }
        std::uint64_t key_index_size = 0;
        if (const EvidenceError status = field_u64(verification, "key_index_size", key_index_size);
            status != EvidenceError::kOk) {
            return status;
        }
        if (key_index_size != admin->key_index_size) {
            return EvidenceError::kSize;
        }
        if (const EvidenceError status =
                field_is(verification, "key_index_sha256", admin->key_index_sha256);
            status != EvidenceError::kOk) {
            return EvidenceError::kDigest;
        }
    }
    std::string verification_date;
    if (const EvidenceError status =
            field_text(verification, "verification_date", verification_date);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_date(verification_date);
        status != EvidenceError::kOk) {
        return status;
    }
    if (verification_date != context_date) {
        return EvidenceError::kDate;
    }
    for (AuthorityEntry& entry : entries) {
        const std::string key_path =
            "benchmarks/generative_phase9c/keys/" + entry.entry_id + ".bin";
        const std::string key_index_path = "keys/" + entry.entry_id + ".index";
        Fd key_file;
        struct stat key_status{};
        if (const EvidenceError opened = open_readonly(evidence_fd, key_path, key_file, key_status);
            opened != EvidenceError::kOk) {
            return unavailable_input(opened);
        }
        if (key_status.st_size != 32) {
            return EvidenceError::kSize;
        }
        if (const EvidenceError status = read_opened(key_file.get(), 32, 32, entry.public_key);
            status != EvidenceError::kOk) {
            return unavailable_input(status);
        }
        if (digest_hex(entry.public_key) != entry.public_key_sha256) {
            return EvidenceError::kDigest;
        }
        std::string key_index_bytes;
        if (const EvidenceError status = read_tree_file(admission_fd, key_index_path,
                                                        kMaxCanonicalRecordBytes, key_index_bytes);
            status != EvidenceError::kOk) {
            return status;
        }
        if (key_index_bytes.size() != entry.key_index_size ||
            digest_hex(key_index_bytes) != entry.key_index_sha256) {
            return EvidenceError::kDigest;
        }
        ClosedRecord key_index;
        if (const EvidenceError status = parse_closed(key_index_bytes, kBlobIndexKeys, key_index);
            status != EvidenceError::kOk) {
            return status;
        }
        if (const EvidenceError status = check_blob_index(key_index, entry.entry_id, "public_key",
                                                          32, entry.public_key_sha256, key_path);
            status != EvidenceError::kOk) {
            return status;
        }
    }
    out.registry_id = std::string(registry_id);
    out.bytes = std::move(registry_bytes);
    out.digest = digest_hex(out.bytes);
    out.entries = std::move(entries);
    return EvidenceError::kOk;
}

constexpr std::string_view kTrustStateKeys[] = {"schema_version",
                                                "record_kind",
                                                "trust_id",
                                                "trust_generation",
                                                "effective_date",
                                                "terminal_transition_id",
                                                "terminal_transition_path",
                                                "revocation_generation",
                                                "revocation_id",
                                                "revocation_file_size",
                                                "revocation_file_sha256",
                                                "config_id",
                                                "configuration_file_size",
                                                "configuration_file_sha256",
                                                "registry_id",
                                                "authority_registry_file_size",
                                                "authority_registry_file_sha256",
                                                "authority_envelope_id",
                                                "authority_envelope_file_size",
                                                "authority_envelope_file_sha256",
                                                "bootstrap_key_handle",
                                                "bootstrap_key_size",
                                                "bootstrap_key_sha256",
                                                "admission_manifest_id",
                                                "admission_root_manifest_file_size",
                                                "admission_root_manifest_file_sha256",
                                                "host_id",
                                                "publication_kind",
                                                "publication_authorization_id",
                                                "publication_authorization_path",
                                                "publication_authorization_file_size",
                                                "publication_authorization_file_sha256",
                                                "publication_authorization_preimage_sha256",
                                                "signer_key_id",
                                                "signer_publication_key_size",
                                                "signer_publication_key_sha256",
                                                "signature_family_id",
                                                "lineage_kind",
                                                "binding_record_kind",
                                                "binding_record_size",
                                                "binding_record_sha256",
                                                "telemetry_mode",
                                                "telemetry_governance_id",
                                                "telemetry_governance_file_size",
                                                "telemetry_governance_file_sha256",
                                                "predecessor_trust_id",
                                                "predecessor_file_size",
                                                "predecessor_file_sha256"};

constexpr std::string_view kTrustTransitionKeys[] = {"schema_version",
                                                     "record_kind",
                                                     "transition_id",
                                                     "trust_generation",
                                                     "transition_date",
                                                     "predecessor_transition_id",
                                                     "predecessor_transition_path",
                                                     "predecessor_transition_file_size",
                                                     "predecessor_transition_file_sha256",
                                                     "predecessor_successor_trust_id",
                                                     "predecessor_successor_trust_file_size",
                                                     "predecessor_successor_trust_file_sha256",
                                                     "successor_trust_id",
                                                     "successor_trust_path",
                                                     "successor_trust_file_size",
                                                     "successor_trust_file_sha256",
                                                     "bootstrap_key_sha256",
                                                     "signer_key_id",
                                                     "registry_id",
                                                     "authority_registry_file_size",
                                                     "authority_registry_file_sha256",
                                                     "signed_payload_id",
                                                     "signed_payload_path",
                                                     "signed_payload_size",
                                                     "signed_payload_sha256",
                                                     "signature_id",
                                                     "signature_blob_id",
                                                     "signature_index_id",
                                                     "signature_file_size",
                                                     "signature_file_sha256",
                                                     "publication_kind",
                                                     "binding_record_kind",
                                                     "binding_record_size",
                                                     "binding_record_sha256"};

EvidenceError append_bound(std::string& prefix, const ClosedRecord& record, std::string_view key) {
    std::string value;
    if (const EvidenceError status = field_text(record, key, value); status != EvidenceError::kOk) {
        return status;
    }
    append_kv(prefix, key, value);
    return EvidenceError::kOk;
}

bool allowed_publication_kind(std::string_view kind) {
    return kind == "withdrawal" || kind == "ordinary" || kind == "revocation" ||
           kind == "rollback" || kind == "native_initialization" || kind == "governance_pin" ||
           kind == "activation";
}

EvidenceError check_one_way_binding(const ClosedRecord& trust, const ClosedRecord* transition) {
    std::string kind;
    if (const EvidenceError status = field_text(trust, "publication_kind", kind);
        status != EvidenceError::kOk) {
        return status;
    }
    if (!allowed_publication_kind(kind)) {
        return EvidenceError::kSchema;
    }
    if (const EvidenceError status =
            field_is(trust, "binding_record_kind", "transition_operation_binding");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(trust, "signature_family_id", "trust_transition");
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t signer_size = 0;
    if (const EvidenceError status = field_u64(trust, "signer_publication_key_size", signer_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (signer_size != 32) {
        return EvidenceError::kSize;
    }
    std::string signer_digest;
    if (const EvidenceError status =
            field_text(trust, "signer_publication_key_sha256", signer_digest);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_sha(signer_digest, false);
        status != EvidenceError::kOk) {
        return status;
    }
    if (kind == "native_initialization") {
        if (const EvidenceError status = field_is(trust, "publication_authorization_id", "none");
            status != EvidenceError::kOk) {
            return status;
        }
        if (const EvidenceError status = field_is(trust, "publication_authorization_path", "none");
            status != EvidenceError::kOk) {
            return status;
        }
        std::uint64_t authorization_size = 0;
        if (const EvidenceError status =
                field_u64(trust, "publication_authorization_file_size", authorization_size);
            status != EvidenceError::kOk) {
            return status;
        }
        if (authorization_size != 0) {
            return EvidenceError::kSize;
        }
        if (field_is(trust, "publication_authorization_file_sha256", kZeroDigest) !=
                EvidenceError::kOk ||
            field_is(trust, "publication_authorization_preimage_sha256", kZeroDigest) !=
                EvidenceError::kOk) {
            return EvidenceError::kDigest;
        }
        if (const EvidenceError status = field_is(trust, "signer_key_id", "none");
            status != EvidenceError::kOk) {
            return status;
        }
        if (const EvidenceError status = field_is(trust, "lineage_kind", "native");
            status != EvidenceError::kOk) {
            return status;
        }
        if (const EvidenceError status = require_absent_ref(
                trust, "predecessor_trust_id", "predecessor_file_size", "predecessor_file_sha256");
            status != EvidenceError::kOk) {
            return status;
        }
    } else {
        std::string predecessor;
        if (const EvidenceError status = field_text(trust, "predecessor_trust_id", predecessor);
            status != EvidenceError::kOk) {
            return status;
        }
        if (predecessor == "none") {
            return EvidenceError::kSchema;
        }
        std::string predecessor_digest;
        if (const EvidenceError status =
                field_text(trust, "predecessor_file_sha256", predecessor_digest);
            status != EvidenceError::kOk) {
            return status;
        }
        if (predecessor_digest == kZeroDigest) {
            return EvidenceError::kDigest;
        }
        std::string authorization_digest;
        if (const EvidenceError status =
                field_text(trust, "publication_authorization_file_sha256", authorization_digest);
            status != EvidenceError::kOk) {
            return status;
        }
        if (authorization_digest == kZeroDigest) {
            return EvidenceError::kDigest;
        }
    }
    std::string prefix;
    append_kv(prefix, "schema_version", "1");
    append_kv(prefix, "record_kind", "transition_operation_binding");
    static constexpr std::string_view kCopied[] = {"publication_kind",
                                                   "publication_authorization_id",
                                                   "publication_authorization_path",
                                                   "publication_authorization_file_size",
                                                   "publication_authorization_file_sha256",
                                                   "publication_authorization_preimage_sha256",
                                                   "signer_key_id",
                                                   "signer_publication_key_size",
                                                   "signer_publication_key_sha256",
                                                   "signature_family_id",
                                                   "trust_id",
                                                   "trust_generation",
                                                   "effective_date",
                                                   "terminal_transition_id",
                                                   "terminal_transition_path",
                                                   "predecessor_trust_id",
                                                   "predecessor_file_size",
                                                   "predecessor_file_sha256",
                                                   "host_id",
                                                   "lineage_kind",
                                                   "revocation_generation",
                                                   "revocation_id",
                                                   "revocation_file_size",
                                                   "revocation_file_sha256",
                                                   "config_id",
                                                   "configuration_file_size",
                                                   "configuration_file_sha256",
                                                   "registry_id",
                                                   "authority_registry_file_size",
                                                   "authority_registry_file_sha256",
                                                   "authority_envelope_id",
                                                   "authority_envelope_file_size",
                                                   "authority_envelope_file_sha256",
                                                   "bootstrap_key_handle",
                                                   "bootstrap_key_size",
                                                   "bootstrap_key_sha256",
                                                   "admission_manifest_id",
                                                   "admission_root_manifest_file_size",
                                                   "admission_root_manifest_file_sha256"};
    for (std::string_view key : kCopied) {
        if (const EvidenceError status = append_bound(prefix, trust, key);
            status != EvidenceError::kOk) {
            return status;
        }
    }
    if (!prefix.empty() && prefix.back() != '\n') {
        return EvidenceError::kInternal;
    }
    std::string binding;
    if (const EvidenceError status = seal_prefix(prefix, binding); status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t recorded_size = 0;
    if (const EvidenceError status = field_u64(trust, "binding_record_size", recorded_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (recorded_size != binding.size() ||
        field_is(trust, "binding_record_sha256", digest_hex(binding)) != EvidenceError::kOk) {
        return EvidenceError::kDigest;
    }
    if (transition != nullptr) {
        if (field_is(*transition, "publication_kind", kind) != EvidenceError::kOk ||
            field_is(*transition, "binding_record_kind", "transition_operation_binding") !=
                EvidenceError::kOk ||
            field_is(*transition, "binding_record_sha256", digest_hex(binding)) !=
                EvidenceError::kOk) {
            return EvidenceError::kDigest;
        }
        std::uint64_t transition_size = 0;
        if (const EvidenceError status =
                field_u64(*transition, "binding_record_size", transition_size);
            status != EvidenceError::kOk) {
            return status;
        }
        if (transition_size != binding.size()) {
            return EvidenceError::kDigest;
        }
    }
    return EvidenceError::kOk;
}

EvidenceError read_trust_state(int admission_fd, std::string_view trust_id, std::string& bytes,
                               ClosedRecord& record, std::uint64_t& chain_bytes) {

    const std::string path = "trust/trust-" + std::string(trust_id) + ".record";
    if (const EvidenceError status =
            read_tree_file(admission_fd, path, kMaxCanonicalRecordBytes, bytes);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = add_chain_bytes(chain_bytes, bytes.size());
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = parse_closed(bytes, kTrustStateKeys, record);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(record, "schema_version", "1");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(record, "record_kind", "trust_state");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(record, "trust_id", trust_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(record, "bootstrap_key_handle", "bootstrap-key-v1");
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t bootstrap_size = 0;
    if (const EvidenceError status = field_u64(record, "bootstrap_key_size", bootstrap_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (bootstrap_size != 32) {
        return EvidenceError::kSize;
    }
    if (const EvidenceError status = field_is(record, "telemetry_mode", "disabled");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            require_absent_ref(record, "telemetry_governance_id", "telemetry_governance_file_size",
                               "telemetry_governance_file_sha256");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_lexical_ref(
            record, "revocation_id", "revocation_file_size", "revocation_file_sha256");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_lexical_ref(
            record, "config_id", "configuration_file_size", "configuration_file_sha256");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_lexical_ref(record, "admission_manifest_id",
                                                         "admission_root_manifest_file_size",
                                                         "admission_root_manifest_file_sha256");
        status != EvidenceError::kOk) {
        return status;
    }
    std::string host;
    if (const EvidenceError status = field_text(record, "host_id", host);
        status != EvidenceError::kOk) {
        return status;
    }
    return require_id(host);
}

EvidenceError payload_canonical_shape(std::string_view bytes) {
    if (bytes.find('\0') != std::string_view::npos || bytes.find('\r') != std::string_view::npos) {
        return EvidenceError::kSchema;
    }
    constexpr std::string_view checksum_key = "record_sha256=";
    std::size_t line_start = std::string_view::npos;
    const std::size_t marked = bytes.rfind("\nrecord_sha256=");
    if (marked != std::string_view::npos) {
        line_start = marked + 1;
    } else if (bytes.size() >= checksum_key.size() &&
               bytes.substr(0, checksum_key.size()) == checksum_key) {
        line_start = 0;
    }
    constexpr std::size_t checksum_line = checksum_key.size() + 64;
    if (line_start != std::string_view::npos && bytes.size() >= line_start + checksum_line + 1 &&
        bytes[line_start + checksum_line] == '\n' &&
        bytes.size() > line_start + checksum_line + 1) {
        return EvidenceError::kSchema;
    }
    return EvidenceError::kOk;
}

EvidenceError load_transition_payload(int trust_fd, std::string_view relative, std::string& bytes,
                                      struct stat& status, Fd& file) {
    if (const EvidenceError opened =
            open_readonly(trust_fd, relative, file, status, FilePolicy::kTrustFamily);
        opened != EvidenceError::kOk) {
        return unavailable_input(opened);
    }
    if (static_cast<std::uint64_t>(status.st_size) > kMaxCanonicalRecordBytes) {
        return EvidenceError::kCap;
    }
    if (const EvidenceError read =
            unavailable_input(read_opened(file.get(), static_cast<std::uint64_t>(status.st_size),
                                          kMaxCanonicalRecordBytes, bytes));
        read != EvidenceError::kOk) {
        return read;
    }
    return payload_canonical_shape(bytes);
}

EvidenceError authenticate_transition_payload(
    int trust_fd, ChainState& state, const ClosedRecord& transition, std::string_view transition_id,
    std::string_view payload_path, std::string_view signature, std::string_view public_key,
    const ClosedRecord& verification, std::string_view verification_bytes,
    std::string_view index_bytes, std::string_view signature_id, std::string_view blob_id,
    std::string_view index_id, std::uint64_t& chain_bytes) {
    struct stat payload_status{};
    std::string payload_bytes;
    Fd payload_file;
    if (const EvidenceError status = load_transition_payload(trust_fd, payload_path, payload_bytes,
                                                             payload_status, payload_file);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = add_chain_bytes(chain_bytes, payload_bytes.size());
        status != EvidenceError::kOk) {
        return status;
    }
    ClosedRecord payload;
    if (const EvidenceError status = parse_closed(payload_bytes, kSignedPayloadKeys, payload);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(payload, "schema_version", "1");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            field_is(payload, "record_kind", "trust_transition_signed_payload");
        status != EvidenceError::kOk) {
        return status;
    }
    for (const std::string_view key : kPayloadSharedKeys) {
        std::string payload_value;
        std::string transition_value;
        if (const EvidenceError status = field_text(payload, key, payload_value);
            status != EvidenceError::kOk) {
            return status;
        }
        if (const EvidenceError status = field_text(transition, key, transition_value);
            status != EvidenceError::kOk) {
            return status;
        }
        if (payload_value != transition_value) {
            return key.find("sha256") != std::string_view::npos ? EvidenceError::kDigest
                                                                : EvidenceError::kState;
        }
    }
    const std::string payload_id = "transition-payload-" + std::string(transition_id);
    if (const EvidenceError status = field_is(transition, "signed_payload_id", payload_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(transition, "signed_payload_path", payload_path);
        status != EvidenceError::kOk) {
        return status == EvidenceError::kSchema &&
                       record_field(transition, "signed_payload_path") != nullptr
                   ? EvidenceError::kPath
                   : status;
    }
    std::uint64_t recorded_size = 0;
    if (const EvidenceError status = field_u64(transition, "signed_payload_size", recorded_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (recorded_size != payload_bytes.size()) {
        return EvidenceError::kSize;
    }
    if (const EvidenceError status =
            field_is(transition, "signed_payload_sha256", digest_hex(payload_bytes));
        status != EvidenceError::kOk) {
        return EvidenceError::kDigest;
    }
    std::uint64_t signature_file_size = 0;
    if (const EvidenceError status =
            field_u64(transition, "signature_file_size", signature_file_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (signature_file_size != verification_bytes.size()) {
        return EvidenceError::kSize;
    }
    std::string signature_file_digest;
    if (const EvidenceError status =
            field_text(transition, "signature_file_sha256", signature_file_digest);
        status != EvidenceError::kOk) {
        return status;
    }
    if (signature_file_digest == kZeroDigest ||
        signature_file_digest != digest_hex(verification_bytes)) {
        return EvidenceError::kDigest;
    }
    const std::string message = trust_transition_message(payload_bytes);
    if (const EvidenceError status = check_ed25519_preimage(message, signature, public_key);
        status != EvidenceError::kOk) {
        return status;
    }
    const std::string index_digest = self_prefix_digest(index_bytes);
    if (index_digest.empty()) {
        return EvidenceError::kChecksum;
    }
    if (const EvidenceError status = check_verification_common(
            verification, signature_id, payload_id, payload_bytes.size(), digest_hex(payload_bytes),
            blob_id, index_id, index_bytes.size(), index_digest, digest_hex(signature),
            digest_hex(public_key), EvidenceError::kState);
        status != EvidenceError::kOk) {
        return status;
    }
    state.payload_path = std::string(payload_path);
    state.payload_device = static_cast<std::uint64_t>(payload_status.st_dev);
    state.payload_inode = static_cast<std::uint64_t>(payload_status.st_ino);
    state.payload_mode = static_cast<std::uint32_t>(payload_status.st_mode & 0777);
    state.payload_nlink = static_cast<std::uint64_t>(payload_status.st_nlink);
    state.payload_size = payload_bytes.size();
    state.payload_bytes = std::move(payload_bytes);
    state.payload_fd = std::move(payload_file);
    return EvidenceError::kOk;
}

EvidenceError verify_genesis(int trust_fd, int admission_fd, int evidence_fd, ChainState& state,
                             std::string& bootstrap_key, std::uint64_t& chain_bytes) {
    if (const EvidenceError status =
            read_tree_file(trust_fd, "key/bootstrap-key-v1", 32, bootstrap_key);
        status != EvidenceError::kOk) {
        return status;
    }
    if (bootstrap_key.size() != 32) {
        return EvidenceError::kSize;
    }
    if (const EvidenceError status = add_chain_bytes(chain_bytes, 32);
        status != EvidenceError::kOk) {
        return status;
    }

    std::string transition_bytes;
    if (const EvidenceError status =
            read_tree_file(trust_fd, "genesis/transition.record", kMaxCanonicalRecordBytes,
                           transition_bytes, FilePolicy::kTrustFamily);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = add_chain_bytes(chain_bytes, transition_bytes.size());
        status != EvidenceError::kOk) {
        return status;
    }
    ClosedRecord transition;
    if (const EvidenceError status =
            parse_closed(transition_bytes, kTrustTransitionKeys, transition);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(transition, "schema_version", "1");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(transition, "record_kind", "trust_transition");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(transition, "transition_id", "genesis-v1");
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t generation = 0;
    if (const EvidenceError status = field_u64(transition, "trust_generation", generation);
        status != EvidenceError::kOk) {
        return status;
    }
    if (generation < 1 || generation > 64) {
        return EvidenceError::kSchema;
    }
    if (generation != 1) {
        return EvidenceError::kState;
    }
    if (const EvidenceError status = require_absent_ref(transition, "predecessor_transition_id",
                                                        "predecessor_transition_file_size",
                                                        "predecessor_transition_file_sha256");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(transition, "predecessor_transition_path", "none");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_absent_ref(
            transition, "predecessor_successor_trust_id", "predecessor_successor_trust_file_size",
            "predecessor_successor_trust_file_sha256");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(transition, "signer_key_id", "none");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            field_is(transition, "bootstrap_key_sha256", digest_hex(bootstrap_key));
        status != EvidenceError::kOk) {
        return EvidenceError::kDigest;
    }
    if (const EvidenceError status =
            field_is(transition, "signature_id", "transition-signature-genesis-v1");
        status != EvidenceError::kOk) {
        return status;
    }
    std::string blob_id;
    std::string index_id;
    if (const EvidenceError status = field_text(transition, "signature_blob_id", blob_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_text(transition, "signature_index_id", index_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(blob_id); status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(index_id); status != EvidenceError::kOk) {
        return status;
    }
    std::string signature;
    std::string index_bytes;
    std::string verification_bytes;
    ClosedRecord index;
    ClosedRecord verification;
    if (const EvidenceError status = load_signature_parts(
            trust_fd, "genesis/signature.bin", trust_fd, "genesis/signature.index", trust_fd,
            "genesis/verification.record", blob_id, index_id, "genesis/signature.bin", signature,
            index, index_bytes, verification, verification_bytes, chain_bytes,
            FilePolicy::kTrustFamily, EvidenceError::kState);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = authenticate_transition_payload(
            trust_fd, state, transition, "genesis-v1", "genesis/transition.payload", signature,
            bootstrap_key, verification, verification_bytes, index_bytes,
            "transition-signature-genesis-v1", blob_id, index_id, chain_bytes);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string transition_date;
    if (const EvidenceError status = field_text(transition, "transition_date", transition_date);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_date(transition_date); status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            field_is(verification, "signer_key_source", "bootstrap_handle");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            field_binds(verification, "signer_key_id", "none", EvidenceError::kState);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(verification, "key_index_id", "none");
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t key_index_size = 0;
    if (const EvidenceError status = field_u64(verification, "key_index_size", key_index_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (key_index_size != 0 ||
        field_is(verification, "key_index_sha256", kZeroDigest) != EvidenceError::kOk) {
        return EvidenceError::kDigest;
    }
    if (const EvidenceError status = field_is(verification, "verification_date", transition_date);
        status != EvidenceError::kOk) {
        return EvidenceError::kDate;
    }
    std::string trust_id;
    if (const EvidenceError status = field_text(transition, "successor_trust_id", trust_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(trust_id); status != EvidenceError::kOk) {
        return status;
    }
    const std::string successor_path = "admission/trust/trust-" + trust_id + ".record";
    if (const EvidenceError status = field_is(transition, "successor_trust_path", successor_path);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            read_trust_state(admission_fd, trust_id, state.trust_bytes, state.trust, chain_bytes);
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t successor_size = 0;
    if (const EvidenceError status =
            field_u64(transition, "successor_trust_file_size", successor_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (successor_size != state.trust_bytes.size() ||
        field_is(transition, "successor_trust_file_sha256", digest_hex(state.trust_bytes)) !=
            EvidenceError::kOk) {
        return EvidenceError::kDigest;
    }
    std::uint64_t trust_generation = 0;
    if (const EvidenceError status = field_u64(state.trust, "trust_generation", trust_generation);
        status != EvidenceError::kOk) {
        return status;
    }
    if (trust_generation != 1 ||
        field_is(state.trust, "effective_date", transition_date) != EvidenceError::kOk ||
        field_is(state.trust, "terminal_transition_id", "genesis-v1") != EvidenceError::kOk ||
        field_is(state.trust, "terminal_transition_path", "genesis/transition.record") !=
            EvidenceError::kOk) {
        return EvidenceError::kSchema;
    }
    if (const EvidenceError status = check_one_way_binding(state.trust, &transition);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            field_is(state.trust, "bootstrap_key_sha256", digest_hex(bootstrap_key));
        status != EvidenceError::kOk) {
        return EvidenceError::kDigest;
    }
    if (const EvidenceError status =
            require_absent_ref(state.trust, "predecessor_trust_id", "predecessor_file_size",
                               "predecessor_file_sha256");
        status != EvidenceError::kOk) {
        return status;
    }
    std::string registry_id;
    if (const EvidenceError status = field_text(transition, "registry_id", registry_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(state.trust, "registry_id", registry_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            load_registry(admission_fd, evidence_fd, registry_id, transition_date, bootstrap_key,
                          nullptr, state.registry, chain_bytes);
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t registry_size = 0;
    if (const EvidenceError status =
            field_u64(transition, "authority_registry_file_size", registry_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (registry_size != state.registry.bytes.size() ||
        field_is(transition, "authority_registry_file_sha256", state.registry.digest) !=
            EvidenceError::kOk ||
        field_u64(state.trust, "authority_registry_file_size", registry_size) !=
            EvidenceError::kOk ||
        registry_size != state.registry.bytes.size() ||
        field_is(state.trust, "authority_registry_file_sha256", state.registry.digest) !=
            EvidenceError::kOk) {
        return EvidenceError::kDigest;
    }
    const std::string envelope_id = "registry-envelope-signature-" + registry_id;
    if (const EvidenceError status = field_is(state.trust, "authority_envelope_id", envelope_id);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string envelope_bytes;
    if (const EvidenceError status =
            read_tree_file(admission_fd, "signature-records/" + envelope_id + ".record",
                           kMaxCanonicalRecordBytes, envelope_bytes);
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t envelope_size = 0;
    if (const EvidenceError status =
            field_u64(state.trust, "authority_envelope_file_size", envelope_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (envelope_size != envelope_bytes.size() ||
        field_is(state.trust, "authority_envelope_file_sha256", digest_hex(envelope_bytes)) !=
            EvidenceError::kOk) {
        return EvidenceError::kDigest;
    }
    if (const EvidenceError status =
            field_u64(state.trust, "revocation_generation", state.revocation_generation);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_text(state.trust, "revocation_id", state.revocation_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            field_text(state.trust, "revocation_file_sha256", state.revocation_digest);
        status != EvidenceError::kOk) {
        return status;
    }
    state.transition_date = transition_date;
    return EvidenceError::kOk;
}

EvidenceError verify_forward_transition(int trust_fd, int admission_fd, int evidence_fd,
                                        const ChainState& previous, std::string_view transition_id,
                                        const std::string& bootstrap_key, ChainState& next,
                                        std::uint64_t& chain_bytes) {

    const std::string relative = "transitions/" + std::string(transition_id) + "/transition.record";
    std::string transition_bytes;
    if (const EvidenceError status = read_tree_file(trust_fd, relative, kMaxCanonicalRecordBytes,
                                                    transition_bytes, FilePolicy::kTrustFamily);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = add_chain_bytes(chain_bytes, transition_bytes.size());
        status != EvidenceError::kOk) {
        return status;
    }
    ClosedRecord transition;
    if (const EvidenceError status =
            parse_closed(transition_bytes, kTrustTransitionKeys, transition);
        status != EvidenceError::kOk) {
        return status;
    }
    if (field_is(transition, "record_kind", "trust_transition") != EvidenceError::kOk ||
        field_is(transition, "transition_id", transition_id) != EvidenceError::kOk ||
        field_is(transition, "schema_version", "1") != EvidenceError::kOk) {
        return EvidenceError::kSchema;
    }
    std::string previous_transition_id;
    if (const EvidenceError status =
            field_text(previous.trust, "terminal_transition_id", previous_transition_id);
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t previous_generation = 0;
    if (const EvidenceError status =
            field_u64(previous.trust, "trust_generation", previous_generation);
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t generation = 0;
    if (const EvidenceError status = field_u64(transition, "trust_generation", generation);
        status != EvidenceError::kOk) {
        return status;
    }
    if (generation < 1 || generation > 64) {
        return EvidenceError::kSchema;
    }
    if (generation != previous_generation + 1) {
        return EvidenceError::kState;
    }
    const std::string previous_path =
        previous_transition_id == "genesis-v1"
            ? "genesis/transition.record"
            : "transitions/" + previous_transition_id + "/transition.record";
    if (field_is(transition, "predecessor_transition_id", previous_transition_id) !=
            EvidenceError::kOk ||
        field_is(transition, "predecessor_transition_path", previous_path) != EvidenceError::kOk) {
        return EvidenceError::kSchema;
    }
    std::string previous_transition_bytes;
    if (const EvidenceError status =
            read_tree_file(trust_fd, previous_path, kMaxCanonicalRecordBytes,
                           previous_transition_bytes, FilePolicy::kTrustFamily);
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t predecessor_size = 0;
    if (const EvidenceError status =
            field_u64(transition, "predecessor_transition_file_size", predecessor_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (predecessor_size != previous_transition_bytes.size() ||
        field_is(transition, "predecessor_transition_file_sha256",
                 digest_hex(previous_transition_bytes)) != EvidenceError::kOk) {
        return EvidenceError::kDigest;
    }
    std::string previous_trust_id;
    if (const EvidenceError status = field_text(previous.trust, "trust_id", previous_trust_id);
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t predecessor_trust_size = 0;
    if (const EvidenceError status =
            field_u64(transition, "predecessor_successor_trust_file_size", predecessor_trust_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (field_is(transition, "predecessor_successor_trust_id", previous_trust_id) !=
            EvidenceError::kOk ||
        predecessor_trust_size != previous.trust_bytes.size() ||
        field_is(transition, "predecessor_successor_trust_file_sha256",
                 digest_hex(previous.trust_bytes)) != EvidenceError::kOk) {
        return EvidenceError::kDigest;
    }
    if (field_is(transition, "bootstrap_key_sha256", digest_hex(bootstrap_key)) !=
        EvidenceError::kOk) {
        return EvidenceError::kDigest;
    }
    std::string signer_id;
    if (const EvidenceError status = field_text(transition, "signer_key_id", signer_id);
        status != EvidenceError::kOk) {
        return status;
    }
    const AuthorityEntry* signer = find_authority(previous.registry, signer_id);
    if (signer == nullptr) {
        return EvidenceError::kAuthorization;
    }
    const std::string signature_id = "transition-signature-" + std::string(transition_id);
    if (field_is(transition, "signature_id", signature_id) != EvidenceError::kOk) {
        return EvidenceError::kSchema;
    }
    std::string blob_id;
    std::string index_id;
    if (const EvidenceError status = field_text(transition, "signature_blob_id", blob_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_text(transition, "signature_index_id", index_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (require_id(blob_id) != EvidenceError::kOk || require_id(index_id) != EvidenceError::kOk) {
        return EvidenceError::kSchema;
    }
    const std::string directory = "transitions/" + std::string(transition_id);
    std::string signature;
    std::string index_bytes;
    std::string verification_bytes;
    ClosedRecord index;
    ClosedRecord verification;
    if (const EvidenceError status = load_signature_parts(
            trust_fd, directory + "/signature.bin", trust_fd, directory + "/signature.index",
            trust_fd, directory + "/verification.record", blob_id, index_id,
            directory + "/signature.bin", signature, index, index_bytes, verification,
            verification_bytes, chain_bytes, FilePolicy::kTrustFamily, EvidenceError::kState);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = authenticate_transition_payload(
            trust_fd, next, transition, transition_id, directory + "/transition.payload", signature,
            signer->public_key, verification, verification_bytes, index_bytes, signature_id,
            blob_id, index_id, chain_bytes);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string transition_date;
    if (const EvidenceError status = field_text(transition, "transition_date", transition_date);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_date(transition_date); status != EvidenceError::kOk) {
        return status;
    }
    if (transition_date < previous.transition_date ||
        !date_in_interval(transition_date, signer->valid_from, signer->valid_until)) {
        return EvidenceError::kDate;
    }
    if (signer->role != "trust_admin" || signer->revoked != "no") {
        return EvidenceError::kAuthorization;
    }
    if (const EvidenceError status = field_is(verification, "signer_key_source", "blob");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            field_binds(verification, "signer_key_id", signer->entry_id, EvidenceError::kState);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_is(verification, "verification_date", transition_date);
        status != EvidenceError::kOk) {
        return record_field(verification, "verification_date") == nullptr ? status
                                                                          : EvidenceError::kDate;
    }
    std::string trust_id;
    if (const EvidenceError status = field_text(transition, "successor_trust_id", trust_id);
        status != EvidenceError::kOk) {
        return status;
    }
    const std::string successor_path = "admission/trust/trust-" + trust_id + ".record";
    if (field_is(transition, "successor_trust_path", successor_path) != EvidenceError::kOk) {
        return EvidenceError::kPath;
    }
    if (const EvidenceError status =
            read_trust_state(admission_fd, trust_id, next.trust_bytes, next.trust, chain_bytes);
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t successor_size = 0;
    if (const EvidenceError status =
            field_u64(transition, "successor_trust_file_size", successor_size);
        status != EvidenceError::kOk) {
        return status;
    }
    if (successor_size != next.trust_bytes.size() ||
        field_is(transition, "successor_trust_file_sha256", digest_hex(next.trust_bytes)) !=
            EvidenceError::kOk ||
        field_is(next.trust, "effective_date", transition_date) != EvidenceError::kOk ||
        field_is(next.trust, "terminal_transition_id", transition_id) != EvidenceError::kOk ||
        field_is(next.trust, "terminal_transition_path", relative) != EvidenceError::kOk) {
        return EvidenceError::kSchema;
    }
    if (const EvidenceError status = check_one_way_binding(next.trust, &transition);
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t trust_generation = 0;
    if (field_u64(next.trust, "trust_generation", trust_generation) != EvidenceError::kOk ||
        trust_generation != generation) {
        return EvidenceError::kSchema;
    }
    if (field_is(next.trust, "predecessor_trust_id", previous_trust_id) != EvidenceError::kOk) {
        return EvidenceError::kSchema;
    }
    std::uint64_t predecessor_file_size = 0;
    if (field_u64(next.trust, "predecessor_file_size", predecessor_file_size) !=
            EvidenceError::kOk ||
        predecessor_file_size != previous.trust_bytes.size() ||
        field_is(next.trust, "predecessor_file_sha256", digest_hex(previous.trust_bytes)) !=
            EvidenceError::kOk) {
        return EvidenceError::kDigest;
    }
    if (field_is(next.trust, "bootstrap_key_sha256", digest_hex(bootstrap_key)) !=
        EvidenceError::kOk) {
        return EvidenceError::kDigest;
    }
    if (field_u64(next.trust, "revocation_generation", next.revocation_generation) !=
            EvidenceError::kOk ||
        next.revocation_generation < previous.revocation_generation) {
        return EvidenceError::kAuthorization;
    }
    if (const EvidenceError status = field_text(next.trust, "revocation_id", next.revocation_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            field_text(next.trust, "revocation_file_sha256", next.revocation_digest);
        status != EvidenceError::kOk) {
        return status;
    }
    if (next.revocation_generation == previous.revocation_generation &&
        (next.revocation_id != previous.revocation_id ||
         next.revocation_digest != previous.revocation_digest)) {
        return EvidenceError::kAuthorization;
    }
    std::string registry_id;
    if (const EvidenceError status = field_text(next.trust, "registry_id", registry_id);
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t registry_size = 0;
    if (const EvidenceError status =
            field_u64(next.trust, "authority_registry_file_size", registry_size);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string registry_digest;
    if (const EvidenceError status =
            field_text(next.trust, "authority_registry_file_sha256", registry_digest);
        status != EvidenceError::kOk) {
        return status;
    }
    if (registry_digest == previous.registry.digest) {
        if (registry_id != previous.registry.registry_id ||
            registry_size != previous.registry.bytes.size()) {
            return EvidenceError::kDigest;
        }
        next.registry = previous.registry;
    } else {
        if (const EvidenceError status =
                load_registry(admission_fd, evidence_fd, registry_id, transition_date,
                              bootstrap_key, &previous.registry, next.registry, chain_bytes);
            status != EvidenceError::kOk) {
            return status;
        }
        if (next.registry.bytes.size() != registry_size ||
            next.registry.digest != registry_digest) {
            return EvidenceError::kDigest;
        }
    }
    if (field_is(transition, "registry_id", registry_id) != EvidenceError::kOk) {
        return EvidenceError::kSchema;
    }
    std::uint64_t transition_registry_size = 0;
    if (field_u64(transition, "authority_registry_file_size", transition_registry_size) !=
            EvidenceError::kOk ||
        transition_registry_size != next.registry.bytes.size() ||
        field_is(transition, "authority_registry_file_sha256", next.registry.digest) !=
            EvidenceError::kOk) {
        return EvidenceError::kDigest;
    }
    next.transition_date = transition_date;
    return EvidenceError::kOk;
}

EvidenceError reread_retained_current(PreparedCapability& prepared) {
    struct stat status{};
    if (fstat(prepared.current_fd.get(), &status) != 0) {
        return EvidenceError::kIo;
    }
    if (!S_ISREG(status.st_mode) ||
        static_cast<std::uint64_t>(status.st_dev) != prepared.current_device ||
        static_cast<std::uint64_t>(status.st_ino) != prepared.current_inode ||
        static_cast<std::uint64_t>(status.st_size) != prepared.current_size) {
        return EvidenceError::kRace;
    }
    if (lseek(prepared.current_fd.get(), 0, SEEK_SET) < 0) {
        return EvidenceError::kIo;
    }
    std::string again(static_cast<std::size_t>(prepared.current_size), '\0');
    std::size_t got = 0;
    while (got < again.size()) {
        const ssize_t count =
            read(prepared.current_fd.get(), again.data() + got, again.size() - got);
        if (count < 0) {
            return EvidenceError::kIo;
        }
        if (count == 0) {
            return EvidenceError::kSize;
        }
        got += static_cast<std::size_t>(count);
    }
    if (again != prepared.current_bytes || again != prepared.successor_bytes) {
        return EvidenceError::kRace;
    }
    if (!prepared.trust_fd || !prepared.payload_fd || prepared.payload_path.empty()) {
        return EvidenceError::kInternal;
    }
    struct stat retained{};
    if (fstat(prepared.payload_fd.get(), &retained) != 0) {
        return EvidenceError::kIo;
    }
    if (!S_ISREG(retained.st_mode) ||
        static_cast<std::uint32_t>(retained.st_mode & 0777) != prepared.payload_mode ||
        static_cast<std::uint64_t>(retained.st_nlink) != prepared.payload_nlink ||
        static_cast<std::uint64_t>(retained.st_dev) != prepared.payload_device ||
        static_cast<std::uint64_t>(retained.st_ino) != prepared.payload_inode ||
        static_cast<std::uint64_t>(retained.st_size) != prepared.payload_size) {
        return EvidenceError::kRace;
    }
    std::string retained_bytes;
    const EvidenceError retained_read = read_limited(
        prepared.payload_fd.get(), prepared.payload_size, kMaxCanonicalRecordBytes, retained_bytes);
    if (retained_read == EvidenceError::kIo) {
        return retained_read;
    }
    if (retained_read != EvidenceError::kOk || retained_bytes != prepared.payload_bytes) {
        return EvidenceError::kRace;
    }
    struct stat payload_status{};
    Fd payload;
    if (const EvidenceError opened =
            open_readonly(prepared.trust_fd.get(), prepared.payload_path, payload, payload_status,
                          FilePolicy::kTrustFamily);
        opened != EvidenceError::kOk) {
        return EvidenceError::kRace;
    }
    if (static_cast<std::uint64_t>(payload_status.st_dev) != prepared.payload_device ||
        static_cast<std::uint64_t>(payload_status.st_ino) != prepared.payload_inode ||
        static_cast<std::uint64_t>(payload_status.st_size) != prepared.payload_size ||
        static_cast<std::uint64_t>(payload_status.st_dev) !=
            static_cast<std::uint64_t>(retained.st_dev) ||
        static_cast<std::uint64_t>(payload_status.st_ino) !=
            static_cast<std::uint64_t>(retained.st_ino)) {
        return EvidenceError::kRace;
    }
    std::string payload_again;
    if (const EvidenceError reread_payload = read_opened(payload.get(), prepared.payload_size,
                                                         kMaxCanonicalRecordBytes, payload_again);
        reread_payload != EvidenceError::kOk || payload_again != prepared.payload_bytes) {
        return EvidenceError::kRace;
    }
    return EvidenceError::kOk;
}

EvidenceError prepare_authenticated_capability(
    std::string_view operation, std::string_view authorization_bytes,
    std::string_view canonical_authorization_path, std::string_view target_id,
    std::string_view trust_root, std::string_view evidence_root, std::string_view admission_root,
    std::string_view terminal_trust_state_id, std::string_view terminal_effective_date,
    std::string_view signer_key_id, std::string_view key_valid_from,
    std::string_view key_valid_until, PreparedCapability& prepared) {
    if (operation != "stale-lock" && operation != "reservation-deletion" &&
        operation != "quarantine-deletion") {
        return EvidenceError::kAuthorization;
    }
    if (authorization_bytes.empty()) {
        return EvidenceError::kUnavailable;
    }
    if (const EvidenceError status = require_id(terminal_trust_state_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_id(signer_key_id); status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_date(terminal_effective_date);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_date(key_valid_from); status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = require_date(key_valid_until); status != EvidenceError::kOk) {
        return status;
    }
    if (key_valid_from > key_valid_until) {
        return EvidenceError::kDate;
    }
    if (canonical_authorization_path.empty() || target_id.empty()) {
        return EvidenceError::kPath;
    }
    Fd trust;
    Fd evidence;
    Fd admission;
    if (const EvidenceError status = open_root(trust_root, trust); status != EvidenceError::kOk) {
        return unavailable_input(status);
    }
    if (const EvidenceError status = open_root(evidence_root, evidence);
        status != EvidenceError::kOk) {
        return unavailable_input(status);
    }
    if (const EvidenceError status = open_root(admission_root, admission);
        status != EvidenceError::kOk) {
        return unavailable_input(status);
    }
    if (const EvidenceError status =
            directory_identity(evidence.get(), prepared.evidence_device, prepared.evidence_inode,
                               prepared.evidence_kind);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status =
            directory_identity(admission.get(), prepared.admission_device, prepared.admission_inode,
                               prepared.admission_kind);
        status != EvidenceError::kOk) {
        return status;
    }
    struct stat current_status{};
    if (const EvidenceError status =
            open_readonly(trust.get(), "current", prepared.current_fd, current_status);
        status != EvidenceError::kOk) {
        return unavailable_input(status);
    }
    if (current_status.st_size == 0 ||
        static_cast<std::uint64_t>(current_status.st_size) > kMaxCanonicalRecordBytes) {
        return EvidenceError::kSize;
    }
    prepared.current_device = static_cast<std::uint64_t>(current_status.st_dev);
    prepared.current_inode = static_cast<std::uint64_t>(current_status.st_ino);
    prepared.current_size = static_cast<std::uint64_t>(current_status.st_size);
    if (const EvidenceError status = read_opened(prepared.current_fd.get(), prepared.current_size,
                                                 kMaxCanonicalRecordBytes, prepared.current_bytes);
        status != EvidenceError::kOk) {
        return unavailable_input(status);
    }

    ClosedRecord locator;
    if (const EvidenceError status = parse_closed(prepared.current_bytes, kTrustStateKeys, locator);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string terminal_id;
    std::string terminal_path;
    if (const EvidenceError status = field_text(locator, "terminal_transition_id", terminal_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_text(locator, "terminal_transition_path", terminal_path);
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t terminal_generation = 0;
    if (const EvidenceError status = field_u64(locator, "trust_generation", terminal_generation);
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t chain_bytes = prepared.current_bytes.size();
    std::vector<std::string> forward_ids;
    if (terminal_id == "genesis-v1") {
        if (terminal_path != "genesis/transition.record" || terminal_generation != 1) {
            return EvidenceError::kSchema;
        }
    } else {
        if (terminal_generation < 1 || terminal_generation > 64) {
            return EvidenceError::kSchema;
        }
        if (terminal_generation < 2) {
            return EvidenceError::kState;
        }
        std::string walk_id = terminal_id;
        std::string walk_path = terminal_path;
        std::uint64_t walk_generation = terminal_generation;
        std::set<std::string> seen_ids;
        std::set<std::string> seen_paths;
        std::vector<std::string> backward;
        while (walk_id != "genesis-v1") {
            if (!seen_ids.insert(walk_id).second || !seen_paths.insert(walk_path).second) {
                return EvidenceError::kSchema;
            }
            if (backward.size() >= 63) {
                return EvidenceError::kSize;
            }
            if (const EvidenceError status = require_id(walk_id); status != EvidenceError::kOk) {
                return status;
            }
            const std::string expected = "transitions/" + walk_id + "/transition.record";
            if (walk_path != expected) {
                return EvidenceError::kPath;
            }
            std::string bytes;
            if (const EvidenceError status =
                    read_tree_file(trust.get(), walk_path, kMaxCanonicalRecordBytes, bytes,
                                   FilePolicy::kTrustFamily);
                status != EvidenceError::kOk) {
                return status;
            }
            if (const EvidenceError status = add_chain_bytes(chain_bytes, bytes.size());
                status != EvidenceError::kOk) {
                return status;
            }

            ClosedRecord located;
            if (const EvidenceError status = parse_closed(bytes, kTrustTransitionKeys, located);
                status != EvidenceError::kOk) {
                return status;
            }
            std::uint64_t file_generation = 0;
            if (const EvidenceError status =
                    field_u64(located, "trust_generation", file_generation);
                status != EvidenceError::kOk) {
                return status;
            }
            if (file_generation < 1 || file_generation > 64 || walk_generation == 0) {
                return EvidenceError::kSchema;
            }
            if (file_generation != walk_generation) {
                return EvidenceError::kState;
            }
            backward.push_back(walk_id);
            if (const EvidenceError status =
                    field_text(located, "predecessor_transition_id", walk_id);
                status != EvidenceError::kOk) {
                return status;
            }
            if (const EvidenceError status =
                    field_text(located, "predecessor_transition_path", walk_path);
                status != EvidenceError::kOk) {
                return status;
            }
            --walk_generation;
        }
        if (walk_path != "genesis/transition.record") {
            return EvidenceError::kSchema;
        }
        if (walk_generation != 1) {
            return EvidenceError::kState;
        }
        forward_ids.assign(backward.rbegin(), backward.rend());
    }
    std::string bootstrap_key;
    ChainState state;
    if (const EvidenceError status = verify_genesis(trust.get(), admission.get(), evidence.get(),
                                                    state, bootstrap_key, chain_bytes);
        status != EvidenceError::kOk) {
        return status;
    }
    for (const std::string& transition_id : forward_ids) {
        ChainState successor;
        if (const EvidenceError status =
                verify_forward_transition(trust.get(), admission.get(), evidence.get(), state,
                                          transition_id, bootstrap_key, successor, chain_bytes);
            status != EvidenceError::kOk) {
            return status;
        }
        state = std::move(successor);
    }
    if (state.trust_bytes != prepared.current_bytes) {
        return EvidenceError::kRace;
    }
    prepared.successor_bytes = state.trust_bytes;
    prepared.payload_path = state.payload_path;
    prepared.payload_device = state.payload_device;
    prepared.payload_inode = state.payload_inode;
    prepared.payload_mode = state.payload_mode;
    prepared.payload_nlink = state.payload_nlink;
    prepared.payload_size = state.payload_size;
    prepared.payload_bytes = std::move(state.payload_bytes);
    prepared.payload_fd = std::move(state.payload_fd);
    prepared.trust_fd = Fd(dup(trust.get()));
    if (!prepared.trust_fd) {
        return EvidenceError::kIo;
    }
    std::string authenticated_trust_id;
    std::string authenticated_date;
    if (const EvidenceError status = field_text(state.trust, "trust_id", authenticated_trust_id);
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = field_text(state.trust, "effective_date", authenticated_date);
        status != EvidenceError::kOk) {
        return status;
    }
    if (authenticated_trust_id != terminal_trust_state_id ||
        authenticated_date != terminal_effective_date) {
        return EvidenceError::kAuthorization;
    }
    std::string family_target;
    std::string signature_id;
    std::string blob_id;
    std::string index_id;
    std::string relative_path;
    std::string bound_target;
    std::string family_name;
    if (operation == "stale-lock") {
        StaleLockAuthorization authorization;
        if (const EvidenceError status =
                parse_stale_lock_authorization(authorization_bytes, authorization);
            status != EvidenceError::kOk) {
            return status;
        }
        family_name = "stale_lock_authorization";
        family_target = "stale-" + authorization.authorization_id;
        signature_id = "stalelock-signature-" + authorization.authorization_id;
        blob_id = "stalelock-sigblob-" + authorization.authorization_id;
        index_id = "stalelock-sigindex-" + authorization.authorization_id;
        relative_path = stale_authorization_path(authorization.authorization_id);
        bound_target = authorization.lock_path;
    } else {
        std::vector<std::string> lines;
        if (const EvidenceError status = split_canonical(authorization_bytes, lines);
            status != EvidenceError::kOk) {
            return status;
        }
        std::string target_kind;
        if (lines.size() > 3) {
            if (const EvidenceError status = take_value(lines[3], "target_kind", target_kind);
                status != EvidenceError::kOk) {
                return status;
            }
        }
        if (operation == "reservation-deletion") {
            ReservationDeletionAuthorization authorization;
            if (const EvidenceError status =
                    parse_reservation_deletion(authorization_bytes, authorization);
                status != EvidenceError::kOk) {
                return status;
            }
            bound_target = authorization.target_id;
            family_target = "deletion-" + authorization.authorization_id;
            signature_id = "deletion-signature-" + authorization.authorization_id;
            blob_id = "deletion-sigblob-" + authorization.authorization_id;
            index_id = "deletion-sigindex-" + authorization.authorization_id;
            relative_path = deletion_authorization_path(authorization.authorization_id);
        } else {
            QuarantineDeletionAuthorization authorization;
            if (const EvidenceError status =
                    parse_quarantine_deletion(authorization_bytes, authorization);
                status != EvidenceError::kOk) {
                return status;
            }
            bound_target = authorization.target_id;
            family_target = "deletion-" + authorization.authorization_id;
            signature_id = "deletion-signature-" + authorization.authorization_id;
            blob_id = "deletion-sigblob-" + authorization.authorization_id;
            index_id = "deletion-sigindex-" + authorization.authorization_id;
            relative_path = deletion_authorization_path(authorization.authorization_id);
        }
        family_name = "deletion_authorization";
        static_cast<void>(target_kind);
    }
    if (canonical_authorization_path != relative_path || target_id != bound_target) {
        return EvidenceError::kAuthorization;
    }
    std::string stored_authorization;
    if (const EvidenceError status = read_tree_file(admission.get(), relative_path,
                                                    kMaxCanonicalRecordBytes, stored_authorization);
        status != EvidenceError::kOk) {
        return status;
    }
    if (stored_authorization != authorization_bytes) {
        return EvidenceError::kDigest;
    }
    const std::string blob_path = "benchmarks/generative_phase9c/signatures/" + blob_id + ".bin";
    const std::string index_path = "signature-indexes/" + index_id + ".index";
    const std::string verification_path = "signature-records/" + signature_id + ".record";
    std::string signature;
    std::string index_bytes;
    std::string verification_bytes;
    ClosedRecord index;
    ClosedRecord verification;
    if (const EvidenceError status = load_signature_parts(
            evidence.get(), blob_path, admission.get(), index_path, admission.get(),
            verification_path, blob_id, index_id, blob_path, signature, index, index_bytes,
            verification, verification_bytes, chain_bytes);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string record_signer;
    if (const EvidenceError status = field_text(verification, "signer_key_id", record_signer);
        status != EvidenceError::kOk) {
        return status;
    }
    const AuthorityEntry* entry = find_authority(state.registry, record_signer);
    if (entry == nullptr) {
        return EvidenceError::kAuthorization;
    }
    if (const EvidenceError status = check_verification_common(
            verification, signature_id, family_target, authorization_bytes.size(),
            digest_hex(authorization_bytes), blob_id, index_id, index_bytes.size(),
            digest_hex(index_bytes), digest_hex(signature), digest_hex(entry->public_key));
        status != EvidenceError::kOk) {
        return status;
    }
    if (field_is(verification, "signer_key_source", "blob") != EvidenceError::kOk ||
        field_is(verification, "key_index_id", entry->entry_id) != EvidenceError::kOk) {
        return EvidenceError::kAuthorization;
    }
    std::uint64_t key_index_size = 0;
    if (field_u64(verification, "key_index_size", key_index_size) != EvidenceError::kOk ||
        key_index_size != entry->key_index_size ||
        field_is(verification, "key_index_sha256", entry->key_index_sha256) != EvidenceError::kOk ||
        field_is(verification, "public_key_sha256", entry->public_key_sha256) !=
            EvidenceError::kOk) {
        return EvidenceError::kDigest;
    }
    std::string verification_date;
    if (const EvidenceError status =
            field_text(verification, "verification_date", verification_date);
        status != EvidenceError::kOk) {
        return status;
    }
    VerificationRequest request;
    request.message = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(authorization_bytes.data()),
        authorization_bytes.size());
    request.signature = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(signature.data()), signature.size());
    request.public_key = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(entry->public_key.data()), entry->public_key.size());
    const std::string authorization_digest = digest_hex(authorization_bytes);
    request.authorization_digest = authorization_digest;
    request.signature_family_id = family_name;
    request.terminal_trust_state_id = authenticated_trust_id;
    request.terminal_effective_date = authenticated_date;
    request.registry_entry_id = entry->entry_id;
    request.signer_key_id = entry->entry_id;
    request.key_valid_from = entry->valid_from;
    request.key_valid_until = entry->valid_until;
    request.role = "trust_admin";
    request.revocation_result = "unrevoked";
    request.verification_date = authenticated_date;
    request.operation = operation;
    request.canonical_authorization_path = relative_path;
    request.target_id = bound_target;
    const VerificationDecision decision = verify_ed25519(request);
    if (decision.status == VerificationStatus::kUnavailable) {
        return EvidenceError::kUnavailable;
    }
    if (decision.status == VerificationStatus::kMalformed) {
        return EvidenceError::kSchema;
    }
    if (decision.status != VerificationStatus::kAccepted || !decision.accepted()) {
        return EvidenceError::kAuthorization;
    }
    if (decision.authorization_digest != request.authorization_digest ||
        decision.signature_family_id != family_name ||
        decision.terminal_trust_state_id != authenticated_trust_id ||
        decision.terminal_effective_date != authenticated_date ||
        decision.registry_entry_id != entry->entry_id ||
        decision.signer_key_id != entry->entry_id || decision.key_valid_from != entry->valid_from ||
        decision.key_valid_until != entry->valid_until || decision.role != "trust_admin" ||
        decision.revocation_result != "unrevoked" ||
        decision.verification_date != authenticated_date || decision.operation != operation ||
        decision.canonical_authorization_path != relative_path ||
        decision.target_id != bound_target) {
        return EvidenceError::kAuthorization;
    }
    if (!date_in_interval(authenticated_date, entry->valid_from, entry->valid_until)) {
        return EvidenceError::kDate;
    }
    if (entry->role != "trust_admin" || entry->revoked != "no" ||
        verification_date != authenticated_date || signer_key_id != entry->entry_id ||
        key_valid_from != entry->valid_from || key_valid_until != entry->valid_until ||
        record_signer != entry->entry_id) {
        return EvidenceError::kAuthorization;
    }
    prepared.operation = std::string(operation);
    prepared.authorization_bytes = std::string(authorization_bytes);
    prepared.path = std::string(relative_path);
    prepared.target_id = std::string(bound_target);
    prepared.evidence_root = std::string(evidence_root);
    prepared.admission_root = std::string(admission_root);
    prepared.trust_state_id = authenticated_trust_id;
    prepared.effective_date = authenticated_date;
    prepared.signer_key_id = entry->entry_id;
    prepared.required_role = "trust_admin";
    prepared.key_valid_from = entry->valid_from;
    prepared.key_valid_until = entry->valid_until;
    prepared.revocation_decision = "unrevoked";
    prepared.signature_result = "verified";
    return EvidenceError::kOk;
}

} // namespace

EvidenceError cleanup_reservation(EvidenceLock& lock, std::string_view admission_root,
                                  std::string_view terminal_effective_date,
                                  std::string_view terminal_trust_state_id,
                                  std::string_view signer_key_id,
                                  std::string_view authorization_bytes,
                                  const AuthenticatedCapability* capability,
                                  std::string_view deletion_list_bytes, EvidenceTime* clock) {
    if (!lock.held()) {
        return EvidenceError::kLock;
    }
    if (const EvidenceError status =
            screen_capability(capability, "reservation-deletion", lock.evidence_root(),
                              admission_root, terminal_trust_state_id, terminal_effective_date,
                              signer_key_id, authorization_bytes, lock.root_fd_);
        status != EvidenceError::kOk) {
        return status;
    }
    ReservationDeletionAuthorization authorization;
    if (const EvidenceError status = parse_reservation_deletion(authorization_bytes, authorization);
        status != EvidenceError::kOk) {
        return status;
    }
    if (capability->target_id() != authorization.target_id ||
        capability->canonical_authorization_path() !=
            deletion_authorization_path(authorization.authorization_id)) {
        return EvidenceError::kAuthorization;
    }
    ReservationDeletionList list;
    if (const EvidenceError status = parse_reservation_deletion_list(deletion_list_bytes, list);
        status != EvidenceError::kOk) {
        return status;
    }
    if (list.list_id != authorization.deletion_list_id ||
        list.reservation_id != authorization.target_id ||
        list.owner_run_id != authorization.owner_run_id ||
        deletion_list_bytes.size() != authorization.deletion_list_file_size ||
        digest_hex(deletion_list_bytes) != authorization.deletion_list_file_sha256) {
        return EvidenceError::kDigest;
    }
    Fd admission;
    if (const EvidenceError status = open_root(admission_root, admission);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string stored_authorization;
    if (const EvidenceError status = read_record_file(
            admission.get(), deletion_authorization_path(authorization.authorization_id),
            stored_authorization);
        status != EvidenceError::kOk) {
        return status;
    }
    if (stored_authorization != authorization_bytes) {
        return EvidenceError::kDigest;
    }
    std::string stored_list;
    if (const EvidenceError status = read_record_file(
            admission.get(), deletion_list_path(authorization.target_id), stored_list);
        status != EvidenceError::kOk) {
        return status;
    }
    if (stored_list != deletion_list_bytes) {
        return EvidenceError::kDigest;
    }
    const std::string pending = pending_path(authorization.authorization_id);
    const std::string completion = completion_path(authorization.authorization_id);
    std::string completion_bytes;
    const EvidenceError completion_read =
        read_record_file(admission.get(), completion, completion_bytes);
    if (completion_read == EvidenceError::kOk) {
        CleanupCompletionRecord complete;
        if (const EvidenceError status = parse_cleanup_completion(completion_bytes, complete);
            status != EvidenceError::kOk) {
            return status;
        }
        std::string pending_bytes;
        if (const EvidenceError status = read_record_file(admission.get(), pending, pending_bytes);
            status != EvidenceError::kOk) {
            return status;
        }
        if (complete.authorization_id != authorization.authorization_id ||
            complete.authorization_file_size != authorization_bytes.size() ||
            complete.authorization_file_sha256 != digest_hex(authorization_bytes) ||
            complete.deletion_list_id != list.list_id ||
            complete.deletion_list_file_size != deletion_list_bytes.size() ||
            complete.deletion_list_file_sha256 != digest_hex(deletion_list_bytes) ||
            complete.owner_run_id != authorization.owner_run_id ||
            complete.pending_file_size != pending_bytes.size() ||
            complete.pending_file_sha256 != digest_hex(pending_bytes)) {
            return EvidenceError::kDigest;
        }
        return EvidenceError::kOk;
    }
    if (completion_read != EvidenceError::kAbsent) {
        return completion_read;
    }
    std::string pending_bytes;
    const EvidenceError pending_read = read_record_file(admission.get(), pending, pending_bytes);
    if (pending_read == EvidenceError::kOk) {
        CleanupPendingRecord existing;
        if (const EvidenceError status = parse_cleanup_pending(pending_bytes, existing);
            status != EvidenceError::kOk) {
            return status;
        }
        if (existing.authorization_id != authorization.authorization_id ||
            existing.authorization_file_size != authorization_bytes.size() ||
            existing.authorization_file_sha256 != digest_hex(authorization_bytes) ||
            existing.deletion_list_id != list.list_id ||
            existing.deletion_list_file_size != deletion_list_bytes.size() ||
            existing.deletion_list_file_sha256 != digest_hex(deletion_list_bytes) ||
            existing.owner_run_id != authorization.owner_run_id) {
            return EvidenceError::kAuthorization;
        }
    } else if (pending_read != EvidenceError::kAbsent) {
        return pending_read;
    }
    {
        std::string reservation_bytes;
        if (const EvidenceError status =
                read_record_file(lock.root_fd_, authorization.reservation_path, reservation_bytes);
            status != EvidenceError::kOk) {
            return status;
        }
        if (reservation_bytes.size() != authorization.target_file_size ||
            digest_hex(reservation_bytes) != authorization.target_file_sha256) {
            return EvidenceError::kDigest;
        }
        EvidenceReservation reservation;
        if (const EvidenceError status = parse_evidence_reservation(reservation_bytes, reservation);
            status != EvidenceError::kOk) {
            return status;
        }
        if (reservation.owner_run_id != authorization.owner_run_id) {
            return EvidenceError::kOwner;
        }
        int reservation_lines = 0;
        for (const DeletionListEntry& entry : list.entries) {
            if (const EvidenceError status = validate_evidence_relative_path(entry.relative_path);
                status != EvidenceError::kOk) {
                return status;
            }
            if (forbidden_cleanup_root(entry.relative_path)) {
                return EvidenceError::kForbidden;
            }
            if (entry.object_class == CleanupObjectClass::kEvidenceReservation) {
                ++reservation_lines;
                if (entry.relative_path != authorization.reservation_path) {
                    return EvidenceError::kForbidden;
                }
            } else if (entry.object_class == CleanupObjectClass::kPartialStaging ||
                       entry.object_class == CleanupObjectClass::kTemporaryMetadata) {
                if (validate_staging(entry.relative_path, authorization.owner_run_id) !=
                    EvidenceError::kOk) {
                    return EvidenceError::kForbidden;
                }
                const ReservationEntry* match = nullptr;
                for (const ReservationEntry& candidate : reservation.entries) {
                    if (candidate.staging_relative_path == entry.relative_path) {
                        match = &candidate;
                        break;
                    }
                }
                if (match == nullptr ||
                    (entry.object_class == CleanupObjectClass::kTemporaryMetadata &&
                     (match->object_class != EvidenceObjectClass::kTemporaryMetadata ||
                      match->final_relative_path != "none")) ||
                    (entry.object_class == CleanupObjectClass::kPartialStaging &&
                     (match->final_relative_path == "none" ||
                      match->final_relative_path == entry.relative_path))) {
                    return EvidenceError::kForbidden;
                }
                for (const ReservationEntry& candidate : reservation.entries) {
                    if (candidate.final_relative_path == entry.relative_path) {
                        return EvidenceError::kForbidden;
                    }
                }
            }
            struct stat listed{};
            bool present = false;
            if (const EvidenceError checked =
                    stat_leaf(lock.root_fd_, entry.relative_path, listed, present);
                checked != EvidenceError::kOk) {
                return checked;
            }
            if (present && S_ISLNK(listed.st_mode)) {
                return EvidenceError::kSymlink;
            }
        }
        if (reservation_lines != 1) {
            return EvidenceError::kForbidden;
        }
        if (pending_read != EvidenceError::kOk) {
            std::string today;
            if (const EvidenceError status = time_of(clock).civil_date(today);
                status != EvidenceError::kOk) {
                return status;
            }
            CleanupPendingRecord created;
            created.authorization_id = authorization.authorization_id;
            created.authorization_file_size = authorization_bytes.size();
            created.authorization_file_sha256 = digest_hex(authorization_bytes);
            created.deletion_list_id = list.list_id;
            created.deletion_list_file_size = deletion_list_bytes.size();
            created.deletion_list_file_sha256 = digest_hex(deletion_list_bytes);
            created.owner_run_id = authorization.owner_run_id;
            created.creation_date = today;
            std::string bytes;
            if (const EvidenceError status = serialize_cleanup_pending(created, bytes);
                status != EvidenceError::kOk) {
                return status;
            }
            Fd parent;
            Fd file;
            std::uint64_t device = 0;
            std::uint64_t inode = 0;
            if (const EvidenceError status =
                    create_file(admission.get(), pending, bytes, file, parent, device, inode);
                status != EvidenceError::kOk) {
                return status;
            }
        }
    }
    for (const DeletionListEntry& entry : list.entries) {
        if (const EvidenceError status = unlink_matching_record(lock.root_fd_, entry);
            status != EvidenceError::kOk) {
            return status;
        }
    }
    for (const DeletionListEntry& entry : list.entries) {
        struct stat status{};
        bool present = false;
        if (const EvidenceError checked =
                stat_leaf(lock.root_fd_, entry.relative_path, status, present);
            checked != EvidenceError::kOk) {
            return checked;
        }
        if (present) {
            return EvidenceError::kPresent;
        }
    }
    std::string pending_final;
    if (const EvidenceError status = read_record_file(admission.get(), pending, pending_final);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string completed_on;
    if (const EvidenceError status = time_of(clock).civil_date(completed_on);
        status != EvidenceError::kOk) {
        return status;
    }
    CleanupCompletionRecord complete;
    complete.authorization_id = authorization.authorization_id;
    complete.pending_file_size = pending_final.size();
    complete.pending_file_sha256 = digest_hex(pending_final);
    complete.authorization_file_size = authorization_bytes.size();
    complete.authorization_file_sha256 = digest_hex(authorization_bytes);
    complete.deletion_list_id = list.list_id;
    complete.deletion_list_file_size = deletion_list_bytes.size();
    complete.deletion_list_file_sha256 = digest_hex(deletion_list_bytes);
    complete.owner_run_id = authorization.owner_run_id;
    complete.completion_date = completed_on;
    std::string complete_bytes;
    if (const EvidenceError status = serialize_cleanup_completion(complete, complete_bytes);
        status != EvidenceError::kOk) {
        return status;
    }
    Fd parent;
    Fd file;
    std::uint64_t device = 0;
    std::uint64_t inode = 0;
    return create_file(admission.get(), completion, complete_bytes, file, parent, device, inode);
}

// Section 22 inventory, closed:
// checksum_rejected, reproduction_mismatch, cap_rejected, symlink_rejected, signature_failed,
// invariant_mismatch, other_rejected: quarantine.record only.
// recipe_echo_rejected: quarantine.record and, optionally, recipe-echo.bin.
// deletion.record is created only by authorized deletion.
// No other payload filename is defined by the contract, so none is accepted.
// Section 19's evidence_reservation is the publication batch. Quarantine bytes are not
// published finals. This writer holds the evidence lock, counts the metadata record and the
// optional recipe-echo.bin once in the root sum and the directory cap, creates those files
// exclusively, synchronizes them, and on failure removes only the inodes it created.
EvidenceError create_quarantine(EvidenceLock& lock, std::string_view run_id,
                                std::string_view phase_id, std::string_view reason_code,
                                std::string_view manifest_path,
                                const std::vector<QuarantinePayload>& payloads,
                                EvidenceTime* clock) {
    if (!lock.held()) {
        return EvidenceError::kLock;
    }
    if (const EvidenceError status = require_id(run_id); status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = from_admission(validate_phase_id(phase_id));
        status != EvidenceError::kOk) {
        return status;
    }
    if (!known_reason(reason_code)) {
        return EvidenceError::kLexical;
    }
    std::vector<std::string> names;
    for (const QuarantinePayload& payload : payloads) {
        if (const EvidenceError status = from_admission(validate_filename(payload.filename));
            status != EvidenceError::kOk) {
            return status;
        }
        if (payload.filename != "recipe-echo.bin" || reason_code != "recipe_echo_rejected") {
            return EvidenceError::kForbidden;
        }
        names.push_back(payload.filename);
    }
    std::sort(names.begin(), names.end());
    if (std::adjacent_find(names.begin(), names.end()) != names.end()) {
        return EvidenceError::kDuplicate;
    }
    std::string today;
    if (const EvidenceError status = time_of(clock).civil_date(today);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string deadline;
    if (const EvidenceError status = from_admission(add_calendar_days(today, 30, deadline));
        status != EvidenceError::kOk) {
        return status;
    }
    QuarantineMetadata metadata;
    metadata.quarantine_id = std::string(run_id);
    metadata.associated_phase_id = std::string(phase_id);
    metadata.associated_run_id = std::string(run_id);
    metadata.directory_id = "quarantine/" + std::string(run_id);
    metadata.creation_date = today;
    metadata.reason_code = std::string(reason_code);
    metadata.retention_start = today;
    metadata.retention_deadline = deadline;
    if (manifest_path == "none") {
        metadata.accepted_run_manifest_path = "none";
        metadata.accepted_run_manifest_file_size = 0;
        metadata.accepted_run_manifest_file_sha256 = std::string(kZeroDigest);
    } else {
        std::string manifest_bytes;
        struct stat manifest_status{};
        if (const EvidenceError status = read_relative(lock.root_fd_, manifest_path, UINT64_MAX,
                                                       manifest_bytes, manifest_status);
            status != EvidenceError::kOk) {
            return status;
        }
        metadata.accepted_run_manifest_path = std::string(manifest_path);
        metadata.accepted_run_manifest_file_size = manifest_bytes.size();
        metadata.accepted_run_manifest_file_sha256 = digest_hex(manifest_bytes);
    }
    std::string record;
    if (const EvidenceError status = serialize_quarantine_metadata(metadata, record);
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t adding = record.size();
    for (const QuarantinePayload& payload : payloads) {
        std::uint64_t next = 0;
        if (add_uint64(adding, payload.bytes.size(), next) != AdmissionError::kOk) {
            return EvidenceError::kOverflow;
        }
        adding = next;
    }
    std::uint64_t root_sum = 0;
    if (const EvidenceError status = sum_fd(lock.root_fd_, root_sum);
        status != EvidenceError::kOk) {
        return status;
    }
    std::uint64_t projected = 0;
    if (add_uint64(root_sum, adding, projected) != AdmissionError::kOk) {
        return EvidenceError::kOverflow;
    }
    if (projected > kEvidenceCapBytes) {
        return EvidenceError::kCap;
    }
    const std::string directory = metadata.directory_id;
    const std::string record_path = directory + "/quarantine.record";
    std::uint64_t existing = 0;
    struct stat directory_status{};
    bool directory_present = false;
    if (const EvidenceError status =
            stat_leaf(lock.root_fd_, directory, directory_status, directory_present);
        status != EvidenceError::kOk) {
        return status;
    }
    if (directory_present) {
        if (S_ISLNK(directory_status.st_mode)) {
            return EvidenceError::kSymlink;
        }
        if (!S_ISDIR(directory_status.st_mode)) {
            return EvidenceError::kType;
        }
        Fd parent;
        std::string leaf;
        if (const EvidenceError status =
                walk_parent(lock.root_fd_, directory + "/quarantine.record", false, parent, leaf);
            status != EvidenceError::kOk) {
            return status;
        }
        if (const EvidenceError status = sum_fd(parent.get(), existing);
            status != EvidenceError::kOk) {
            return status;
        }
    }
    std::uint64_t directory_total = 0;
    if (add_uint64(existing, adding, directory_total) != AdmissionError::kOk) {
        return EvidenceError::kOverflow;
    }
    if (directory_total > kQuarantineDirectoryCapBytes) {
        return EvidenceError::kCap;
    }
    if (const EvidenceError status = require_absent_path(lock.root_fd_, record_path);
        status != EvidenceError::kOk) {
        return status;
    }
    for (const QuarantinePayload& payload : payloads) {
        if (const EvidenceError status =
                require_absent_path(lock.root_fd_, directory + "/" + payload.filename);
            status != EvidenceError::kOk) {
            return status;
        }
    }
    std::vector<TrackedFile> created;
    const auto fail = [&](EvidenceError status) {
        const EvidenceError cleaned = cleanup_tracked(lock.root_fd_, created);
        return cleaned == EvidenceError::kOk ? status : cleaned;
    };
    Fd parent;
    Fd file;
    std::uint64_t device = 0;
    std::uint64_t inode = 0;
    if (const EvidenceError status =
            create_file(lock.root_fd_, record_path, record, file, parent, device, inode);
        status != EvidenceError::kOk) {
        return status;
    }
    created.push_back(TrackedFile{record_path, device, inode});
    struct stat mode{};
    if (fstat(parent.get(), &mode) != 0) {
        return fail(EvidenceError::kIo);
    }
    if (!S_ISDIR(mode.st_mode) || (mode.st_mode & 0777) != 0700) {
        return fail(EvidenceError::kIo);
    }
    for (const QuarantinePayload& payload : payloads) {
        Fd payload_parent;
        Fd payload_file;
        std::uint64_t payload_device = 0;
        std::uint64_t payload_inode = 0;
        const std::string path = directory + "/" + payload.filename;
        if (const EvidenceError status =
                create_file(lock.root_fd_, path, payload.bytes, payload_file, payload_parent,
                            payload_device, payload_inode);
            status != EvidenceError::kOk) {
            return fail(status);
        }
        created.push_back(TrackedFile{path, payload_device, payload_inode});
    }
    return EvidenceError::kOk;
}

EvidenceError delete_quarantine(EvidenceLock& lock, std::string_view admission_root,
                                std::string_view terminal_effective_date,
                                std::string_view terminal_trust_state_id,
                                std::string_view signer_key_id,
                                std::string_view authorization_bytes,
                                const AuthenticatedCapability* capability, EvidenceTime* clock) {
    if (!lock.held()) {
        return EvidenceError::kLock;
    }
    if (const EvidenceError status =
            screen_capability(capability, "quarantine-deletion", lock.evidence_root(),
                              admission_root, terminal_trust_state_id, terminal_effective_date,
                              signer_key_id, authorization_bytes, lock.root_fd_);
        status != EvidenceError::kOk) {
        return status;
    }
    QuarantineDeletionAuthorization authorization;
    if (const EvidenceError status = parse_quarantine_deletion(authorization_bytes, authorization);
        status != EvidenceError::kOk) {
        return status;
    }
    if (capability->target_id() != authorization.target_id ||
        capability->canonical_authorization_path() !=
            deletion_authorization_path(authorization.authorization_id)) {
        return EvidenceError::kAuthorization;
    }
    Fd admission;
    if (const EvidenceError status = open_root(admission_root, admission);
        status != EvidenceError::kOk) {
        return status;
    }
    std::string stored;
    if (const EvidenceError status = read_record_file(
            admission.get(), deletion_authorization_path(authorization.authorization_id), stored);
        status != EvidenceError::kOk) {
        return status;
    }
    if (stored != authorization_bytes) {
        return EvidenceError::kDigest;
    }
    const std::string directory = "quarantine/" + authorization.target_id;
    const std::string metadata_path = directory + "/quarantine.record";
    const std::string journal_path = directory + "/deletion.record";
    std::string metadata_bytes;
    if (const EvidenceError status = read_record_file(lock.root_fd_, metadata_path, metadata_bytes);
        status != EvidenceError::kOk) {
        return status;
    }
    if (metadata_bytes.size() != authorization.target_file_size ||
        digest_hex(metadata_bytes) != authorization.target_file_sha256) {
        return EvidenceError::kDigest;
    }
    QuarantineMetadata metadata;
    if (const EvidenceError status = parse_quarantine_metadata(metadata_bytes, metadata);
        status != EvidenceError::kOk) {
        return status;
    }
    if (metadata.quarantine_id != authorization.target_id) {
        return EvidenceError::kSchema;
    }
    std::string journal_bytes;
    const EvidenceError journal_read = read_record_file(lock.root_fd_, journal_path, journal_bytes);
    if (journal_read == EvidenceError::kOk) {
        DeletionJournal journal;
        if (const EvidenceError status = parse_deletion_journal(journal_bytes, journal);
            status != EvidenceError::kOk) {
            return status;
        }
        if (journal.quarantine_id != authorization.target_id ||
            journal.authorization_id != authorization.authorization_id ||
            journal.authorization_file_size != authorization_bytes.size() ||
            journal.authorization_file_sha256 != digest_hex(authorization_bytes)) {
            return EvidenceError::kAuthorization;
        }
    } else if (journal_read != EvidenceError::kAbsent) {
        return journal_read;
    } else {
        std::string today;
        if (const EvidenceError status = time_of(clock).civil_date(today);
            status != EvidenceError::kOk) {
            return status;
        }
        if (const EvidenceError status = date_on_or_after(today, metadata.retention_deadline);
            status != EvidenceError::kOk) {
            return status;
        }
        DeletionJournal journal;
        journal.quarantine_id = authorization.target_id;
        journal.authorization_id = authorization.authorization_id;
        journal.authorization_file_size = authorization_bytes.size();
        journal.authorization_file_sha256 = digest_hex(authorization_bytes);
        std::string bytes;
        if (const EvidenceError status = serialize_deletion_journal(journal, bytes);
            status != EvidenceError::kOk) {
            return status;
        }
        if (const EvidenceError status =
                require_bound_root(*capability, true, lock.evidence_root(), lock.root_fd_);
            status != EvidenceError::kOk) {
            return status;
        }
        Fd parent;
        Fd file;
        std::uint64_t device = 0;
        std::uint64_t inode = 0;
        if (const EvidenceError status =
                create_file(lock.root_fd_, journal_path, bytes, file, parent, device, inode);
            status != EvidenceError::kOk) {
            return status;
        }
    }
    Fd quarantine_parent;
    std::string run_leaf;
    if (const EvidenceError status =
            walk_parent(lock.root_fd_, directory, false, quarantine_parent, run_leaf);
        status != EvidenceError::kOk) {
        return status;
    }
    const int run_fd = openat(quarantine_parent.get(), run_leaf.c_str(),
                              O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (run_fd < 0) {
        return from_errno(errno);
    }
    Fd run_directory(run_fd);
    std::vector<std::string> names;
    if (const EvidenceError status = list_names(run_directory.get(), names);
        status != EvidenceError::kOk) {
        return status;
    }
    const bool recipe_allowed = metadata.reason_code == "recipe_echo_rejected";
    bool saw_metadata = false;
    bool saw_journal = false;
    bool saw_recipe = false;
    struct ExpectedInode {
        std::string name;
        std::uint64_t device{0};
        std::uint64_t inode{0};
        std::uint64_t size{0};
    };
    std::vector<ExpectedInode> expected;
    for (const std::string& name : names) {
        struct stat status{};
        bool present = false;
        if (const EvidenceError checked =
                stat_leaf(lock.root_fd_, directory + "/" + name, status, present);
            checked != EvidenceError::kOk) {
            return checked;
        }
        if (!present) {
            return EvidenceError::kRace;
        }
        if (const EvidenceError kind = classify_file(status); kind != EvidenceError::kOk) {
            return kind;
        }
        const bool known = name == "quarantine.record" || name == "deletion.record" ||
                           (recipe_allowed && name == "recipe-echo.bin");
        if (!known) {
            return EvidenceError::kForbidden;
        }
        if (name == "quarantine.record") {
            saw_metadata = true;
        } else if (name == "deletion.record") {
            saw_journal = true;
        } else {
            saw_recipe = true;
        }
        expected.push_back(ExpectedInode{name, static_cast<std::uint64_t>(status.st_dev),
                                         static_cast<std::uint64_t>(status.st_ino),
                                         static_cast<std::uint64_t>(status.st_size)});
    }
    if (!saw_metadata || !saw_journal) {
        return EvidenceError::kAbsent;
    }
    std::vector<std::string> rescanned;
    if (const EvidenceError status = list_names(run_directory.get(), rescanned);
        status != EvidenceError::kOk) {
        return status;
    }
    if (rescanned != names) {
        return EvidenceError::kRace;
    }
    for (const ExpectedInode& item : expected) {
        struct stat again{};
        bool present = false;
        if (const EvidenceError checked =
                stat_leaf(lock.root_fd_, directory + "/" + item.name, again, present);
            checked != EvidenceError::kOk) {
            return checked;
        }
        if (!present || static_cast<std::uint64_t>(again.st_dev) != item.device ||
            static_cast<std::uint64_t>(again.st_ino) != item.inode ||
            static_cast<std::uint64_t>(again.st_size) != item.size) {
            return EvidenceError::kRace;
        }
    }
    auto unlink_expected = [&](const std::string& name) -> EvidenceError {
        const ExpectedInode* match = nullptr;
        for (const ExpectedInode& item : expected) {
            if (item.name == name) {
                match = &item;
                break;
            }
        }
        if (match == nullptr) {
            return EvidenceError::kAbsent;
        }
        struct stat status{};
        bool present = false;
        if (const EvidenceError checked =
                stat_leaf(lock.root_fd_, directory + "/" + name, status, present);
            checked != EvidenceError::kOk) {
            return checked;
        }
        if (!present || static_cast<std::uint64_t>(status.st_dev) != match->device ||
            static_cast<std::uint64_t>(status.st_ino) != match->inode ||
            static_cast<std::uint64_t>(status.st_size) != match->size) {
            return EvidenceError::kRace;
        }
        DeletionListEntry entry;
        entry.relative_path = directory + "/" + name;
        entry.deletion_entry_size = match->size;
        std::string bytes;
        struct stat ignored{};
        if (const EvidenceError read =
                read_relative(lock.root_fd_, entry.relative_path, UINT64_MAX, bytes, ignored);
            read != EvidenceError::kOk) {
            return read;
        }
        if (bytes.size() != match->size) {
            return EvidenceError::kRace;
        }
        entry.deletion_entry_sha256 = digest_hex(bytes);
        return unlink_matching_record(lock.root_fd_, entry);
    };
    if (saw_recipe) {
        if (const EvidenceError status = unlink_expected("recipe-echo.bin");
            status != EvidenceError::kOk) {
            return status;
        }
    }
    if (const EvidenceError status = unlink_expected("deletion.record");
        status != EvidenceError::kOk) {
        return status;
    }
    if (const EvidenceError status = unlink_expected("quarantine.record");
        status != EvidenceError::kOk) {
        return status;
    }
    run_directory.reset();
    if (unlinkat(quarantine_parent.get(), run_leaf.c_str(), AT_REMOVEDIR) != 0) {
        return from_errno(errno);
    }
    if (fsync(quarantine_parent.get()) != 0) {
        return EvidenceError::kIo;
    }
    return EvidenceError::kOk;
}

struct AuthenticatedCapability::State {
    const CapabilityStamp* stamp{nullptr};
    std::string operation;
    std::string authorization_bytes;
    std::string path;
    std::string target_id;
    std::string evidence_root;
    std::string admission_root;
    std::uint64_t evidence_device{0};
    std::uint64_t evidence_inode{0};
    std::uint32_t evidence_kind{0};
    std::uint64_t admission_device{0};
    std::uint64_t admission_inode{0};
    std::uint32_t admission_kind{0};
    std::string trust_state_id;
    std::string effective_date;
    std::string signer_key_id;
    std::string required_role;
    std::string key_valid_from;
    std::string key_valid_until;
    std::string revocation_decision;
    std::string signature_result;
};

AuthenticatedCapability::AuthenticatedCapability() noexcept = default;
AuthenticatedCapability::AuthenticatedCapability(AuthenticatedCapability&&) noexcept = default;
AuthenticatedCapability&
AuthenticatedCapability::operator=(AuthenticatedCapability&&) noexcept = default;
AuthenticatedCapability::~AuthenticatedCapability() = default;

bool AuthenticatedCapability::holds_authority() const noexcept {
    return state_ != nullptr && state_->stamp == &kCapabilityStamp;
}
std::string_view AuthenticatedCapability::operation() const noexcept {
    return state_ == nullptr ? std::string_view{} : std::string_view{state_->operation};
}
std::string_view AuthenticatedCapability::authorization_bytes() const noexcept {
    return state_ == nullptr ? std::string_view{} : std::string_view{state_->authorization_bytes};
}
std::string_view AuthenticatedCapability::canonical_authorization_path() const noexcept {
    return state_ == nullptr ? std::string_view{} : std::string_view{state_->path};
}
std::string_view AuthenticatedCapability::target_id() const noexcept {
    return state_ == nullptr ? std::string_view{} : std::string_view{state_->target_id};
}
std::string_view AuthenticatedCapability::evidence_root() const noexcept {
    return state_ == nullptr ? std::string_view{} : std::string_view{state_->evidence_root};
}
std::string_view AuthenticatedCapability::admission_root() const noexcept {
    return state_ == nullptr ? std::string_view{} : std::string_view{state_->admission_root};
}
std::uint64_t AuthenticatedCapability::evidence_device() const noexcept {
    return state_ == nullptr ? 0 : state_->evidence_device;
}
std::uint64_t AuthenticatedCapability::evidence_inode() const noexcept {
    return state_ == nullptr ? 0 : state_->evidence_inode;
}
std::uint32_t AuthenticatedCapability::evidence_kind() const noexcept {
    return state_ == nullptr ? 0 : state_->evidence_kind;
}
std::uint64_t AuthenticatedCapability::admission_device() const noexcept {
    return state_ == nullptr ? 0 : state_->admission_device;
}
std::uint64_t AuthenticatedCapability::admission_inode() const noexcept {
    return state_ == nullptr ? 0 : state_->admission_inode;
}
std::uint32_t AuthenticatedCapability::admission_kind() const noexcept {
    return state_ == nullptr ? 0 : state_->admission_kind;
}
std::string_view AuthenticatedCapability::terminal_trust_state_id() const noexcept {
    return state_ == nullptr ? std::string_view{} : std::string_view{state_->trust_state_id};
}
std::string_view AuthenticatedCapability::terminal_effective_date() const noexcept {
    return state_ == nullptr ? std::string_view{} : std::string_view{state_->effective_date};
}
std::string_view AuthenticatedCapability::signer_key_id() const noexcept {
    return state_ == nullptr ? std::string_view{} : std::string_view{state_->signer_key_id};
}
std::string_view AuthenticatedCapability::required_role() const noexcept {
    return state_ == nullptr ? std::string_view{} : std::string_view{state_->required_role};
}
std::string_view AuthenticatedCapability::key_valid_from() const noexcept {
    return state_ == nullptr ? std::string_view{} : std::string_view{state_->key_valid_from};
}
std::string_view AuthenticatedCapability::key_valid_until() const noexcept {
    return state_ == nullptr ? std::string_view{} : std::string_view{state_->key_valid_until};
}
std::string_view AuthenticatedCapability::revocation_decision() const noexcept {
    return state_ == nullptr ? std::string_view{} : std::string_view{state_->revocation_decision};
}
std::string_view AuthenticatedCapability::signature_verification_result() const noexcept {
    return state_ == nullptr ? std::string_view{} : std::string_view{state_->signature_result};
}

struct CapabilityResult::Impl {
    EvidenceError status{EvidenceError::kUnavailable};
    std::unique_ptr<AuthenticatedCapability> capability;
};

CapabilityResult::CapabilityResult() noexcept = default;
CapabilityResult::CapabilityResult(CapabilityResult&&) noexcept = default;
CapabilityResult& CapabilityResult::operator=(CapabilityResult&&) noexcept = default;
CapabilityResult::~CapabilityResult() = default;

EvidenceError CapabilityResult::status() const noexcept {
    return impl_ == nullptr ? EvidenceError::kUnavailable : impl_->status;
}
bool CapabilityResult::has_capability() const noexcept {
    return impl_ != nullptr && impl_->capability != nullptr && impl_->capability->holds_authority();
}
AuthenticatedCapability CapabilityResult::release_capability() {
    if (!has_capability()) {
        if (impl_ != nullptr) {
            impl_->capability.reset();
        }
        return {};
    }
    AuthenticatedCapability released = std::move(*impl_->capability);
    impl_->capability.reset();
    return released;
}

CapabilityResult AuthenticatedCapability::authenticate(
    std::string_view operation, std::string_view authorization_bytes,
    std::string_view canonical_authorization_path, std::string_view target_id,
    std::string_view trust_root, std::string_view evidence_root, std::string_view admission_root,
    std::string_view terminal_trust_state_id, std::string_view terminal_effective_date,
    std::string_view signer_key_id, std::string_view key_valid_from,
    std::string_view key_valid_until) {
    CapabilityResult result;
    result.impl_ = std::make_unique<CapabilityResult::Impl>();
    PreparedCapability prepared;
    const EvidenceError status = prepare_authenticated_capability(
        operation, authorization_bytes, canonical_authorization_path, target_id, trust_root,
        evidence_root, admission_root, terminal_trust_state_id, terminal_effective_date,
        signer_key_id, key_valid_from, key_valid_until, prepared);
    if (status != EvidenceError::kOk) {
        result.impl_->status = status;
        return result;
    }
    const EvidenceError reread = reread_retained_current(prepared);
    if (reread != EvidenceError::kOk) {
        result.impl_->status = reread;
        return result;
    }
    auto state = std::make_unique<State>();
    state->stamp = &kCapabilityStamp;
    state->operation = std::move(prepared.operation);
    state->authorization_bytes = std::move(prepared.authorization_bytes);
    state->path = std::move(prepared.path);
    state->target_id = std::move(prepared.target_id);
    state->evidence_root = std::move(prepared.evidence_root);
    state->admission_root = std::move(prepared.admission_root);
    state->evidence_device = prepared.evidence_device;
    state->evidence_inode = prepared.evidence_inode;
    state->evidence_kind = prepared.evidence_kind;
    state->admission_device = prepared.admission_device;
    state->admission_inode = prepared.admission_inode;
    state->admission_kind = prepared.admission_kind;
    state->trust_state_id = std::move(prepared.trust_state_id);
    state->effective_date = std::move(prepared.effective_date);
    state->signer_key_id = std::move(prepared.signer_key_id);
    state->required_role = std::move(prepared.required_role);
    state->key_valid_from = std::move(prepared.key_valid_from);
    state->key_valid_until = std::move(prepared.key_valid_until);
    state->revocation_decision = std::move(prepared.revocation_decision);
    state->signature_result = std::move(prepared.signature_result);
    auto capability = std::make_unique<AuthenticatedCapability>();
    capability->state_ = std::move(state);
    if (!capability->holds_authority()) {
        result.impl_->status = EvidenceError::kInternal;
        return result;
    }
    result.impl_->capability = std::move(capability);
    result.impl_->status = EvidenceError::kOk;
    return result;
}

} // namespace inferedge
