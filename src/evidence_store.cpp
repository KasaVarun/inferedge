#if defined(__linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#endif

#include "inferedge/evidence_store.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <fcntl.h>
#include <memory>
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

EvidenceError classify_file(const struct stat& status) {
    if (S_ISLNK(status.st_mode)) {
        return EvidenceError::kSymlink;
    }
    if (S_ISDIR(status.st_mode)) {
        return EvidenceError::kType;
    }
    if (!S_ISREG(status.st_mode)) {
        return EvidenceError::kSpecial;
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
    std::string_view evidence_root, std::string_view admission_root,
    std::string_view terminal_trust_state_id, std::string_view terminal_effective_date,
    std::string_view signer_key_id, std::string_view key_valid_from,
    std::string_view key_valid_until) {
    CapabilityResult result;
    result.impl_ = std::make_unique<CapabilityResult::Impl>();
    static_cast<void>(operation);
    static_cast<void>(authorization_bytes);
    static_cast<void>(canonical_authorization_path);
    static_cast<void>(target_id);
    static_cast<void>(evidence_root);
    static_cast<void>(admission_root);
    static_cast<void>(terminal_trust_state_id);
    static_cast<void>(terminal_effective_date);
    static_cast<void>(signer_key_id);
    static_cast<void>(key_valid_from);
    static_cast<void>(key_valid_until);
    result.impl_->status = EvidenceError::kUnavailable;
    return result;
}

} // namespace inferedge
