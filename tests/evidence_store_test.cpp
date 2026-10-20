#include "inferedge/evidence_store.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <set>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <vector>

namespace {

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

class TempRoot {
public:
    TempRoot() {
        char name[] = "/private/tmp/inferedge-9c6-XXXXXX";
        if (mkdtemp(name) == nullptr) {
            path_ = {};
            return;
        }
        path_ = name;
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

void plant(const std::string& root, std::string_view relative, std::string_view bytes) {
    const std::string full = join_path(root, relative);
    make_parents(full);
    const int fd = open(full.c_str(), O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC | O_WRONLY, 0600);
    ASSERT_GE(fd, 0) << full;
    ASSERT_EQ(write(fd, bytes.data(), bytes.size()), static_cast<ssize_t>(bytes.size()));
    close(fd);
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
    auto make = [](const char* pattern) {
        std::string name = pattern;
        EXPECT_NE(mkdtemp(name.data()), nullptr);
        return name;
    };
    const std::string real = make("/private/tmp/inferedge-real-XXXXXX");
    ASSERT_EQ(mkdir(join_path(real, "chain").c_str(), 0700), 0);
    ASSERT_EQ(mkdir(join_path(real, "chain/mid").c_str(), 0700), 0);
    ASSERT_EQ(mkdir(join_path(real, "chain/mid/root").c_str(), 0700), 0);
    ASSERT_EQ(mkdir(join_path(real, "chain-evil").c_str(), 0700), 0);
    plant(real, "chain-evil/secret.bin", "secret");
    plant(real, "chain/mid/root/keep.bin", "keep");
    std::uint64_t total = 0;
    EXPECT_EQ(sum_evidence_root(join_path(real, "chain/mid/root"), total), EvidenceError::kOk);
    EXPECT_EQ(total, 4u);
    EXPECT_EQ(slurp(join_path(real, "chain-evil/secret.bin")), "secret");

    const std::string links = make("/private/tmp/inferedge-link-XXXXXX");
    ASSERT_EQ(symlink(join_path(real, "chain/mid/root").c_str(), join_path(links, "final").c_str()),
              0);
    EXPECT_EQ(sum_evidence_root(join_path(links, "final"), total), EvidenceError::kSymlink);
    ASSERT_EQ(symlink(join_path(real, "chain/mid").c_str(), join_path(links, "parent").c_str()), 0);
    EXPECT_EQ(sum_evidence_root(join_path(links, "parent/root"), total), EvidenceError::kSymlink);
    ASSERT_EQ(symlink(join_path(real, "chain").c_str(), join_path(links, "middle").c_str()), 0);
    EXPECT_EQ(sum_evidence_root(join_path(links, "middle/mid/root"), total),
              EvidenceError::kSymlink);
    ASSERT_EQ(symlink(real.c_str(), join_path(links, "first").c_str()), 0);
    EXPECT_EQ(sum_evidence_root(join_path(links, "first/chain/mid/root"), total),
              EvidenceError::kSymlink);
    EXPECT_EQ(slurp(join_path(real, "chain/mid/root/keep.bin")), "keep");
    EXPECT_EQ(slurp(join_path(real, "chain-evil/secret.bin")), "secret");
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
    ASSERT_TRUE(evidence.ok() && admission.ok());
    ManualClock clock;
    EvidenceLock lock = hold_lock(evidence.path(), clock);
    plant(evidence.path(), "provenance/item-a/keep.bin", "keep");
    inferedge::CapabilityResult failed = inferedge::AuthenticatedCapability::authenticate(
        "quarantine-deletion", "bytes", "deletion/deletion-a.record", "qrun-a", evidence.path(),
        admission.path(), "trust-a", "2026-03-31", "key-a", "2020-01-01", "2100-12-31");
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
        "stale-lock", "other", "lock/stale-a.record", "lock/evidence.lock", evidence.path(),
        admission.path(), "trust-a", "2026-01-03", "key-a", "2020-01-01", "2100-12-31");
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
    ASSERT_TRUE(evidence.ok() && admission.ok());
    auto unavailable = inferedge::AuthenticatedCapability::authenticate(
        "quarantine-deletion", "bytes", "deletion/deletion-a.record", "qrun-a", evidence.path(),
        admission.path(), "trust-a", "2026-03-31", "key-a", "2020-01-01", "2100-12-31");
    EXPECT_EQ(unavailable.status(), EvidenceError::kUnavailable);
    EXPECT_FALSE(unavailable.has_capability());
    const EvidenceError ignored = unavailable.status();
    EXPECT_EQ(ignored, EvidenceError::kUnavailable);
    EXPECT_FALSE(unavailable.has_capability());
    {
        auto destroyed = inferedge::AuthenticatedCapability::authenticate(
            "stale-lock", "bytes", "lock/stale-a.record", "lock/evidence.lock", evidence.path(),
            admission.path(), "trust-a", "2026-01-03", "key-a", "2020-01-01", "2100-12-31");
        EXPECT_FALSE(destroyed.has_capability());
    }
    inferedge::AuthenticatedCapability owned;
    {
        auto extracted_from = inferedge::AuthenticatedCapability::authenticate(
            "reservation-deletion", "bytes", "deletion/deletion-a.record", "batch-a",
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
        "quarantine-deletion", "later", "deletion/deletion-b.record", "qrun-b", evidence.path(),
        admission.path(), "trust-b", "2026-04-01", "key-b", "2020-01-01", "2100-12-31");
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
    ASSERT_TRUE(evidence.ok() && admission.ok());
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
        "quarantine-deletion", bytes, "deletion/deletion-a.record", "qrun-a", evidence.path(),
        admission.path(), "trust-a", "2026-03-31", "key-a", "2020-01-01", "2100-12-31");
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
