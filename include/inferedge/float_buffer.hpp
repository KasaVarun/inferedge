#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace inferedge {

enum class BufferReadError {
    kNone,
    kOutOfRange,
};

struct BufferRead {
    BufferReadError error{BufferReadError::kNone};
    float value{0.0f};
};

// Owns one contiguous float buffer. clone() is the only way to duplicate it.
// view() borrows that storage and is valid until this object is moved or
// destroyed. This is not a tensor: it has no shape, dtype, or device.
class FloatBuffer {
public:
    explicit FloatBuffer(std::vector<float>&& values) noexcept;

    FloatBuffer(const FloatBuffer&) = delete;
    FloatBuffer& operator=(const FloatBuffer&) = delete;
    FloatBuffer(FloatBuffer&&) noexcept = default;
    FloatBuffer& operator=(FloatBuffer&&) noexcept = default;
    ~FloatBuffer() = default;

    FloatBuffer clone() const;

    std::size_t size() const noexcept;
    bool empty() const noexcept;
    std::span<const float> view() const noexcept;
    BufferRead try_at(std::size_t index) const noexcept;

private:
    std::vector<float> values_;
};

} // namespace inferedge
