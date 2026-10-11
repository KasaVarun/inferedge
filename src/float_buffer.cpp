#include "inferedge/float_buffer.hpp"

#include <utility>
#include <vector>

namespace inferedge {

FloatBuffer::FloatBuffer(std::vector<float>&& values) noexcept : values_{std::move(values)} {
}

FloatBuffer FloatBuffer::clone() const {
    std::vector<float> copied{values_};
    return FloatBuffer{std::move(copied)};
}

std::size_t FloatBuffer::size() const noexcept {
    return values_.size();
}

bool FloatBuffer::empty() const noexcept {
    return values_.empty();
}

std::span<const float> FloatBuffer::view() const noexcept {
    return std::span<const float>{values_.data(), values_.size()};
}

BufferRead FloatBuffer::try_at(std::size_t index) const noexcept {
    if (index >= values_.size()) {
        return BufferRead{BufferReadError::kOutOfRange, 0.0f};
    }
    return BufferRead{BufferReadError::kNone, values_[index]};
}

} // namespace inferedge
