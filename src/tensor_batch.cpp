#include "inferedge/tensor_batch.hpp"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace inferedge {
namespace {

TensorBatchError Error(TensorBatchErrorKind kind, std::string message) {
    return TensorBatchError{kind, std::move(message)};
}

TensorBatchAssembly AssemblyFailure(TensorBatchErrorKind kind, std::string message) {
    return TensorBatchAssembly{Error(kind, std::move(message)), {}};
}

TensorBatchSplit SplitFailure(TensorBatchErrorKind kind, std::string message) {
    return TensorBatchSplit{Error(kind, std::move(message)), {}};
}

bool CheckedMultiply(std::size_t left, std::size_t right, std::size_t& product) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    product = left * right;
    return true;
}

bool FitsSize(std::int64_t value, std::size_t& converted) {
    if (value < 0) {
        return false;
    }
    const auto unsigned_value = static_cast<std::uint64_t>(value);
    if (unsigned_value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return false;
    }
    converted = static_cast<std::size_t>(unsigned_value);
    return true;
}

bool RequestCountFitsInt64(std::size_t request_count) {
    return request_count <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
}

TensorBatchError ValidateMetadata(std::span<const TensorMetadata> metadata, const char* role,
                                  std::vector<std::size_t>& row_elements) {
    row_elements.clear();
    if (metadata.empty()) {
        return Error(TensorBatchErrorKind::kUnsupportedMetadata,
                     std::string{role} + " metadata is empty");
    }

    row_elements.reserve(metadata.size());
    for (std::size_t index = 0; index < metadata.size(); ++index) {
        const TensorMetadata& tensor = metadata[index];
        if (tensor.element_type != ElementType::kFloat32) {
            return Error(TensorBatchErrorKind::kUnsupportedType,
                         std::string{role} + " metadata " + tensor.name + " is not float32");
        }
        if (tensor.name.empty()) {
            return Error(TensorBatchErrorKind::kEmptyName,
                         std::string{role} + " metadata name is empty");
        }
        for (std::size_t earlier = 0; earlier < index; ++earlier) {
            if (metadata[earlier].name == tensor.name) {
                return Error(TensorBatchErrorKind::kDuplicateName,
                             std::string{role} + " metadata name is duplicated: " + tensor.name);
            }
        }
        if (tensor.dimensions.size() < 2) {
            return Error(TensorBatchErrorKind::kUnsupportedMetadata,
                         std::string{role} + " metadata " + tensor.name + " rank is below 2");
        }
        if (tensor.dimensions[0] >= 0) {
            return Error(TensorBatchErrorKind::kUnsupportedMetadata,
                         std::string{role} + " metadata " + tensor.name + " axis 0 is not dynamic");
        }
        std::size_t row = 1;
        for (std::size_t axis = 1; axis < tensor.dimensions.size(); ++axis) {
            const std::int64_t dimension = tensor.dimensions[axis];
            if (dimension < 0) {
                return Error(TensorBatchErrorKind::kUnsupportedMetadata,
                             std::string{role} + " metadata " + tensor.name +
                                 " has an additional dynamic axis");
            }
            if (dimension == 0) {
                return Error(TensorBatchErrorKind::kUnsupportedMetadata,
                             std::string{role} + " metadata " + tensor.name +
                                 " has a non-positive dimension");
            }
            std::size_t converted = 0;
            if (!FitsSize(dimension, converted) || !CheckedMultiply(row, converted, row)) {
                return Error(TensorBatchErrorKind::kOverflow, std::string{role} + " metadata " +
                                                                  tensor.name +
                                                                  " shape product overflows");
            }
        }
        row_elements.push_back(row);
    }
    return {};
}

const NamedTensor* FindTensor(std::span<const NamedTensor> tensors, const std::string& name,
                              bool& duplicated) {
    const NamedTensor* found = nullptr;
    duplicated = false;
    for (const NamedTensor& tensor : tensors) {
        if (tensor.name != name) {
            continue;
        }
        if (found != nullptr) {
            duplicated = true;
            return found;
        }
        found = &tensor;
    }
    return found;
}

bool KnownName(std::span<const TensorMetadata> metadata, const std::string& name) {
    for (const TensorMetadata& tensor : metadata) {
        if (tensor.name == name) {
            return true;
        }
    }
    return false;
}

TensorBatchError ValidateSuppliedNames(std::span<const TensorMetadata> metadata,
                                       std::span<const NamedTensor> tensors,
                                       const std::string& where) {
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        const std::string& name = tensors[index].name;
        if (name.empty()) {
            return Error(TensorBatchErrorKind::kEmptyName, where + " tensor name is empty");
        }
        for (std::size_t earlier = 0; earlier < index; ++earlier) {
            if (tensors[earlier].name == name) {
                return Error(TensorBatchErrorKind::kDuplicateName,
                             where + " duplicates tensor " + name);
            }
        }
        if (!KnownName(metadata, name)) {
            return Error(TensorBatchErrorKind::kUnexpectedTensor,
                         where + " has unexpected tensor " + name);
        }
    }
    return {};
}

TensorBatchError ValidateSingleRequestShape(const TensorMetadata& expected,
                                            const NamedTensor& tensor, std::size_t row_elements,
                                            const char* role, std::size_t request_index) {
    const std::string where =
        std::string{role} + " request " + std::to_string(request_index) + " tensor " + tensor.name;
    if (tensor.shape.size() != expected.dimensions.size()) {
        return Error(TensorBatchErrorKind::kRankMismatch, where + " rank does not match metadata");
    }
    if (tensor.shape[0] != 1) {
        return Error(TensorBatchErrorKind::kBatchAxisMismatch, where + " batch axis is not 1");
    }
    for (std::size_t axis = 1; axis < expected.dimensions.size(); ++axis) {
        if (tensor.shape[axis] != expected.dimensions[axis]) {
            return Error(TensorBatchErrorKind::kShapeMismatch,
                         where + " static dimension does not match metadata");
        }
    }
    if (tensor.values.size() != row_elements) {
        return Error(TensorBatchErrorKind::kElementCountMismatch,
                     where + " element count does not match its shape");
    }
    return {};
}

TensorBatchError ValidateBatchedShape(const TensorMetadata& expected, const NamedTensor& tensor,
                                      std::size_t row_elements, std::size_t request_count) {
    const std::string where = "batched output " + tensor.name;
    if (tensor.shape.size() != expected.dimensions.size()) {
        return Error(TensorBatchErrorKind::kRankMismatch, where + " rank does not match metadata");
    }
    if (!RequestCountFitsInt64(request_count)) {
        return Error(TensorBatchErrorKind::kOverflow, where + " request count does not fit int64");
    }
    if (tensor.shape[0] != static_cast<std::int64_t>(request_count)) {
        return Error(TensorBatchErrorKind::kBatchAxisMismatch,
                     where + " batch axis does not match the request count");
    }
    for (std::size_t axis = 1; axis < expected.dimensions.size(); ++axis) {
        if (tensor.shape[axis] != expected.dimensions[axis]) {
            return Error(TensorBatchErrorKind::kShapeMismatch,
                         where + " static dimension does not match metadata");
        }
    }
    std::size_t product = 0;
    if (!CheckedMultiply(row_elements, request_count, product)) {
        return Error(TensorBatchErrorKind::kOverflow, where + " shape product overflows");
    }
    if (tensor.values.size() != product) {
        return Error(TensorBatchErrorKind::kElementCountMismatch,
                     where + " element count does not match its shape");
    }
    return {};
}

std::vector<std::int64_t> SingleRequestShape(const TensorMetadata& metadata) {
    std::vector<std::int64_t> shape;
    shape.reserve(metadata.dimensions.size());
    shape.push_back(1);
    shape.insert(shape.end(), metadata.dimensions.begin() + 1, metadata.dimensions.end());
    return shape;
}

std::vector<std::int64_t> BatchedShape(const TensorMetadata& metadata, std::size_t request_count) {
    std::vector<std::int64_t> shape = SingleRequestShape(metadata);
    shape[0] = static_cast<std::int64_t>(request_count);
    return shape;
}

} // namespace

TensorBatchAssembly AssembleBatchInputs(std::span<const TensorMetadata> model_inputs,
                                        std::span<const std::span<const NamedTensor>> requests) {
    std::vector<std::size_t> row_elements;
    const TensorBatchError metadata = ValidateMetadata(model_inputs, "input", row_elements);
    if (metadata.kind != TensorBatchErrorKind::kNone) {
        return TensorBatchAssembly{metadata, {}};
    }
    if (requests.empty()) {
        return AssemblyFailure(TensorBatchErrorKind::kEmptyBatch, "request batch is empty");
    }
    if (!RequestCountFitsInt64(requests.size())) {
        return AssemblyFailure(TensorBatchErrorKind::kOverflow, "request count does not fit int64");
    }

    std::vector<std::vector<const NamedTensor*>> matched(model_inputs.size());
    for (std::vector<const NamedTensor*>& column : matched) {
        column.reserve(requests.size());
    }

    for (std::size_t request_index = 0; request_index < requests.size(); ++request_index) {
        const std::span<const NamedTensor> tensors = requests[request_index];
        const TensorBatchError names = ValidateSuppliedNames(
            model_inputs, tensors, "input request " + std::to_string(request_index));
        if (names.kind != TensorBatchErrorKind::kNone) {
            return TensorBatchAssembly{names, {}};
        }
        for (std::size_t input_index = 0; input_index < model_inputs.size(); ++input_index) {
            bool duplicated = false;
            const NamedTensor* tensor =
                FindTensor(tensors, model_inputs[input_index].name, duplicated);
            if (duplicated) {
                return AssemblyFailure(TensorBatchErrorKind::kDuplicateName,
                                       "input request " + std::to_string(request_index) +
                                           " duplicates tensor " + model_inputs[input_index].name);
            }
            if (tensor == nullptr) {
                return AssemblyFailure(TensorBatchErrorKind::kMissingTensor,
                                       "input request " + std::to_string(request_index) +
                                           " is missing tensor " + model_inputs[input_index].name);
            }
            const TensorBatchError shape =
                ValidateSingleRequestShape(model_inputs[input_index], *tensor,
                                           row_elements[input_index], "input", request_index);
            if (shape.kind != TensorBatchErrorKind::kNone) {
                return TensorBatchAssembly{shape, {}};
            }
            matched[input_index].push_back(tensor);
        }
    }

    std::vector<NamedTensor> assembled;
    assembled.reserve(model_inputs.size());
    for (std::size_t input_index = 0; input_index < model_inputs.size(); ++input_index) {
        std::size_t total = 0;
        if (!CheckedMultiply(row_elements[input_index], requests.size(), total)) {
            return AssemblyFailure(TensorBatchErrorKind::kOverflow,
                                   "assembled input " + model_inputs[input_index].name +
                                       " element count overflows");
        }
        std::vector<float> values(total);
        std::size_t offset = 0;
        for (const NamedTensor* tensor : matched[input_index]) {
            for (std::size_t element = 0; element < tensor->values.size(); ++element) {
                values[offset + element] = tensor->values[element];
            }
            offset += tensor->values.size();
        }
        assembled.push_back(NamedTensor{model_inputs[input_index].name,
                                        BatchedShape(model_inputs[input_index], requests.size()),
                                        std::move(values)});
    }
    return TensorBatchAssembly{{}, std::move(assembled)};
}

TensorBatchSplit SplitBatchOutputs(std::span<const TensorMetadata> model_outputs,
                                   std::span<const NamedTensor> batched_outputs,
                                   std::size_t request_count) {
    std::vector<std::size_t> row_elements;
    const TensorBatchError metadata = ValidateMetadata(model_outputs, "output", row_elements);
    if (metadata.kind != TensorBatchErrorKind::kNone) {
        return TensorBatchSplit{metadata, {}};
    }
    if (request_count == 0) {
        return SplitFailure(TensorBatchErrorKind::kEmptyBatch, "request batch is empty");
    }
    if (!RequestCountFitsInt64(request_count)) {
        return SplitFailure(TensorBatchErrorKind::kOverflow, "request count does not fit int64");
    }

    const TensorBatchError names =
        ValidateSuppliedNames(model_outputs, batched_outputs, "batched outputs");
    if (names.kind != TensorBatchErrorKind::kNone) {
        return TensorBatchSplit{names, {}};
    }

    std::vector<const NamedTensor*> matched;
    matched.reserve(model_outputs.size());
    for (std::size_t output_index = 0; output_index < model_outputs.size(); ++output_index) {
        bool duplicated = false;
        const NamedTensor* tensor =
            FindTensor(batched_outputs, model_outputs[output_index].name, duplicated);
        if (duplicated) {
            return SplitFailure(TensorBatchErrorKind::kDuplicateName,
                                "batched outputs duplicate tensor " +
                                    model_outputs[output_index].name);
        }
        if (tensor == nullptr) {
            return SplitFailure(TensorBatchErrorKind::kMissingTensor,
                                "batched outputs are missing tensor " +
                                    model_outputs[output_index].name);
        }
        const TensorBatchError shape = ValidateBatchedShape(
            model_outputs[output_index], *tensor, row_elements[output_index], request_count);
        if (shape.kind != TensorBatchErrorKind::kNone) {
            return TensorBatchSplit{shape, {}};
        }
        matched.push_back(tensor);
    }

    std::vector<std::vector<NamedTensor>> split_requests(request_count);
    for (std::vector<NamedTensor>& request : split_requests) {
        request.reserve(model_outputs.size());
    }
    for (std::size_t output_index = 0; output_index < model_outputs.size(); ++output_index) {
        const NamedTensor& source = *matched[output_index];
        const std::size_t row = row_elements[output_index];
        const std::vector<std::int64_t> shape = SingleRequestShape(model_outputs[output_index]);
        for (std::size_t request_index = 0; request_index < request_count; ++request_index) {
            const std::size_t offset = request_index * row;
            std::vector<float> values(row);
            for (std::size_t element = 0; element < row; ++element) {
                values[element] = source.values[offset + element];
            }
            split_requests[request_index].push_back(
                NamedTensor{model_outputs[output_index].name, shape, std::move(values)});
        }
    }
    return TensorBatchSplit{{}, std::move(split_requests)};
}

} // namespace inferedge
