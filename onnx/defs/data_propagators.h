#include "onnx\defs\shape_inference.h"

namespace ONNX_NAMESPACE {


// squeeze
// unsqueeze
// cast
inline void PropagateShapeDataFromInputToOutput(InferenceContext& ctx, int idx) {
  // propogate input data
  const auto input_data = ctx.getGeneratedShapeData(idx);
  if (input_data != nullptr) {
    TensorShapeProto tp;
    tp.CopyFrom(*input_data);
    ctx.addGeneratedShapeData(0, std::move(tp));
  }
}

template <typename T>
inline void ConcatenateData(InferenceContext& ctx, int32_t type = 7) {
  auto axisAttr = ctx.getAttribute("axis");
  if (!axisAttr) {
    fail_shape_inference("Required attribute axis is missing");
  }
  int axis = static_cast<int>(axisAttr->i());
  if (axis < 0) {
    return; // TODO: check if negative axis must be supported
  }

  auto output_type = ctx.getOutputType(0);
  if (output_type != nullptr && output_type->has_tensor_type() && output_type->tensor_type().has_shape()) {
    int64_t output_data_size = 0;
    for (size_t i = 0; i < output_type->tensor_type().shape().dim_size(); i++) {
      if (!output_type->tensor_type().shape().dim(i).has_dim_value()) {
        return;
      }
      output_data_size += output_type->tensor_type().shape().dim(i).dim_value();
    }
    TensorShapeProto tp;
    for (size_t i = 0; i < ctx.getNumInputs(); i++) {
      auto input_data = ctx.getInputData(i);
      if (input_data != nullptr) {
        const auto& input_vals = ParseData<T>(input_data);
        for (int j = 0; j < input_vals.size(); j++) {
          tp.mutable_dim()->Add()->set_dim_value(input_vals[j]);
        }
      } else {
        auto input_shape_proto = ctx.getGeneratedShapeData(i);
        if (input_shape_proto == nullptr) {
          return;
        }
        for (int j = 0; j < input_shape_proto->dim_size(); j++) {
          tp.mutable_dim()->Add()->CopyFrom(input_shape_proto->dim(j));
        }
      }
    }

    if (tp.dim_size() > 0) {
      ctx.addGeneratedShapeData(0, std::move(tp));
    }
  } else if (output_type != nullptr && output_type->has_tensor_type() && !output_type->tensor_type().has_shape()) {
    std::cout << "input type does not have shape: " << ctx.getOutputName(0);
  }
}

inline void GatherDataPropagator(InferenceContext& ctx) {
  auto input_data = ctx.getGeneratedShapeData(0);
  auto indices_data = ctx.getInputData(1);

  if (input_data == nullptr || indices_data == nullptr || !hasInputShape(ctx, 1)) {
    return;
  }

  const auto& indices_type = ctx.getInputType(1)->tensor_type();
  int axis = static_cast<int>(getAttribute(ctx, "axis", 0));

  if (axis == 0) {
    // currently only supporting int64 as it is generally used in shape computations
    if (indices_type.elem_type() != TensorProto_DataType_INT64) {
      return;
    }

    const auto& indices = ParseData<int64_t>(indices_data);
    auto input_size = input_data->dim_size();
    TensorShapeProto tp;
    if (indices.size() == 1 && input_size > indices[0]) {
      tp.mutable_dim()->Add()->CopyFrom(input_data->dim(indices[0]));
    }
    else {
      for (size_t i = 0; i < indices.size() && input_size > (int)indices[i]; i++) {
        tp.mutable_dim()->Add()->CopyFrom(input_data->dim(indices[i]));
      }
    }
    if (tp.dim_size() > 0){
      ctx.addGeneratedShapeData(0, std::move(tp));
    }
  }
}

inline void ShapeDataPropagator(InferenceContext& ctx) {
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  if (ctx.getInputType(0)->tensor_type().has_shape()) {
    auto input_shape = ctx.getInputType(0)->tensor_type().shape();
    auto dim_size = input_shape.dim_size();

    TensorShapeProto tsp;
    tsp.CopyFrom(input_shape);
    ctx.addGeneratedShapeData(0, std::move(tsp));
  }
}


inline void SliceDataPropagator(InferenceContext& ctx) {
  const TensorProto* startsInitializer = ctx.getInputData(1);
  const TensorProto* endsInitializer = ctx.getInputData(2);
  const TensorProto* axesInitializer = hasInputShape(ctx, 3) ? ctx.getInputData(3) : nullptr;
  const TensorProto* stepsInitializer = hasInputShape(ctx, 4) ? ctx.getInputData(4) : nullptr;

  if (!startsInitializer || !endsInitializer || (hasInputShape(ctx, 3) && !ctx.getInputData(3)) ||
      (hasInputShape(ctx, 4) && !ctx.getInputData(4))) {
    return;
  }

  auto get_initializer_data = [](const TensorProto* initializer) -> std::vector<int64_t> {
    std::vector<int64_t> vec;
    if (initializer->data_type() == TensorProto::INT64) {
      const auto& data = ParseData<int64_t>(initializer);
      vec.insert(vec.end(), data.begin(), data.end());
    } else if (initializer->data_type() == TensorProto::INT32) {
      const auto& data = ParseData<int32_t>(initializer);
      vec.insert(vec.end(), data.begin(), data.end());
    } else {
      // unaccepted data type
      fail_shape_inference("Only supports `int32_t` or `int64_t` inputs for starts/ends/axes/steps");
    }
    return vec;
  };

  std::vector<int64_t> starts = get_initializer_data(startsInitializer);
  std::vector<int64_t> ends = get_initializer_data(endsInitializer);

  if (starts.size() != ends.size()) {
    fail_shape_inference("Incorrect or missing input value for starts and ends");
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_rank = input_shape.dim_size();
  std::vector<int64_t> axes(starts.size());
  if (!axesInitializer) {
    std::iota(axes.begin(), axes.end(), 0);
  } else {
    axes = get_initializer_data(axesInitializer);
    if (axes.size() != starts.size()) {
      fail_shape_inference("Input axes has incorrect length");
    }
  }
  if ((axes.size() == 1 && axes[0] == 0) && (starts.size() == 1) && (ends.size() == 1)) {
    auto input_proto = ctx.getGeneratedShapeData(0);
    if (input_proto == nullptr) {
      return;
    }
    auto dim_size = input_proto->dim_size();
    TensorShapeProto tsp;
    for (auto i = starts[0]; i < ends[0] && i < dim_size; i++) {
      tsp.mutable_dim()->Add()->CopyFrom(input_proto->dim(i));
    }
    ctx.addGeneratedShapeData(0, std::move(tsp));
  }
}
} // namespace ONNX_NAMESPACE
