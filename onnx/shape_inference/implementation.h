/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
#include "onnx/proto_utils.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {
namespace shape_inference {

struct GraphInferenceContext {
  GraphInferenceContext(
      const std::unordered_map<std::string, TypeProto*>&
       outer_scope_value_types_by_name_in,
      const std::unordered_map<std::string, int> opset_imports_in,
      const ISchemaRegistry* schema_registry_in = OpSchemaRegistry::Instance())
      : outer_scope_value_types_by_name{&outer_scope_value_types_by_name_in},
        opset_imports{opset_imports_in},
        schema_registry{schema_registry_in} {}


  const std::unordered_map<std::string, TypeProto*>*
   outer_scope_value_types_by_name;
  const std::unordered_map<std::string, int> opset_imports;
  const ISchemaRegistry* schema_registry;

};

class GraphInferencerImpl : public GraphInferencer {
 public:
  GraphInferencerImpl(GraphProto& g, const GraphInferenceContext& context)
   : g_{&g}, context_{&context} {}

  std::vector<const TypeProto*> doInferencing(
      const std::vector<const TypeProto*>& inputTypes,
      const std::vector<const TensorProto*>& inputData) override;

 private:
  GraphProto* g_;
  const GraphInferenceContext* context_;
};

class SymbolManagerImpl : public SymbolManager {
 public:
  std::string createNewSymbol() override;

  bool addSymbol(const std::string& symbol) override;

 private:
  std::string getNewSymbol() {
    auto randchar = []() -> char {
      const char charset[] = "0123456789";
      const size_t max_index = (sizeof(charset) - 1);
      return charset[rand() % max_index];
    };
    std::string str(3, 0);
    std::generate_n(str.begin(), 3, randchar);
    return "unk__" + str;
  }
  std::vector<std::string> symbols;
  int symbolCreationRetries = 3;
};

struct InferenceContextImpl : public InferenceContext {
  InferenceContextImpl(
      NodeProto& n,
      const std::unordered_map<std::string, TypeProto*>& valueTypesByName,
      const std::unordered_map<std::string, const TensorProto*>&
      inputDataByName,
      const GraphInferenceContext* graphInferenceContext = nullptr)
      : graphInferenceContext_{graphInferenceContext} {
    for (auto& attr : *n.mutable_attribute()) {
      attributesByName_[attr.name()] = &attr;
      if (attr.has_g()) {
        // need a mutable GraphProto to run inferencing on this attribute
        graphProtoAttributesByName_[attr.name()] = attr.mutable_g();
      }
    }

    for (const auto& input : n.input()) {
      auto valueTypesIter = valueTypesByName.find(input);
      if (valueTypesIter != valueTypesByName.end()) {
        allInputTypes_.push_back(valueTypesIter->second);
      } else {
        allInputTypes_.push_back(nullptr);
      }

      const auto inputDataIter = inputDataByName.find(input);
      if (inputDataIter != inputDataByName.cend()) {
        allInputData_.push_back(inputDataIter->second);
      } else {
        allInputData_.push_back(nullptr);
      }
    }

    allOutputTypes_.resize(n.output_size());
  }

  const AttributeProto* getAttribute(const std::string& name) const override {
    auto iter = attributesByName_.find(name);
    if (iter == attributesByName_.end()) {
      return nullptr;
    } else {
      return iter->second;
    }
  }

  size_t getNumInputs() const override {
    return allInputTypes_.size();
  }

  const TypeProto* getInputType(size_t index) const override {
    if (index >= allInputTypes_.size()) {
      ONNX_THROW("input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds");
    }
    return allInputTypes_[index];
  }

  const TensorProto* getInputData(size_t index) const override {
    if (index >= allInputData_.size()) {
      ONNX_THROW("input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds");
    }
    return allInputData_[index];
  }

  size_t getNumOutputs() const override {
    return allOutputTypes_.size();
  }

  TypeProto* getOutputType(size_t index) override {
    if (index >= allOutputTypes_.size()) {
      ONNX_THROW("output " + ONNX_NAMESPACE::to_string(index) + " is out of bounds");
    }
    return &allOutputTypes_[index];
  }

  std::string getOutputName(size_t) const override {
    return "";
  }

  GraphInferencer* getGraphAttributeInferencer(
    const std::string& attr_name) override {
    if (!graphInferenceContext_) {
      fail_type_inference(
        "GraphProto attribute inferencing is not enabled in this InferenceContextImpl instance.");
    }

    GraphInferencer* inferencer = nullptr;

    auto entry = graphAttributeInferencers_.find(attr_name);
    if (entry == graphAttributeInferencers_.cend()) {
      // create GraphInferencer instance
      auto attrNameToGraphProto = graphProtoAttributesByName_.find(attr_name);
      if (attrNameToGraphProto == graphProtoAttributesByName_.cend()) {
        fail_type_inference(
          "Attribute ", attr_name, " does not contain a graph.");
      }

      std::unique_ptr<GraphInferencer> new_inferencer{new GraphInferencerImpl(
        *attrNameToGraphProto->second, *graphInferenceContext_)};

      inferencer = new_inferencer.get();
      graphAttributeInferencers_.emplace(attr_name, std::move(new_inferencer));
    } else {
      inferencer = entry->second.get();
    }

    return inferencer;
  }

  void addOutputData(size_t, TensorProto&&) override {}
  void addGeneratedShapeData(size_t, TensorShapeProto&&) override {}
  const TensorShapeProto* getGeneratedShapeData(size_t) const override {
    return nullptr;
  }

  std::vector<const TensorProto*> allInputData_;
  std::unordered_map<std::string, const AttributeProto*> attributesByName_;
  std::unordered_map<std::string, GraphProto*> graphProtoAttributesByName_;
  std::vector<const TypeProto*> allInputTypes_;
  std::vector<TypeProto> allOutputTypes_;
  const GraphInferenceContext* graphInferenceContext_;

  // mutable as internal cache of GraphInferencer instances
  mutable std::unordered_map<std::string, std::unique_ptr<GraphInferencer>>
   graphAttributeInferencers_;
};

struct SymbolicShapeInferenceContextImpl : public InferenceContext {
  SymbolicShapeInferenceContextImpl(
      NodeProto& n,
      const std::unordered_map<std::string, TypeProto*>& valueTypesByName,
      const std::unordered_map<std::string, const TensorProto*>& inputDataByName,
      std::unordered_map<std::string, TensorShapeProto>& generatedShapeData,
      const GraphInferenceContext* graphInferenceContext = nullptr)
      //: generatedIntermidiateData_{generatedIntermidiateData},
      : generatedShapeData_{generatedShapeData}, graphInferenceContext_{graphInferenceContext} {

    for (auto& attr : *n.mutable_attribute()) {
      attributesByName_[attr.name()] = &attr;
      if (attr.has_g()) {
        // need a mutable GraphProto to run inferencing on this attribute
        graphProtoAttributesByName_[attr.name()] = attr.mutable_g();
      }
    }

    size_t input_idx = 0;
    for (const auto& input : n.input()) {
      inputIndexToNameMap_.insert({input_idx++, input});

      auto valueTypesIter = valueTypesByName.find(input);
      if (valueTypesIter != valueTypesByName.end()) {
        allInputTypes_.push_back(valueTypesIter->second);
      } else {
        allInputTypes_.push_back(nullptr);
      }

      const auto inputDataIter = inputDataByName.find(input);
      if (inputDataIter != inputDataByName.cend()) {
        allInputData_.push_back(inputDataIter->second);
      } else {
        allInputData_.push_back(nullptr);
      }
    }

    size_t output_idx = 0;
    for (const auto& output : n.output()) {
      outputIndexToNameMap_.insert({output_idx++, output});
    }

    allOutputTypes_.resize(n.output_size());
  }

  void addOutputData(size_t index, TensorProto&& tp) override {
    std::cout << "SHOULD NOT BE HERE";
    /*if (index >= outputIndexToNameMap_.size()) {
      throw std::runtime_error("input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds");
    }
    auto result = generatedIntermidiateData_.insert({outputIndexToNameMap_.at(index), std::move(tp)});
    if (!result.second) {
      fail_shape_inference("data for input  " + ONNX_NAMESPACE::to_string(index) + " already exists.");
    }*/
  }

  const TensorProto* getInputData(size_t index) const override {
    if (index >= allInputData_.size()) {
      throw std::runtime_error("input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds");
    }

    if (allInputData_[index] != nullptr) {
      return allInputData_[index];
    } else {
      auto iter = generatedIntermidiateData_.find(inputIndexToNameMap_.at(index));
      if (iter != generatedIntermidiateData_.end()) {
        return &iter->second;
      }
    }
    return nullptr;
  }

  void addGeneratedShapeData(size_t index, TensorShapeProto&& tp) override {
    if (index >= outputIndexToNameMap_.size()) {
      throw std::runtime_error("input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds");
    }
    auto result = generatedShapeData_.insert({outputIndexToNameMap_.at(index), std::move(tp)});
    if (!result.second) {
      fail_shape_inference("data for input  " + ONNX_NAMESPACE::to_string(index) + " already exists.");
    }
  }

  const TensorShapeProto* getGeneratedShapeData(size_t index) const override {
    if (index >= allInputData_.size()) {
      throw std::runtime_error("input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds");
    }

    auto iter = generatedShapeData_.find(inputIndexToNameMap_.at(index));
    if (iter != generatedShapeData_.end()) {
        return &iter->second;
    }
    
    return nullptr;
  }

  const AttributeProto* getAttribute(const std::string& name) const override {
    auto iter = attributesByName_.find(name);
    if (iter == attributesByName_.end()) {
      return nullptr;
    } else {
      return iter->second;
    }
  }

  size_t getNumInputs() const override {
    return allInputTypes_.size();
  }

  const TypeProto* getInputType(size_t index) const override {
    if (index >= allInputTypes_.size()) {
      ONNX_THROW("input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds");
    }
    return allInputTypes_[index];
  }

  size_t getNumOutputs() const override {
    return allOutputTypes_.size();
  }

  TypeProto* getOutputType(size_t index) override {
    if (index >= allOutputTypes_.size()) {
      ONNX_THROW("output " + ONNX_NAMESPACE::to_string(index) + " is out of bounds");
    }
    return &allOutputTypes_[index];
  }

  std::string getOutputName(size_t index) const override {
    if (index > outputIndexToNameMap_.size()) {
      throw InferenceError("input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds");
    }

    return outputIndexToNameMap_.at(index);
  }

  GraphInferencer* getGraphAttributeInferencer(const std::string& attr_name) override {
    if (!graphInferenceContext_) {
      fail_type_inference("GraphProto attribute inferencing is not enabled in this InferenceContextImpl instance.");
    }

    GraphInferencer* inferencer = nullptr;

    auto entry = graphAttributeInferencers_.find(attr_name);
    if (entry == graphAttributeInferencers_.cend()) {
      // create GraphInferencer instance
      auto attrNameToGraphProto = graphProtoAttributesByName_.find(attr_name);
      if (attrNameToGraphProto == graphProtoAttributesByName_.cend()) {
        fail_type_inference("Attribute ", attr_name, " does not contain a graph.");
      }

      std::unique_ptr<GraphInferencer> new_inferencer{
          new GraphInferencerImpl(*attrNameToGraphProto->second, *graphInferenceContext_)};

      inferencer = new_inferencer.get();
      graphAttributeInferencers_.emplace(attr_name, std::move(new_inferencer));
    } else {
      inferencer = entry->second.get();
    }

    return inferencer;
  }

  std::unordered_map<std::string, TensorProto> generatedIntermidiateData_;
  std::unordered_map<std::string, TensorShapeProto>& generatedShapeData_;
  const GraphInferenceContext* graphInferenceContext_;

  std::unordered_map<size_t, std::string> outputIndexToNameMap_;
  std::unordered_map<size_t, std::string> inputIndexToNameMap_;
  std::vector<const TensorProto*> allInputData_;
  std::unordered_map<std::string, const AttributeProto*> attributesByName_;
  std::unordered_map<std::string, GraphProto*> graphProtoAttributesByName_;
  std::vector<const TypeProto*> allInputTypes_;
  std::vector<TypeProto> allOutputTypes_;
  // mutable as internal cache of GraphInferencer instances
  mutable std::unordered_map<std::string, std::unique_ptr<GraphInferencer>> graphAttributeInferencers_;
};
void checkShapesAndTypes(
  const TypeProto_Tensor& inferredType,
  const TypeProto_Tensor& existingType);

void checkShapesAndTypes(
  const TypeProto_Sequence& inferredType,
  const TypeProto_Sequence& existingType);

void checkShapesAndTypes(
  const TypeProto& inferredType,
  const TypeProto& existingType);

void mergeShapesAndTypes(
  const TypeProto_Tensor& inferredType, 
  TypeProto_Tensor* existingType);

void mergeShapesAndTypes(
  const TypeProto_Sequence& inferredType, 
  TypeProto_Tensor* existingType);

void mergeShapesAndTypes(
  const TypeProto& inferredType, 
  TypeProto* existingType);

void InferShapes(
    ModelProto& m,
    const bool check_type = false,
    const ISchemaRegistry* schema_registry = OpSchemaRegistry::Instance(),
    const int error_mode = 0
    );

void InferShapes(
    GraphProto* g,
    const std::unordered_map<std::string, int>& opset_imports,
    const bool check_type = false,
    const ISchemaRegistry* schema_registry = OpSchemaRegistry::Instance(),
    const int error_mode = 0
    );

void InferShapes(
    const std::string& model_path,
    const bool check_type = false,
    const std::string& save_path = "",
    const ISchemaRegistry* schema_registry = OpSchemaRegistry::Instance(),
    const int error_mode = 0
    );

void InferShapeForFunctionNode(
    const FunctionProto* func,
    const ISchemaRegistry* schema_registry,
    InferenceContext& ctx);

void InferShapeForFunctionNode(
    const FunctionProto* func,
    const std::unordered_map<std::string, int>& func_opset_imports,
    const ISchemaRegistry* schema_registry,
    InferenceContext& ctx);

std::string getErrorWithNodeInfo(NodeProto n, std::runtime_error err);

void deleteCreatedTypes(std::vector<TypeProto*> initializerTypeList);

} // namespace shape_inference
} // namespace ONNX_NAMESPACE
