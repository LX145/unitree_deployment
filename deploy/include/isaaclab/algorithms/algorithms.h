// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <algorithm>
#include <iostream>
#include <mutex>

namespace isaaclab
{

class Algorithms
{
public:
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

    /// Called at episode start. Override to reset internal state (e.g. GRU hidden).
    virtual void reset() {}

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }

    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path)
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        // printf("Loading ONNX model from path: %s\n", model_path.c_str());
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            auto input_name = session->GetInputNameAllocated(i, allocator);
            // printf("Input %zu : name=%s\n", i, input_name.get());
            input_names.push_back(input_name.release());
        }

        for (const auto& shape : input_shapes) {
            size_t size = 1;
            for (const auto& dim : shape) {
                size *= dim;
            }
            input_sizes.push_back(size);
        }

        // Get all output names and shapes
        size_t total_output_size = 0;
        for (size_t i = 0; i < session->GetOutputCount(); ++i) {
            auto output_name = session->GetOutputNameAllocated(i, allocator);
            output_names.push_back(output_name.release());
            Ort::TypeInfo output_type = session->GetOutputTypeInfo(i);
            auto shape = output_type.GetTensorTypeAndShapeInfo().GetShape();
            output_shapes.push_back(shape);
            size_t size = 1;
            for (const auto& dim : shape) {
                if (dim > 0) size *= dim;
            }
            if (i == 0) total_output_size = size;  // primary output
        }
        action.resize(total_output_size);
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        auto result = act_multi(obs);
        // Return the first (or only) output as the action vector
        return result.at(output_names[0]);
    }

    /// Multi-output inference. Returns a map from output name to data vector.
    std::unordered_map<std::string, std::vector<float>>
    act_multi(std::unordered_map<std::string, std::vector<float>> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // Create input tensors
        std::vector<Ort::Value> input_tensors;
        for (size_t i = 0; i < input_names.size(); ++i) {
            const std::string name_str(input_names[i]);
            if (obs.find(name_str) == obs.end()) {
                throw std::runtime_error("Input name " + name_str + " not found in observations.");
            }
            auto& input_data = obs.at(name_str);
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info, input_data.data(), input_sizes[i],
                input_shapes[i].data(), input_shapes[i].size()));
        }

        // Run the model — read ALL outputs
        auto output_tensors = session->Run(Ort::RunOptions{nullptr},
            input_names.data(), input_tensors.data(), input_tensors.size(),
            output_names.data(), output_names.size());

        // Collect all outputs into a map
        std::unordered_map<std::string, std::vector<float>> outputs;
        for (size_t i = 0; i < output_names.size(); ++i) {
            auto* data = output_tensors[i].GetTensorMutableData<float>();
            auto shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
            size_t count = 1;
            for (auto d : shape) if (d > 0) count *= d;
            outputs[output_names[i]] = std::vector<float>(data, data + count);
        }

        // Also store the primary action for get_action() compatibility
        {
            std::lock_guard<std::mutex> lock(act_mtx_);
            auto& primary = outputs.at(output_names[0]);
            action = primary;
        }
        return outputs;
    }

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int64_t> input_sizes;
    std::vector<std::vector<int64_t>> output_shapes;
};

// ===================================================================
// SplitDepthRunner — dual-ONNX inference for depth student policies
// ===================================================================
//
// Loads policy_depth.onnx (encoder) and policy_actor.onnx (actor).
// Encoder runs at ~10 Hz when new depth frames arrive; actor runs
// every policy tick at 50 Hz.
//
// GRU hidden_state [1,1,512] and depth_memory [1,512] are maintained
// internally and zeroed on reset().

class SplitDepthRunner : public Algorithms
{
public:
    SplitDepthRunner(const std::string& encoder_path,
                     const std::string& actor_path,
                     std::shared_ptr<Articulation> robot,
                     int encoder_interval = 5,
                     bool encode_on_new_depth = false)
        : robot_(std::move(robot))
        , depth_encoder_(std::make_unique<OrtRunner>(encoder_path))
        , actor_(std::make_unique<OrtRunner>(actor_path))
        , encoder_interval_(encoder_interval)
        , encode_on_new_depth_(encode_on_new_depth)
    {
        // Allocate GRU hidden state: [1, 1, 512]
        hidden_state_.resize(512, 0.0f);

        // Allocate depth_memory: [1, 512]
        depth_memory_.resize(512, 0.0f);

        std::cout << "[SplitDepthRunner] encoder=" << encoder_path
                  << " actor=" << actor_path << std::endl;
    }

    /// Main inference entry called at 50 Hz.
    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs_map) override
    {
        // Depth frames arrive asynchronously (DDS or camera thread). The GRU
        // encoder must run at the TRAINING cadence (every encoder_interval_
        // control steps = 100 ms), regardless of the depth stream's update_hz:
        // a faster stream (e.g. 50 Hz) must NOT make the encoder run 5x more
        // often than training, or the hidden-state dynamics diverge.
        std::vector<float> new_depth;
        bool run_encoder = step_count_ % encoder_interval_ == 0;
        if (encode_on_new_depth_) {
            // Frame-driven (use a fresh frame when available), but still
            // throttled to encoder_interval_ control steps.
            std::lock_guard<std::mutex> lock(robot_->data.depth_mtx);
            run_encoder = run_encoder &&
                          robot_->data.depth_valid &&
                          robot_->data.depth_seq != last_depth_seq_;
            if (run_encoder) {
                new_depth = robot_->data.depth_obs;
                last_depth_seq_ = robot_->data.depth_seq;
            }
        }

        if (run_encoder) {
            std::unordered_map<std::string, std::vector<float>> enc_in;
            enc_in["depth"] = encode_on_new_depth_ ? new_depth : obs_map.at("depth");
            enc_in["proprio"] = obs_map.at("proprio");
            enc_in["hidden_in"] = hidden_state_;

            auto enc_out = depth_encoder_->act_multi(enc_in);

            depth_memory_ = enc_out.at("depth_memory");
            hidden_state_ = enc_out.at("hidden_out");
        }

        // ---- always run actor ----
        std::unordered_map<std::string, std::vector<float>> act_in;
        act_in["proprio"] = obs_map.at("proprio");
        act_in["depth_memory"] = depth_memory_;

        auto action = actor_->act(act_in);
        step_count_++;
        return action;
    }

    /// Reset GRU hidden state and depth memory (called on episode start).
    void reset() override
    {
        std::fill(hidden_state_.begin(), hidden_state_.end(), 0.0f);
        std::fill(depth_memory_.begin(), depth_memory_.end(), 0.0f);
        step_count_ = 0;
        last_depth_seq_ = 0;
    }

private:
    std::shared_ptr<Articulation> robot_;
    std::unique_ptr<OrtRunner> depth_encoder_;
    std::unique_ptr<OrtRunner> actor_;

    std::vector<float> hidden_state_;   // [1, 1, 512] flattened
    std::vector<float> depth_memory_;   // [1, 512]
    int encoder_interval_ = 5;          // matches training depth_update_interval
    bool encode_on_new_depth_ = false;  // enabled for asynchronous MuJoCo DDS input
    uint64_t last_depth_seq_ = 0;
    int step_count_ = 0;
};
};