#include "engine.h"

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#ifdef WITH_ONNX
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#endif

namespace inference_server {

DummyEngine::DummyEngine(std::size_t output_dim) : output_dim_(output_dim) {}

Output DummyEngine::run(const BatchedInput& batch) {
  // Simulate the model's compute time. A real GPU forward pass for ResNet-50 is
  // in this ballpark -- and, crucially, it costs ~the same whether the batch
  // holds 1 input or 32. THAT is why batching wins: Phase 2 will amortize one
  // ~20ms run() over many requests instead of paying it per request.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  // Produce one fake output tensor per input, preserving order so the caller
  // can match predictions back to inputs by index.
  Output output;
  output.reserve(batch.size());
  for (const Tensor& input : batch) {
    Tensor prediction(output_dim_, 0.0f);
    // A cheap deterministic "prediction" so outputs aren't all identical: fold
    // in the input size. A real engine would run the actual forward pass here.
    if (!prediction.empty()) {
      prediction[0] = static_cast<float>(input.size());
    }
    output.push_back(std::move(prediction));
  }
  return output;
}

// ============================ OnnxEngine =================================
//
// ResNet-50 input geometry. Each request carries one image already flattened to
// C*H*W = 3*224*224 = 150528 floats in CHW (channel-major) order.
namespace {
constexpr int64_t kChannels = 3;
constexpr int64_t kHeight = 224;
constexpr int64_t kWidth = 224;
constexpr std::size_t kImageFloats =
    static_cast<std::size_t>(kChannels * kHeight * kWidth);  // 150528
}  // namespace

#ifdef WITH_ONNX

namespace {
// ONNX Runtime log verbosity, from the optional env var ORT_LOG_SEVERITY
// (verbose|info|warning|error). Defaults to WARNING. Set ORT_LOG_SEVERITY=verbose
// to make ORT print per-node execution-provider placement -- that is the log
// that definitively proves whether nodes ran on CUDA or fell back to CPU.
OrtLoggingLevel ortLogLevelFromEnv() {
  const char* v = std::getenv("ORT_LOG_SEVERITY");
  if (v) {
    const std::string s(v);
    if (s == "verbose") return ORT_LOGGING_LEVEL_VERBOSE;
    if (s == "info") return ORT_LOGGING_LEVEL_INFO;
    if (s == "error") return ORT_LOGGING_LEVEL_ERROR;
  }
  return ORT_LOGGING_LEVEL_WARNING;
}
}  // namespace

// The hidden implementation. Owns every ONNX Runtime object; when the Impl (and
// thus the OnnxEngine) is destroyed, the Ort::Session and Ort::Env are released
// in RAII order automatically.
struct OnnxEngine::Impl {
  Ort::Env env;                        // one per process; owns ORT's logger/threads
  Ort::SessionOptions session_options; // config applied before the session loads
  Ort::Session session{nullptr};       // the loaded model; created in the ctor body
  std::string provider;                // "cpu" | "cuda"
  std::string input_name;              // discovered from the model graph
  std::string output_name;
  std::size_t num_classes = 1000;      // last output dim (ResNet-50 => 1000)

  Impl(const std::string& model_path, const std::string& provider_in,
       int intra_op_threads)
      : env(ortLogLevelFromEnv(), "inference-server"), provider(provider_in) {
    // Clear, early error if the file isn't there -- ORT's own error is cryptic.
    std::ifstream probe(model_path, std::ios::binary);
    if (!probe.good()) {
      throw std::runtime_error(
          "OnnxEngine: model file not found or unreadable: '" + model_path +
          "'. Export it first (see README: torch.onnx.export ... resnet50.onnx).");
    }

    if (intra_op_threads > 0) {
      session_options.SetIntraOpNumThreads(intra_op_threads);
    }
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Report what the LINKED ORT build can actually do, and what we asked for.
    const auto available = Ort::GetAvailableProviders();
    std::cout << "ONNX Runtime providers compiled into this build:";
    for (const auto& p : available) std::cout << " " << p;
    std::cout << "\nOnnxEngine: requested execution provider = " << provider
              << std::endl;

    if (provider == "cuda") {
      // --- Guard 1: this ORT build must actually contain the CUDA provider. ---
      // If you accidentally linked the CPU-only package, stop here LOUDLY rather
      // than silently running on CPU.
      if (std::find(available.begin(), available.end(),
                    "CUDAExecutionProvider") == available.end()) {
        throw std::runtime_error(
            "OnnxEngine: --provider=cuda requested, but the linked ONNX Runtime "
            "has NO CUDAExecutionProvider (you are linking the CPU package). "
            "Point CMake at onnxruntime-linux-x64-gpu-1.17.1 "
            "(-DONNXRUNTIME_ROOT=...) and rebuild.");
      }
      // --- Guard 2: register the CUDA EP. A missing/mismatched CUDA or cuDNN
      // makes this (or session creation below) throw -- we deliberately let it
      // fail LOUD, never a silent CPU fallback. ---
      OrtCUDAProviderOptions cuda_options{};
      cuda_options.device_id = 0;
      try {
        session_options.AppendExecutionProvider_CUDA(cuda_options);
      } catch (const Ort::Exception& e) {
        throw std::runtime_error(
            std::string("OnnxEngine: failed to register the CUDA execution "
                        "provider. This is almost always a CUDA/cuDNN/ONNX-"
                        "Runtime VERSION MISMATCH (expected CUDA 11.8 + cuDNN "
                        "8.9.x for ORT 1.17.1). ORT said: ") +
            e.what());
      }
    } else if (provider != "cpu") {
      throw std::runtime_error("OnnxEngine: unknown provider '" + provider +
                               "' (use cpu|cuda)");
    }

    // Load the model ONCE. This parses the graph and reads the weights; it is
    // the expensive step we deliberately pay at startup, never per request.
    // With CUDA appended, a CUDA/cuDNN init failure surfaces HERE as an
    // Ort::Exception -- caught and re-thrown loudly, still no silent fallback.
    try {
      session = Ort::Session(env, model_path.c_str(), session_options);
    } catch (const Ort::Exception& e) {
      if (provider == "cuda") {
        throw std::runtime_error(
            std::string("OnnxEngine: CUDA session creation failed -- the CUDA "
                        "provider could not initialize (likely CUDA/cuDNN "
                        "version mismatch, or no usable GPU). ORT said: ") +
            e.what());
      }
      throw;
    }
    std::cout << "OnnxEngine: model loaded; session created with requested "
                 "provider = " << provider << std::endl;

    // Discover the input/output tensor names -- session.Run() addresses tensors
    // by name. The exported ResNet-50 has exactly one input and one output.
    Ort::AllocatorWithDefaultOptions alloc;
    input_name = session.GetInputNameAllocated(0, alloc).get();
    output_name = session.GetOutputNameAllocated(0, alloc).get();

    // Read the output's last dimension (the class count) so run() can slice the
    // [N, num_classes] result. The batch dim is dynamic and reports as -1.
    auto out_shape = session.GetOutputTypeInfo(0)
                         .GetTensorTypeAndShapeInfo()
                         .GetShape();
    if (!out_shape.empty() && out_shape.back() > 0) {
      num_classes = static_cast<std::size_t>(out_shape.back());
    }
  }
};

OnnxEngine::OnnxEngine(const std::string& model_path, const std::string& provider,
                       int intra_op_threads)
    : impl_(std::make_unique<Impl>(model_path, provider, intra_op_threads)) {}

OnnxEngine::~OnnxEngine() = default;

Output OnnxEngine::run(const BatchedInput& batch) {
  const int64_t N = static_cast<int64_t>(batch.size());
  if (N == 0) return {};

  // ---- Assemble ONE contiguous [N, 3, 224, 224] input buffer ----
  // ONNX Runtime wants a single flat float buffer whose shape has the batch as
  // the leading dimension. We concatenate the N per-request images back-to-back
  // (row-major), so element layout is exactly [N][C][H][W].
  std::vector<float> input_buf;
  input_buf.reserve(static_cast<std::size_t>(N) * kImageFloats);
  for (const Tensor& img : batch) {
    if (img.size() != kImageFloats) {
      throw std::runtime_error(
          "OnnxEngine: expected input of " + std::to_string(kImageFloats) +
          " floats (3x224x224), got " + std::to_string(img.size()));
    }
    input_buf.insert(input_buf.end(), img.begin(), img.end());
    // NOTE: no ImageNet mean/std normalization is applied. Shape correctness is
    // what matters for a throughput benchmark; classification accuracy does
    // not. If you wanted real predictions, you'd normalize each image here.
  }

  const std::array<int64_t, 4> input_shape{N, kChannels, kHeight, kWidth};

  // CreateTensor WRAPS our buffer (no copy); input_buf must outlive the Run()
  // call below -- it does, since it's a local that lives to the end of run().
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      mem_info, input_buf.data(), input_buf.size(), input_shape.data(),
      input_shape.size());

  // ---- Run the model ONCE on the whole batch ----
  // Run() takes parallel C-string arrays of input/output names and the input
  // Ort::Values, and returns one Ort::Value per requested output. ResNet-50 has
  // a single output of shape [N, num_classes].
  const char* input_names[] = {impl_->input_name.c_str()};
  const char* output_names[] = {impl_->output_name.c_str()};
  std::vector<Ort::Value> outputs =
      impl_->session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor,
                         /*input_count=*/1, output_names, /*output_count=*/1);

  // ---- Slice the [N, num_classes] output row-by-row ----
  // outputs[0]'s data is a contiguous float buffer laid out as N x C row-major.
  // Row i is the prediction for input i (== batch[i]). Copying each row into its
  // own Tensor gives the Response Router exactly what it already expects: one
  // output tensor per request, in order.
  Ort::Value& out = outputs[0];
  auto out_shape = out.GetTensorTypeAndShapeInfo().GetShape();  // [N, C]
  const std::size_t C = (out_shape.size() >= 2)
                            ? static_cast<std::size_t>(out_shape.back())
                            : impl_->num_classes;
  const float* data = out.GetTensorData<float>();

  Output result;
  result.reserve(static_cast<std::size_t>(N));
  for (int64_t i = 0; i < N; ++i) {
    const float* row = data + static_cast<std::size_t>(i) * C;
    result.emplace_back(row, row + C);  // copy this row into its own Tensor
  }
  return result;
}

#else  // !WITH_ONNX -- built without ONNX Runtime; OnnxEngine is a stub.

struct OnnxEngine::Impl {};  // empty, so unique_ptr<Impl> stays destructible

OnnxEngine::OnnxEngine(const std::string& /*model_path*/,
                       const std::string& /*provider*/,
                       int /*intra_op_threads*/) {
  throw std::runtime_error(
      "OnnxEngine: this binary was built WITHOUT ONNX Runtime support. "
      "Reconfigure with -DWITH_ONNX=ON and point CMake at ONNX Runtime "
      "(-DONNXRUNTIME_ROOT=<dir>).");
}
OnnxEngine::~OnnxEngine() = default;
Output OnnxEngine::run(const BatchedInput&) {
  throw std::runtime_error("OnnxEngine: not built with ONNX Runtime support.");
}

#endif  // WITH_ONNX

}  // namespace inference_server
