#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "request.h"

namespace inference_server {

// Abstract interface standing in for the ML model. This is the seam that lets a
// real model be dropped in later: Phase 1 ships DummyEngine, and a future phase
// implements e.g. OnnxEngine (ResNet-50 via ONNX Runtime) behind this SAME
// interface. Nothing upstream (queue, worker, server) has to change.
class InferenceEngine {
 public:
  virtual ~InferenceEngine() = default;

  // Run a whole batch through the model in one shot and return one output
  // tensor per input, in the same order. Feeding many inputs into a single
  // run() call is the entire performance premise of the project -- exploited in
  // Phase 2 by the dynamic batcher.
  virtual Output run(const BatchedInput& batch) = 0;
};

// Fake model used for Phase 1: sleeps to imitate compute latency and returns
// correctly-sized placeholder predictions. Lets us build and load-test the
// serving infrastructure before a real model exists.
class DummyEngine : public InferenceEngine {
 public:
  // output_dim mimics a classifier's class count (ResNet-50 => 1000).
  explicit DummyEngine(std::size_t output_dim = 1000);

  Output run(const BatchedInput& batch) override;

 private:
  std::size_t output_dim_;
};

// Real model engine: runs a ResNet-50 .onnx through ONNX Runtime (CPU execution
// provider) behind the SAME InferenceEngine interface as DummyEngine. Because
// the batcher, queue, router, and server only ever see `InferenceEngine&`,
// swapping DummyEngine <-> OnnxEngine requires ZERO changes anywhere else --
// that is the whole point of the interface.
//
// PIMPL idiom: all ONNX Runtime types (Ort::Env, Ort::Session, ...) live in the
// hidden Impl defined in engine.cpp. So this header pulls in NO ONNX Runtime
// headers, and nothing that includes engine.h (server, batcher, main) needs Ort
// on its include path. The clean seam is literal, not just conceptual.
class OnnxEngine : public InferenceEngine {
 public:
  // Loads the model ONCE, here in the constructor -- never per request. Throws
  // std::runtime_error with a clear message if the file is missing or the model
  // cannot be loaded.
  //   provider          : "cpu" (default) or "cuda". With "cuda" the CUDA
  //                       execution provider is registered; if it cannot be
  //                       created (missing/mismatched CUDA or cuDNN, no GPU) the
  //                       constructor throws -- it does NOT silently fall back
  //                       to CPU.
  //   intra_op_threads  : ONNX Runtime's per-Run() CPU thread count (0 = auto).
  explicit OnnxEngine(const std::string& model_path,
                      const std::string& provider = "cpu",
                      int intra_op_threads = 0);

  // Out-of-line (defined in engine.cpp) because Impl is incomplete here -- that
  // is what lets a unique_ptr<Impl> destructor compile with a forward decl.
  ~OnnxEngine();

  OnnxEngine(const OnnxEngine&) = delete;
  OnnxEngine& operator=(const OnnxEngine&) = delete;

  // Shapes the N inputs into a single [N, 3, 224, 224] tensor, runs the model
  // once, and returns one output row per input (same order) so the existing
  // Response Router can scatter each row to its request's promise.
  Output run(const BatchedInput& batch) override;

 private:
  struct Impl;                    // defined in engine.cpp; owns the Ort::Session
  std::unique_ptr<Impl> impl_;    // RAII: session is released when this dies
};

}  // namespace inference_server
