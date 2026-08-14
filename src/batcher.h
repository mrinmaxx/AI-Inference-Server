#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <thread>
#include <vector>

#include "engine.h"
#include "request.h"
#include "request_queue.h"

namespace inference_server {

// The dynamic batcher -- the heart of the server.
//
// It runs on ONE dedicated thread and is the SOLE consumer of the RequestQueue.
// gRPC handler threads (the producers) keep pushing individual requests; the
// batcher GROUPS several of them together and runs the model once per group.
// Because a model processes a batch of N almost as cheaply as a batch of 1,
// grouping is what turns the ~50 inf/s serialized baseline into a much higher
// number. (This class replaces Phase 1's batch-size-1 modelWorker; setting
// max_batch_size == 1 reproduces that old behavior exactly.)
//
// Collection is size-OR-timeout (see the long comment in batcher.cpp):
//   * accumulate requests until max_batch_size is reached, OR
//   * until max_wait has elapsed since the FIRST request of the batch,
// whichever comes first. The timeout is the safety valve so a lone request is
// never stuck waiting for a batch that will not fill.
class Batcher {
 public:
  Batcher(RequestQueue& queue, InferenceEngine& engine,
          std::size_t max_batch_size, std::chrono::milliseconds max_wait);

  // Owns a thread and holds references to shared state -> non-copyable.
  Batcher(const Batcher&) = delete;
  Batcher& operator=(const Batcher&) = delete;

  // Launch the batcher thread. Idempotent (a second call is a no-op).
  void start();

  // Signal shutdown and join the thread. Idempotent; also invoked by the
  // destructor. Completes the in-flight batch and drains everything already
  // queued -- every pending promise is fulfilled -- before returning, so no
  // client is left blocked on an unfulfilled future.
  void stop();

  ~Batcher();

  std::size_t max_batch_size() const { return max_batch_size_; }
  std::chrono::milliseconds max_wait() const { return max_wait_; }

 private:
  void processLoop();                                 // the thread body
  void runBatchAndRoute(std::vector<Request> batch);  // execute once + scatter

  RequestQueue& queue_;      // not owned; outlives the batcher
  InferenceEngine& engine_;  // not owned; outlives the batcher
  std::size_t max_batch_size_;
  std::chrono::milliseconds max_wait_;

  // Lifecycle flag (requirement: use a std::atomic running flag). start() sets
  // it true, stop() sets it false; the exchange/compare_exchange on it is what
  // makes start()/stop() idempotent.
  std::atomic<bool> running_{false};
  std::thread thread_;
};

}  // namespace inference_server
