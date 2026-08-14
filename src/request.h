#pragma once

#include <future>
#include <vector>

namespace inference_server {

// A single flattened input/output tensor. Using a plain vector<float> keeps the
// data model dead simple; a real engine would wrap shape/dtype metadata around
// this, but Phase 1 doesn't need it.
using Tensor = std::vector<float>;

// A batch of input tensors fed to the model in a single run() call. In Phase 1
// every batch holds exactly ONE element. Phase 2's dynamic batcher is what will
// actually fill these with many requests at once -- and that's the whole point
// of the project, because a model runs a batch of 32 almost as fast as a batch
// of 1.
using BatchedInput = std::vector<Tensor>;

// A batch of output tensors: exactly one prediction per input, same order.
using Output = std::vector<Tensor>;

// The unit of work that flows through the producer-consumer pipeline:
//
//   producer (gRPC handler thread) --> RequestQueue --> consumer (worker thread)
//
// THE PROMISE/FUTURE FLOW (how a prediction gets back to the right client):
//
//   A prediction is computed on a DIFFERENT thread than the one handling the
//   client's RPC, so we need a way to route the result back to the exact caller
//   that is waiting for it. A std::promise / std::future pair is that channel:
//
//     1. The gRPC handler creates this Request and, before letting go of it,
//        calls result_promise.get_future() to obtain its own private future.
//     2. It pushes the Request (promise and all) onto the queue and then blocks
//        on future.get().
//     3. The worker thread pops the Request, runs the model, and calls
//        result_promise.set_value(prediction).
//     4. That set_value() unblocks the exact future from step 1 -- delivering
//        this request's prediction to this request's handler. No request IDs,
//        no lookup tables: the promise IS the private mailbox between the two
//        threads.
//
// std::promise is move-only, which forces Request to be move-only too. That is
// deliberate: it guarantees we MOVE Requests through the pipeline (cheap) and
// never accidentally copy a large input tensor.
struct Request {
  Tensor input;                        // the flattened input tensor
  std::promise<Tensor> result_promise; // worker fulfills this with the prediction

  Request() = default;

  // Move-only (a promise cannot be copied).
  Request(Request&&) noexcept = default;
  Request& operator=(Request&&) noexcept = default;
  Request(const Request&) = delete;
  Request& operator=(const Request&) = delete;
};

}  // namespace inference_server
