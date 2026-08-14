#include "batcher.h"

#include <algorithm>
#include <utility>

namespace inference_server {

Batcher::Batcher(RequestQueue& queue, InferenceEngine& engine,
                 std::size_t max_batch_size, std::chrono::milliseconds max_wait)
    : queue_(queue),
      engine_(engine),
      // A batch size of 0 is meaningless; clamp to 1 so the loop always makes
      // progress. max_batch_size == 1 == the Phase 1 single-request baseline.
      max_batch_size_(max_batch_size == 0 ? 1 : max_batch_size),
      max_wait_(max_wait) {}

Batcher::~Batcher() { stop(); }

void Batcher::start() {
  // compare_exchange makes start() idempotent: only the first call (running_
  // still false) flips it to true and spawns the thread.
  bool expected = false;
  if (!running_.compare_exchange_strong(expected, true)) return;
  thread_ = std::thread(&Batcher::processLoop, this);
}

void Batcher::stop() {
  // exchange returns the PREVIOUS value; if it was already false we've already
  // stopped (or never started), so there's nothing to do. This makes stop()
  // safe to call from both an explicit shutdown and the destructor.
  if (!running_.exchange(false)) return;

  // Wake the batcher if it's parked waiting for work. After shutdown(), pop()
  // returns false ONLY once the queue is fully drained -- so processLoop()
  // finishes every already-queued request (fulfilling its promise) before it
  // sees the drained-and-shut-down signal and exits. Nothing is left hanging.
  queue_.shutdown();
  if (thread_.joinable()) thread_.join();
}

void Batcher::processLoop() {
  // ==================== THE SIZE-OR-TIMEOUT BATCHING LOOP ====================
  //
  // Each iteration of the outer loop assembles and runs exactly ONE batch.
  while (true) {
    // (1) Block until the FIRST request of a new batch arrives. pop() returns
    //     false only when the queue is shut down AND empty -> clean exit. This
    //     is also what guarantees a graceful, fully-drained shutdown.
    Request first;
    if (!queue_.pop(first)) break;

    // The batch's clock starts the instant its first request arrives, so a
    // request waits at most max_wait_ before being served even if no other
    // requests show up behind it.
    const auto deadline = std::chrono::steady_clock::now() + max_wait_;

    std::vector<Request> batch;
    batch.reserve(max_batch_size_);
    batch.push_back(std::move(first));  // MOVE in -- never copy the input tensor

    // (2) Greedily pull more requests until EITHER the batch is full OR the
    //     deadline passes -- whichever comes first:
    //       * pop_until() returns true  -> got another request, keep filling.
    //       * pop_until() returns false -> the deadline expired (timeout) or
    //         the queue was shut down. Either way, stop collecting and run what
    //         we have. THAT is the safety valve: a partial batch is served
    //         promptly instead of stalling for a batch that may never fill.
    while (batch.size() < max_batch_size_) {
      Request next;
      if (!queue_.pop_until(next, deadline)) break;
      batch.push_back(std::move(next));
    }

    // (3) Run the model once on the whole batch, then scatter results back.
    runBatchAndRoute(std::move(batch));
  }
}

void Batcher::runBatchAndRoute(std::vector<Request> batch) {
  // -------- BATCH ASSEMBLY: N individual inputs -> one [N, ...] batch --------
  // Collect the per-request input tensors into a single BatchedInput whose
  // leading dimension is the batch size. We MOVE each tensor out of its Request
  // so a large input is never copied. Order is preserved: inputs[i] came from
  // batch[i], which is what lets the router below line results back up by index.
  BatchedInput inputs;
  inputs.reserve(batch.size());
  for (Request& req : batch) {
    inputs.push_back(std::move(req.input));
  }

  // -------- EXECUTE ONCE ON THE WHOLE BATCH --------
  // This single call is the entire performance premise of the project: one
  // run() serves all N requests. With the DummyEngine's one-sleep-per-run()
  // model, a batch of 32 costs ~20 ms total instead of 32 x 20 ms.
  Output outputs = engine_.run(inputs);

  // -------- RESPONSE ROUTER: scatter row i back to request i --------
  // outputs[i] is the prediction for inputs[i], which came from batch[i], so a
  // single index lines everything up. Fulfilling batch[i]'s promise unblocks
  // exactly the std::future that the matching gRPC handler is parked on -> the
  // right client, and only that client, receives this row. This is the mirror
  // image of assembly: gather many in, scatter many out.
  const std::size_t n = std::min(batch.size(), outputs.size());
  for (std::size_t i = 0; i < n; ++i) {
    batch[i].result_promise.set_value(std::move(outputs[i]));
  }
  // Defensive: if the engine ever returned fewer rows than inputs (a broken
  // engine), fulfill the stragglers with an empty tensor rather than leaving
  // their clients blocked forever. The DummyEngine always returns batch.size()
  // rows, so this never triggers in Phase 2.
  for (std::size_t i = n; i < batch.size(); ++i) {
    batch[i].result_promise.set_value(Tensor{});
  }
}

}  // namespace inference_server
