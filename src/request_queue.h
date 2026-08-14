#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>

#include "request.h"

namespace inference_server {

// A thread-safe, blocking producer-consumer queue sitting between the network
// layer and the model-execution layer:
//
//   producers (gRPC handler threads) --push--> [ queue ] --pop--> consumer
//                                                                  (worker thread)
//
// The mutex serializes access to the underlying std::queue. The condition
// variable lets the consumer SLEEP while the queue is empty and be woken the
// instant a request arrives -- far better than busy-polling in a spin loop.
class RequestQueue {
 public:
  RequestQueue() = default;

  // Not copyable/movable: it owns synchronization primitives and is shared by
  // reference across threads.
  RequestQueue(const RequestQueue&) = delete;
  RequestQueue& operator=(const RequestQueue&) = delete;

  // PRODUCER side. Takes the Request by value and moves it in, so a large input
  // tensor is never copied. Wakes one waiting consumer.
  void push(Request request);

  // CONSUMER side. Blocks until a Request is available, moves it into `out`, and
  // returns true. Returns false only once the queue has been shut down AND fully
  // drained -- that's the signal for the worker to leave its loop.
  bool pop(Request& out);

  // CONSUMER side, with a DEADLINE. Like pop(), but also gives up (returns
  // false) if `deadline` passes before a request is available. This is the
  // timeout half of the batcher's size-OR-timeout loop: after grabbing the
  // first request of a batch, the batcher keeps calling pop_until() to gather
  // more, but only until the batch's deadline. Returns true and moves a request
  // into `out` if one arrives in time; false on timeout OR on shutdown-drained.
  bool pop_until(Request& out, std::chrono::steady_clock::time_point deadline);

  // Refuse further work and wake every waiter so blocked threads can exit.
  void shutdown();

 private:
  std::queue<Request> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool shutdown_ = false;
};

}  // namespace inference_server
