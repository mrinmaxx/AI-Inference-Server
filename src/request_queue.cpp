#include "request_queue.h"

#include <utility>

namespace inference_server {

void RequestQueue::push(Request request) {
  {
    // RAII lock: hold the mutex only for the tiny critical section that mutates
    // the queue, then release it (at the closing brace) before we notify.
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(std::move(request));  // move in -- no copy of the input tensor
  }
  // Notify OUTSIDE the lock: if we notified while still holding the mutex, the
  // woken consumer would immediately block trying to acquire it.
  cv_.notify_one();
}

bool RequestQueue::pop(Request& out) {
  std::unique_lock<std::mutex> lock(mutex_);

  // Sleep until there is work to do OR we are shutting down. The predicate form
  // of wait() re-checks the condition on every wake-up, which also correctly
  // handles spurious wake-ups (wait can return without a matching notify).
  cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });

  // Woken because of shutdown and there is nothing left -> tell the worker to
  // stop. (We still drain any queued requests before reporting empty.)
  if (queue_.empty()) {
    return false;
  }

  out = std::move(queue_.front());  // move the Request (and its promise) out
  queue_.pop();
  return true;
}

bool RequestQueue::pop_until(Request& out,
                             std::chrono::steady_clock::time_point deadline) {
  std::unique_lock<std::mutex> lock(mutex_);

  // wait_until(lock, deadline, predicate) sleeps until the predicate becomes
  // true OR the deadline is reached, and returns the predicate's FINAL value:
  //   true  -> woken by a real request or by shutdown,
  //   false -> the deadline expired first (timeout).
  // The predicate form also re-checks on every wake-up, absorbing spurious
  // wake-ups just like the plain pop() above.
  const bool signaled = cv_.wait_until(
      lock, deadline, [this] { return !queue_.empty() || shutdown_; });

  if (!signaled) return false;       // timed out: no request arrived in time
  if (queue_.empty()) return false;  // woken by shutdown with nothing left

  out = std::move(queue_.front());   // move the Request (and its promise) out
  queue_.pop();
  return true;
}

void RequestQueue::shutdown() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_ = true;
  }
  // Wake ALL waiters so every blocked consumer re-evaluates the predicate and
  // can exit its loop.
  cv_.notify_all();
}

}  // namespace inference_server
