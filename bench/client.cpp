#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "inference.grpc.pb.h"

using inference::Inference;
using inference::PredictRequest;
using inference::PredictResponse;

using Clock = std::chrono::steady_clock;

namespace {

// One benchmark thread == one simulated client. It fires Predict requests
// back-to-back (a "closed loop": send, wait for reply, send again) until
// `deadline`, recording each round-trip latency in microseconds.
void clientThread(const std::string& target,
                  int input_size,
                  Clock::time_point deadline,
                  std::vector<int64_t>* out_latencies_us,
                  std::atomic<uint64_t>* error_count) {
  // Each thread gets its own channel + stub. (A channel is thread-safe and
  // could be shared, but per-thread keeps the closed-loop model simple.)
  auto channel =
      grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
  std::unique_ptr<Inference::Stub> stub = Inference::NewStub(channel);

  // Build the request payload once and reuse it every iteration.
  PredictRequest request;
  request.mutable_input()->Resize(input_size, 1.0f);

  std::vector<int64_t> local;
  local.reserve(1 << 16);

  while (Clock::now() < deadline) {
    PredictResponse response;
    grpc::ClientContext context;

    const auto start = Clock::now();
    const grpc::Status status = stub->Predict(&context, request, &response);
    const auto end = Clock::now();

    if (status.ok()) {
      local.push_back(
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count());
    } else {
      error_count->fetch_add(1, std::memory_order_relaxed);
    }
  }

  *out_latencies_us = std::move(local);
}

// Nearest-rank percentile over an already-sorted vector.
int64_t percentileUs(const std::vector<int64_t>& sorted, double pct) {
  if (sorted.empty()) return 0;
  auto idx = static_cast<std::size_t>(pct / 100.0 * sorted.size());
  if (idx >= sorted.size()) idx = sorted.size() - 1;
  return sorted[idx];
}

}  // namespace

int main(int argc, char** argv) {
  // Args: [target] [threads] [duration_sec] [input_size]
  const std::string target = (argc > 1) ? argv[1] : "localhost:50051";
  const int num_threads = (argc > 2) ? std::stoi(argv[2]) : 8;
  const int duration_sec = (argc > 3) ? std::stoi(argv[3]) : 10;
  const int input_size = (argc > 4) ? std::stoi(argv[4]) : 3 * 224 * 224;

  std::cout << "Benchmark config:\n"
            << "  target     = " << target << "\n"
            << "  threads    = " << num_threads << "\n"
            << "  duration   = " << duration_sec << " s\n"
            << "  input_size = " << input_size << " floats\n\n"
            << "Running...\n";

  std::vector<std::thread> threads;
  std::vector<std::vector<int64_t>> per_thread_latencies(num_threads);
  std::atomic<uint64_t> error_count{0};

  const auto bench_start = Clock::now();
  const auto deadline = bench_start + std::chrono::seconds(duration_sec);

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(clientThread, target, input_size, deadline,
                         &per_thread_latencies[i], &error_count);
  }
  for (auto& t : threads) t.join();

  const auto bench_end = Clock::now();

  // Merge every thread's samples into one vector, then sort so we can index
  // percentiles directly.
  std::vector<int64_t> all;
  for (auto& v : per_thread_latencies) {
    all.insert(all.end(), v.begin(), v.end());
  }
  std::sort(all.begin(), all.end());

  const double elapsed_sec =
      std::chrono::duration_cast<std::chrono::duration<double>>(bench_end -
                                                                bench_start)
          .count();
  const uint64_t ok = all.size();
  const double throughput = (elapsed_sec > 0.0) ? ok / elapsed_sec : 0.0;

  std::cout << "\n===== Results =====\n";
  std::cout << "Successful inferences : " << ok << "\n";
  std::cout << "Errors                : " << error_count.load() << "\n";
  std::cout << "Elapsed               : " << elapsed_sec << " s\n";
  std::cout << "Throughput            : " << throughput << " inferences/sec\n";
  if (!all.empty()) {
    std::cout << "Latency p50           : " << percentileUs(all, 50) / 1000.0
              << " ms\n";
    std::cout << "Latency p95           : " << percentileUs(all, 95) / 1000.0
              << " ms\n";
    std::cout << "Latency p99           : " << percentileUs(all, 99) / 1000.0
              << " ms\n";
  } else {
    std::cout << "No successful requests -- is the server running at " << target
              << " ?\n";
  }
  return 0;
}
