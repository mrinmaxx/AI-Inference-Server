#pragma once

#include <grpcpp/grpcpp.h>

#include "request_queue.h"

#include "inference.grpc.pb.h"

namespace inference_server {

// gRPC service implementation. Its Predict() method runs on gRPC's own internal
// handler threads -- these are the PRODUCERS in our producer-consumer pipeline.
class InferenceServiceImpl final : public inference::Inference::Service {
 public:
  explicit InferenceServiceImpl(RequestQueue& queue) : queue_(queue) {}

  grpc::Status Predict(grpc::ServerContext* context,
                       const inference::PredictRequest* request,
                       inference::PredictResponse* response) override;

 private:
  RequestQueue& queue_;  // not owned; lives for the server's lifetime
};

// NOTE: Phase 1's single-worker consumer (modelWorker, batch-size-1) has been
// replaced by the dynamic Batcher (see batcher.h), which is now the sole
// consumer of the queue. Running the Batcher with max_batch_size == 1
// reproduces the old batch-size-1 behavior exactly. The Predict() handler above
// is unchanged: it still only pushes a request and waits on its future, and
// remains completely unaware that batching is happening downstream.

}  // namespace inference_server
