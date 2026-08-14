#include "server.h"

#include <utility>

namespace inference_server {

grpc::Status InferenceServiceImpl::Predict(
    grpc::ServerContext* /*context*/,
    const inference::PredictRequest* request,
    inference::PredictResponse* response) {
  // ----- PRODUCER side of the promise/future handshake -----

  // 1. Package the incoming RPC payload into a Request.
  Request req;
  req.input.assign(request->input().begin(), request->input().end());

  // 2. Grab OUR future now, while we still own the promise. This future is our
  //    private channel to the one prediction that belongs to this exact call.
  std::future<Tensor> result_future = req.result_promise.get_future();

  // 3. Hand the Request (promise included) to the worker via the queue. We MOVE
  //    it in, so after this line `req` is hollow and the worker owns the
  //    promise. This is the last we touch the Request itself.
  queue_.push(std::move(req));

  // 4. Block this handler thread until the worker calls set_value() on the
  //    matching promise. future.get() then returns exactly THIS request's
  //    prediction -- routed back correctly with zero manual bookkeeping.
  Tensor prediction = result_future.get();

  // 5. Serialize the prediction into the response and report success.
  //    protobuf's RepeatedField has no Assign(begin,end); reserve once (so the
  //    backing array is sized in a single allocation) then Add each value.
  auto* out = response->mutable_output();
  out->Reserve(static_cast<int>(prediction.size()));
  for (float v : prediction) out->Add(v);
  return grpc::Status::OK;
}

// The queue's consumer is now the dynamic Batcher (see batcher.cpp), which
// gathers many pending requests, runs the engine once per batch, and scatters
// the results back to each request's promise. Phase 1's batch-size-1
// modelWorker has been retired -- the Batcher with max_batch_size == 1 is its
// exact equivalent.

}  // namespace inference_server
