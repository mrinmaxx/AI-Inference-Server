#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

#include "batcher.h"
#include "engine.h"
#include "request_queue.h"
#include "server.h"

using namespace inference_server;

namespace {

// Runtime configuration. max_batch and max_wait_ms are the two knobs the
// benchmark sweeps -- both settable on the command line so no recompile is
// needed between sweep points.
struct Config {
  std::string address = "0.0.0.0:50051";
  std::size_t max_batch = 8;
  int max_wait_ms = 5;
  std::string engine = "dummy";           // "dummy" | "onnx"
  std::string model_path = "resnet50.onnx";
  std::string provider = "cpu";           // "cpu" | "cuda" (onnx only)
  int intra_op_threads = 0;               // 0 = ONNX Runtime default
};

void printUsage(const char* prog) {
  std::cerr
      << "Usage: " << prog << " [address] [options]\n"
      << "  address                 listen address (default 0.0.0.0:50051)\n"
      << "  --address=host:port     same, as a flag\n"
      << "  --max-batch=N           max requests per batch (default 8; 1 = baseline)\n"
      << "  --max-wait-ms=N         max ms to wait for a batch to fill (default 5)\n"
      << "  --engine=dummy|onnx     which inference engine to run (default dummy)\n"
      << "  --model-path=FILE       .onnx model for --engine=onnx (default resnet50.onnx)\n"
      << "  --provider=cpu|cuda     ONNX execution provider (default cpu; cuda = GPU)\n"
      << "  --intra-op-threads=N    ONNX Runtime CPU threads per Run() (0 = auto)\n"
      << "  -h, --help              show this help\n";
}

// Minimal flag parser supporting BOTH --key=value and --key value forms, plus
// one positional argument for the listen address (keeps Phase 1's
// `inference_server host:port` invocation working).
bool parseArgs(int argc, char** argv, Config& cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    // Consume the value for `key`, whether written as --key=value or --key value.
    // Advances `i` past a separate value token when needed.
    auto take = [&](const std::string& key, std::string& dst) -> bool {
      if (arg == key) {  // "--key value"
        if (i + 1 >= argc) return false;
        dst = argv[++i];
        return true;
      }
      if (arg.rfind(key + "=", 0) == 0) {  // "--key=value"
        dst = arg.substr(key.size() + 1);
        return true;
      }
      return false;
    };

    std::string v;
    if (arg == "-h" || arg == "--help") {
      printUsage(argv[0]);
      return false;
    } else if (take("--max-batch", v)) {
      cfg.max_batch = static_cast<std::size_t>(std::stoul(v));
    } else if (take("--max-wait-ms", v)) {
      cfg.max_wait_ms = std::stoi(v);
    } else if (take("--engine", v)) {
      cfg.engine = v;
    } else if (take("--model-path", v)) {
      cfg.model_path = v;
    } else if (take("--provider", v)) {
      cfg.provider = v;
    } else if (take("--intra-op-threads", v)) {
      cfg.intra_op_threads = std::stoi(v);
    } else if (take("--address", v)) {
      cfg.address = v;
    } else if (!arg.empty() && arg[0] != '-') {
      cfg.address = arg;  // positional address
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      printUsage(argv[0]);
      return false;
    }
  }
  if (cfg.max_batch == 0) cfg.max_batch = 1;
  if (cfg.max_wait_ms < 0) cfg.max_wait_ms = 0;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Config cfg;
  if (!parseArgs(argc, argv, cfg)) return 1;

  // ----- Assemble the pipeline -----

  // Select the engine at runtime. Both DummyEngine and OnnxEngine implement the
  // SAME InferenceEngine interface, so from here on everything (queue, batcher,
  // router, server) works through `InferenceEngine&` and neither knows nor
  // cares which concrete engine is behind it -- that is the drop-in seam.
  // Declared BEFORE the batcher so it is destroyed AFTER the batcher stops.
  std::unique_ptr<InferenceEngine> engine;
  std::string engine_desc;
  try {
    if (cfg.engine == "onnx") {
      // Loads the .onnx once here; throws with a clear message if it can't
      // (missing file, or --provider=cuda with a CUDA/cuDNN mismatch).
      engine = std::make_unique<OnnxEngine>(cfg.model_path, cfg.provider,
                                            cfg.intra_op_threads);
      engine_desc = "OnnxEngine(" + cfg.model_path + ", provider=" + cfg.provider + ")";
    } else if (cfg.engine == "dummy") {
      engine = std::make_unique<DummyEngine>(/*output_dim=*/1000);
      engine_desc = "DummyEngine (~20ms sleep per batch)";
    } else {
      std::cerr << "Unknown --engine='" << cfg.engine << "' (use dummy|onnx)\n";
      return 1;
    }
  } catch (const std::exception& e) {
    std::cerr << "Failed to initialize engine: " << e.what() << "\n";
    return 1;
  }

  std::cout << "Starting inference server\n"
            << "  address       = " << cfg.address << "\n"
            << "  max_batch     = " << cfg.max_batch << "\n"
            << "  max_wait_ms   = " << cfg.max_wait_ms << "\n"
            << "  engine        = " << engine_desc << "\n"
            << std::flush;

  // The buffer between the network layer (producers) and the batcher (consumer).
  RequestQueue queue;

  // The dynamic batcher is the SINGLE consumer of the queue. It runs on its own
  // thread; the gRPC handler threads only ever push onto the queue. With
  // --max-batch=1 this degenerates to Phase 1's one-request-per-run() baseline,
  // so the same binary produces both the "batching off" and "batching on"
  // numbers for the throughput table.
  Batcher batcher(queue, *engine, cfg.max_batch,
                  std::chrono::milliseconds(cfg.max_wait_ms));
  batcher.start();

  // Build and start the gRPC server. gRPC spawns handler threads internally --
  // those are the PRODUCERS that call InferenceServiceImpl::Predict.
  InferenceServiceImpl service(queue);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(cfg.address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  if (!server) {
    std::cerr << "Failed to start server on " << cfg.address << "\n";
    batcher.stop();  // clean up the batcher thread before bailing out
    return 1;
  }

  // std::endl to flush immediately even when stdout is redirected to a file
  // (the sweep script waits to see "listening" before firing the benchmark).
  std::cout << "Inference server listening on " << cfg.address << std::endl;
  std::cout << "Press Ctrl-C to stop." << std::endl;

  // Blocks until the server is shut down. (For now, Ctrl-C terminates the
  // process; the graceful path below runs if server->Shutdown() is ever wired
  // to a signal handler.)
  server->Wait();

  // Graceful shutdown: stop the batcher (it completes the in-flight batch,
  // drains the queue, and joins its thread) before exiting.
  batcher.stop();
  return 0;
}
