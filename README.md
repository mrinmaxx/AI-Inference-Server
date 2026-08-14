# Inference Server — a C++ AI model-serving system with dynamic batching

A minimal, educational version of NVIDIA Triton / TensorFlow Serving, written
from scratch in C++17. It puts a trained machine-learning model behind a network
API so many clients can ask it for predictions at once. Its defining feature is
**dynamic batching**: because ML models run far more efficiently on many inputs
at once, the server **collects** individual requests that arrive one-by-one,
**groups** them into a batch, **runs the model once** on the whole batch, and
**scatters** each result back to the client that asked for it.

It ships with two interchangeable "engines" behind one interface: a **DummyEngine**
(a fake model used to build and benchmark the plumbing) and an **OnnxEngine**
(a real ResNet-50 image classifier run via ONNX Runtime). The whole project is
built to *measure* one thing: **how much batching increases throughput.**

> This README is the single source of truth. Read only this file and you can
> understand the design *and* copy-paste every command to build, run, benchmark,
> regenerate the plots, and enable the GPU. Everything is here.

---

## Table of Contents

1. [The Problem This Solves](#1-the-problem-this-solves)
2. [Networking Primer](#2-networking-primer)
3. [Architecture Diagram](#3-architecture-diagram)
4. [The Journey of a Single Request](#4-the-journey-of-a-single-request)
5. [Component-by-Component Deep Dive](#5-component-by-component-deep-dive)
6. [The DummyEngine, Fully Explained](#6-the-dummyengine-fully-explained)
7. [Tech Stack](#7-tech-stack)
8. [Project Structure](#8-project-structure)
9. [Prerequisites & Installation](#9-prerequisites--installation)
10. [Getting the Model](#10-getting-the-model)
11. [Building](#11-building)
12. [Running the Server](#12-running-the-server)
13. [Running the Benchmark Client](#13-running-the-benchmark-client)
14. [Reproducing the Results / Running the Sweeps](#14-reproducing-the-results--running-the-sweeps)
15. [Understanding the Results](#15-understanding-the-results)
16. [Running on GPU (NVIDIA CUDA)](#16-running-on-gpu-nvidia-cuda)
17. [Troubleshooting](#17-troubleshooting)
18. [Design Decisions & Tradeoffs](#18-design-decisions--tradeoffs)
19. [Limitations & Honest Caveats](#19-limitations--honest-caveats)
20. [Possible Extensions](#20-possible-extensions)
21. [License](#21-license)

---

## 1. The Problem This Solves

Imagine a bakery with one big oven. The oven bakes a tray of cookies in 20
minutes — and it takes the *same* 20 minutes whether the tray holds 1 cookie or
30. But customers walk in one at a time and each orders a single cookie.

- **The naive way:** bake each customer's cookie by itself. One tray, one cookie,
  20 minutes, over and over. You serve **3 cookies/hour** and the oven is almost
  empty every run.
- **The smart way:** when a customer orders, don't bake immediately. Wait a
  moment to see if more customers arrive, load up to 30 cookies onto one tray,
  and bake them together. Now one 20-minute run serves **30 cookies**, and you
  serve up to **90 cookies/hour** — 30× more — from the *same oven*.

An ML model on modern hardware is exactly that oven. Running the model on a
"batch" of many inputs costs almost the same as running it on one, because the
hardware processes them in parallel. But inference requests arrive one at a time
over the network. So a good serving system does what the smart baker does:
briefly **collect** requests, **batch** them, run the model **once**, then hand
each result back. That collect-batch-run-scatter cycle is the heart of this
project, and **dynamic batching** is the name for doing it automatically with a
safety timer so no single request waits too long.

The catch (and a key finding of this project): the win only appears when the
"oven" has a **fixed cost per batch** with spare capacity — like a GPU. On a CPU
that's already fully busy computing one image, batching more images just takes
proportionally longer, so throughput stays flat. This project demonstrates
*both* regimes on purpose (see [§15](#15-understanding-the-results)).

---

## 2. Networking Primer

Plain-language definitions for the terms used throughout. Skip if you know them.

- **Server:** a long-running program that waits for other programs to connect and
  ask it to do something. Ours is `inference_server`; it waits for prediction
  requests.
- **Client:** a program that connects to a server and makes requests. Ours is
  `bench_client` (a load generator), but any program could be a client.
- **Request / Response:** a request is the message a client sends ("here is an
  image, classify it"); the response is what the server sends back ("here are the
  1000 class scores").
- **Port:** a numbered "door" on a machine (e.g. `50051`). A server listens on a
  specific port so clients know where to connect. Our default is
  `0.0.0.0:50051` — `0.0.0.0` means "listen on all network interfaces."
- **RPC (Remote Procedure Call):** a way to make calling a function on another
  machine look like calling a normal local function. The client calls
  `Predict(request)` and gets back a `response`, and the networking is hidden.
- **gRPC:** Google's high-performance RPC framework. You describe your service
  and messages once in a `.proto` file; gRPC generates the C++ client and server
  code (serialization, network I/O, threading) for you. We use a **unary** RPC
  (one request → one response). gRPC runs the server handler on its own internal
  pool of **handler threads**, so many clients can be served concurrently.
- **Protocol Buffers (protobuf):** the compact binary format gRPC uses to encode
  messages on the wire, also defined in the `.proto` file.

Three concurrency terms you'll meet in the code:

- **Producer–consumer:** a pattern where some threads *produce* work items into a
  shared buffer and other threads *consume* them. It decouples the two sides so
  they can run at different speeds. Here the gRPC handler threads are producers
  and the batcher thread is the consumer.
- **`std::promise` / `std::future`:** a one-shot, one-way channel between two
  threads. A `promise` is the "sending end" and its paired `future` is the
  "receiving end." One thread calls `future.get()` and *blocks* (waits) until
  another thread calls `promise.set_value(x)`; then `get()` returns `x`. We use
  it to route a prediction computed on the batcher thread back to the exact gRPC
  handler thread waiting for it.
- **`std::condition_variable`:** a signal that lets a thread *sleep* until some
  condition becomes true, instead of wasting CPU checking in a loop. The queue
  uses one so the batcher sleeps while the queue is empty and wakes the instant a
  request arrives.

---

## 3. Architecture Diagram

```
                 many concurrent clients (bench/client.cpp)
                       │  each sends: Predict(PredictRequest{input floats})
                       ▼
        ┌───────────────────────────────────────────────────────────┐
        │  NETWORK LAYER — gRPC service   (src/server.cpp)            │
        │  InferenceServiceImpl::Predict runs on gRPC handler threads │  ← PRODUCERS
        │   1. wrap payload in Request{ input, std::promise<Tensor> } │
        │   2. future = promise.get_future()                         │
        │   3. queue.push(std::move(request))                        │
        │   4. BLOCK on future.get()  ── never blocks on the model ──┼──┐
        └───────────────────────────────┬───────────────────────────┘  │
                                         │ move Request                 │
                                         ▼                              │
        ┌───────────────────────────────────────────────────────────┐  │
        │  REQUEST QUEUE   (src/request_queue.{h,cpp})               │  │
        │  std::queue + std::mutex + std::condition_variable        │  │
        │  blocking push() / pop() / pop_until(deadline)            │  │
        └───────────────────────────────┬───────────────────────────┘  │
                                         │ pop (one at a time)          │
                                         ▼                              │
        ┌───────────────────────────────────────────────────────────┐  │
        │  DYNAMIC BATCHER  (src/batcher.cpp) — its OWN thread       │  │ ← CONSUMER
        │  size-OR-timeout loop:                                    │  │
        │   • block for the 1st request                             │  │
        │   • pull more until max_batch_size OR max_wait_ms elapses  │  │
        │   • assemble N inputs -> one [N, ...] BatchedInput         │  │
        │                        │                                   │  │
        │                        ▼                                   │  │
        │           engine.run(batch)   ── ONE call for all N ──►    │  │
        │        ┌──────────────────────────────────────────┐       │  │
        │        │ INFERENCE ENGINE  (src/engine.{h,cpp})    │       │  │
        │        │  InferenceEngine (abstract interface)     │       │  │
        │        │   ├─ DummyEngine  (sleep ~20ms, fake out) │       │  │
        │        │   └─ OnnxEngine   (real ResNet-50 / ORT)  │       │  │
        │        └──────────────────────────────────────────┘       │  │
        │                        │ Output = one row per input        │  │
        │                        ▼                                   │  │
        │  RESPONSE ROUTER (in batcher.cpp: runBatchAndRoute)       │  │
        │   for each i:  batch[i].promise.set_value(output[i]) ──────┼──┘  unblocks
        └───────────────────────────────────────────────────────────┘     the right
                                                                           client's future
```

**Overview.** Requests flow left-to-right and top-to-bottom, then each result
takes a private shortcut *back* to its own client via the promise/future pair
(the dashed line on the right). The network layer only ever **pushes** work and
**waits** on a future — it never touches the model. The queue decouples the fast,
bursty network side from the single batcher thread. The batcher is the only
thing that talks to the model, and it always talks to it through the abstract
`InferenceEngine` interface, so it neither knows nor cares whether a dummy or a
real ResNet-50 is behind it.

---

## 4. The Journey of a Single Request

Follow one image from a client and back:

1. A client calls the `Predict` RPC with a `PredictRequest` containing 150,528
   floats (a 3×224×224 image, flattened).
2. gRPC hands the call to `InferenceServiceImpl::Predict` on one of its handler
   threads (`src/server.cpp`). The handler copies the floats into a `Request`
   struct, which also holds a fresh `std::promise<Tensor>`.
3. The handler calls `promise.get_future()` to get **its own** `future` — the
   private receiving end of the answer — then `std::move`s the whole `Request`
   (promise included) onto the **RequestQueue** and calls `future.get()`, which
   **blocks that handler thread**. Crucially, it is blocked on the *future*, not
   on the model, so gRPC's other handler threads stay free.
4. The **Batcher** thread, sleeping on the queue's condition variable, wakes when
   the request lands. It takes this request as the first of a new batch, starts a
   `max_wait_ms` timer, and keeps pulling more requests off the queue until it has
   `max_batch_size` of them **or** the timer expires.
5. The batcher concatenates all the collected inputs into one `BatchedInput` with
   a leading batch dimension (`[N, 3, 224, 224]` for ResNet-50) and calls
   `engine.run(batch)` **once**.
6. The engine returns an `Output` — one result row per input, in the same order.
7. The **Response Router** (inside the batcher) walks the batch and, for each
   request `i`, calls `batch[i].result_promise.set_value(output[i])`.
8. That `set_value` unblocks the exact `future.get()` from step 3 — *this*
   request's handler thread, and only that one, wakes up with *its* result.
9. The handler copies the result into a `PredictResponse` and returns it; gRPC
   sends it over the network to the client. Done.

No request IDs, no lookup tables: the promise *is* the private mailbox between
the batcher and the one client waiting.

---

## 5. Component-by-Component Deep Dive

Each component is described by **JOB** (why it exists), **HOW IT'S BUILT** (the
C++ mechanism), and **HOW IT WORKS** (at runtime).

### 5.1 Network Layer — gRPC (`proto/inference.proto`, `src/server.{h,cpp}`)

**JOB.** Expose the model over the network so remote clients can request
predictions, and do so without ever blocking on the model itself.

**HOW IT'S BUILT.** The wire contract is one unary RPC, defined in
`proto/inference.proto`:

```proto
service Inference {
  rpc Predict(PredictRequest) returns (PredictResponse);
}
message PredictRequest  { repeated float input  = 1; }  // flattened 3x224x224 = 150528 floats
message PredictResponse { repeated float output = 1; }  // 1000 class scores
```

CMake runs `protoc` to generate the C++ client/server stubs from this file.
`InferenceServiceImpl` (in `src/server.cpp`) subclasses the generated
`inference::Inference::Service` and overrides `Predict`.

**HOW IT WORKS.** gRPC calls `Predict` on its own handler threads (the
**producers**). The handler builds a `Request`, grabs its `future`, pushes the
`Request` to the queue, and blocks on `future.get()`. When the result arrives it
fills the response. That's the entire handler — it is deliberately dumb about
batching.

### 5.2 Request Queue (`src/request_queue.{h,cpp}`)

**JOB.** A thread-safe buffer between the many network threads and the single
batcher thread — the **producer–consumer decoupling point**.

**HOW IT'S BUILT.** A `std::queue<Request>` guarded by a `std::mutex`, plus a
`std::condition_variable`. All locking uses RAII guards (`std::lock_guard` /
`std::unique_lock`), never manual lock/unlock. Three operations:

- `push(Request)` — locks, moves the request in, unlocks, then notifies one waiter.
- `pop(Request&)` — blocks on the condition variable until the queue is non-empty
  or shut down; returns `false` only once the queue is shut down **and** drained.
- `pop_until(Request&, deadline)` — like `pop`, but also returns `false` if the
  `deadline` passes first. This is the timeout half of the batching loop.

**HOW IT WORKS.** The batcher sleeps inside `pop`/`pop_until` (zero CPU) until a
producer calls `push`, which wakes it. The condition variable is always waited on
with a predicate, which safely absorbs *spurious wake-ups* (a thread waking with
no matching notify). `shutdown()` flips a flag and wakes everyone so the batcher
can exit cleanly.

### 5.3 Dynamic Batcher — the heart (`src/batcher.{h,cpp}`)

**JOB.** Turn a stream of individual requests into batches and run the model once
per batch. This is where the throughput win is created.

**HOW IT'S BUILT.** A `Batcher` object owns a `std::thread` and a
`std::atomic<bool> running_` flag. `start()` launches the thread; `stop()` (also
called by the destructor) sets the flag, shuts the queue down, and joins — RAII
all the way. It holds references (not copies) to the shared queue and engine.

**HOW IT WORKS — the size-OR-timeout loop (read this carefully).** One iteration
of the loop builds and runs exactly one batch:

```
loop:
  1. Request first;
     if (!queue.pop(first)) break;          // block for the batch's FIRST request;
                                             // false => shutdown + drained => exit
  2. deadline = now() + max_wait_ms;         // the clock starts when the FIRST
                                             // request arrives, not before
     batch = [ move(first) ];

  3. while (batch.size() < max_batch_size):  // greedily fill the batch...
        Request next;
        if (!queue.pop_until(next, deadline)) break;   // ...until FULL or the
        batch.push_back(move(next));                   //    deadline passes
                                                        //    (whichever first)

  4. runBatchAndRoute(move(batch));          // assemble -> engine.run() -> scatter
```

Why both conditions? **`max_batch_size`** caps how big a batch gets (bigger =
more efficient, up to a point). **`max_wait_ms`** is the *safety valve*: if only
one request ever arrives, it must not wait forever for a batch that will never
fill, so after `max_wait_ms` the batcher runs whatever it has. Under heavy load
the size limit fires first (batches fill instantly); under light load the timer
fires first (latency is bounded). These two numbers are the **dials of the whole
experiment** — see [§14](#14-reproducing-the-results--running-the-sweeps).

### 5.4 Inference Engine — the swappable seam (`src/engine.{h,cpp}`)

**JOB.** Represent "the model" behind a single abstract interface so different
model backends drop in without touching anything else.

**HOW IT'S BUILT.** An abstract base class:

```cpp
class InferenceEngine {
 public:
  virtual ~InferenceEngine() = default;
  virtual Output run(const BatchedInput& batch) = 0;   // batch of inputs -> batch of outputs
};
```

where `Tensor = std::vector<float>`, `BatchedInput = std::vector<Tensor>`, and
`Output = std::vector<Tensor>` (defined in `src/request.h`). Two implementations
subclass it: `DummyEngine` and `OnnxEngine`. `main.cpp` picks one at runtime and
hands the batcher a base-class reference (`InferenceEngine&`).

**HOW IT WORKS.** The batcher calls `engine.run(batch)` and gets back one output
row per input. Because every caller only sees `InferenceEngine&`, swapping
`DummyEngine ↔ OnnxEngine` requires **zero** changes to the queue, batcher,
router, or server. That clean seam is the payoff of the design — it's what let
the entire system be built and benchmarked with a fake model first, then have a
real one dropped in later.

`OnnxEngine` specifically: its constructor loads the `.onnx` file **once** via
ONNX Runtime (`Ort::Env` + `Ort::Session`); `run()` copies the N inputs into one
contiguous `[N,3,224,224]` buffer, calls `session.Run()` a single time, and
slices the `[N,1000]` output back into one `Tensor` per request. The
`Ort::Session` is owned by the engine and released on shutdown (RAII). It uses
the CPU execution provider by default, or CUDA if requested (see
[§16](#16-running-on-gpu-nvidia-cuda)).

### 5.5 Response Router (`src/batcher.cpp`, `runBatchAndRoute`)

**JOB.** The mirror image of the batcher: take the one batched output and deliver
each row to the right client.

**HOW IT'S BUILT.** A loop over the batch inside the batcher thread. Order was
preserved through assembly, so `output[i]` corresponds to `batch[i]`.

**HOW IT WORKS.** For each `i`, `batch[i].result_promise.set_value(output[i])`.
Each `set_value` unblocks exactly the one `future.get()` waiting in the matching
gRPC handler. Gather many in, scatter many out. (Defensive detail: if an engine
ever returned fewer rows than inputs, the leftover promises are fulfilled with an
empty tensor so no client hangs forever — the real engines never trigger this.)

### 5.6 Main (`src/main.cpp`)

**JOB.** Parse flags, wire the components together, start the server, and shut
down cleanly.

**HOW IT'S BUILT.** A small hand-written flag parser (supports `--key=value` and
`--key value`), then: construct the chosen engine into a
`std::unique_ptr<InferenceEngine>`, construct the `RequestQueue`, construct and
`start()` the `Batcher`, build the gRPC server, and call `server->Wait()`.

**HOW IT WORKS.** The engine is declared **before** the batcher so it outlives it
(the batcher is destroyed — and joined — first). On shutdown, `batcher.stop()`
drains in-flight work and joins the thread. Engine construction is wrapped in
`try/catch`: a missing model file or a failed CUDA provider prints a clear
message and exits non-zero instead of crashing.

### 5.7 Benchmark Client (`bench/client.cpp`)

**JOB.** Measure the server: simulate many concurrent clients and report
throughput and latency percentiles.

**HOW IT'S BUILT.** Spawns `N` `std::thread`s. Each thread opens its own gRPC
channel + stub and runs a **closed loop** (send a request, wait for the reply,
send the next) until a fixed deadline, recording every round-trip latency. At the
end it merges all latencies, sorts them, and computes percentiles by nearest-rank
indexing.

**HOW IT WORKS.** See [§13](#13-running-the-benchmark-client) for the exact
command and output. It reports **throughput** = completed requests ÷ elapsed
seconds, and **p50/p95/p99** latency in milliseconds.

---

## 6. The DummyEngine, Fully Explained

This is the most misunderstood part of the project, so here it is in full.

**WHAT it is.** `DummyEngine` is a *fake stand-in for a real model*. Its `run()`
does **no AI math whatsoever**. It simply sleeps for a fixed ~20 ms and returns
correctly-sized placeholder outputs (a `Tensor` of 1000 floats per input, with
the input's size stashed in element 0 so outputs aren't all identical).

**WHY it exists.** It lets the *entire serving architecture* — the gRPC layer,
the queue, the batcher, the promise/future routing, the response router, the
benchmark client — be built, tested, and benchmarked with **no real model, no
weights, no ONNX Runtime, no GPU, and no ML dependencies at all**. It is
deterministic and instant to start, so you can iterate on the hard part (the
concurrency and batching machinery) without any model in the way.

**HOW it plugs in.** It implements the exact same `InferenceEngine` interface as
`OnnxEngine`. Because everything upstream only sees `InferenceEngine&`, switching
between them is a single `--engine` flag with **zero code changes anywhere else**.
That swap-ability is the entire point of the interface — the DummyEngine is the
proof that the seam works.

**THE KEY INSIGHT.** `DummyEngine::run()` sleeps a **fixed ~20 ms per batch,
regardless of batch size**. A batch of 32 costs the same ~20 ms as a batch of 1.
This deliberately imitates the **fixed-cost-per-batch regime** — the situation on
a GPU that has spare parallel capacity, where the per-batch launch/latency cost
dominates and each extra item in the batch is nearly free. In that regime,
batching produces a *dramatic* throughput win: a batch of 32 does 32× the work
for the same 20 ms.

This is exactly why the **dummy sweep shows a large throughput gain while a
compute-bound real model on a CPU stays flat**: the CPU has no spare capacity —
one image already saturates the cores, so a batch of 32 genuinely takes ~32× as
long. **That contrast is an intentional, informative result of the project, not
a bug.** It's the whole lesson about *when* batching helps.

**Crucial clarification.** The DummyEngine is **NOT a GPU** and does **NOT run on
one**. It only *behaviorally imitates* the fixed-per-batch-cost regime by
sleeping. It is pure CPU `std::this_thread::sleep_for`. To see the real thing on
real hardware, use the OnnxEngine on a GPU ([§16](#16-running-on-gpu-nvidia-cuda)).

---

## 7. Tech Stack

| Dependency | Why this choice |
|---|---|
| **C++17** | Manual control over threads/memory for a performance-sensitive server; `std::promise`/`future`/`atomic` in the standard library. |
| **gRPC** | Battle-tested, high-performance RPC with a built-in threaded server; generates all the networking boilerplate from one `.proto`. |
| **Protocol Buffers** | Compact, fast, schema-defined message format that gRPC uses on the wire. |
| **ONNX Runtime (C++ API)** | Runs a real model without a training framework; one API for CPU *and* GPU (just swap the execution provider); loads the portable `.onnx` format. |
| **CMake** | Standard C++ build system; handles the `protoc` code-gen step and finds gRPC/protobuf/ONNX Runtime. |
| **CPU execution provider** (default) | No GPU required to build, run, or benchmark the whole system. |
| **ResNet-50 (ONNX)** | A well-known image classifier; input `[batch,3,224,224]`, output 1000 class scores — a realistic, non-trivial compute load. |
| **WSL2 / Linux, GCC** | Developed and tested on Ubuntu 22.04 under WSL2 with GCC 11. |
| **Python + matplotlib** (results only) | Small scripts turn the benchmark CSVs into the summary table and plots. Not needed to build or run the server. |

---

## 8. Project Structure

```
inference-server/
├── CMakeLists.txt              Build: finds gRPC/protobuf (pkg-config) + ONNX Runtime; runs protoc; builds both executables
├── README.md                   This file
├── proto/
│   └── inference.proto         gRPC service (unary Predict) + PredictRequest/PredictResponse messages
├── src/
│   ├── request.h               Request struct (input + std::promise); Tensor/BatchedInput/Output typedefs; the promise/future flow comment
│   ├── request_queue.h/.cpp    Thread-safe blocking queue (mutex + condition_variable): push / pop / pop_until / shutdown
│   ├── batcher.h/.cpp          Dynamic Batcher: its own thread, the size-OR-timeout loop, batch assembly, and the Response Router
│   ├── engine.h/.cpp           InferenceEngine interface + DummyEngine + OnnxEngine (ONNX Runtime, CPU or CUDA; PIMPL hides Ort headers)
│   ├── server.h/.cpp           gRPC InferenceServiceImpl::Predict (the network handler; pushes to queue, waits on future)
│   └── main.cpp                Flag parsing, wiring, server startup, graceful shutdown
├── bench/
│   └── client.cpp              Multi-threaded benchmark client (throughput + p50/p95/p99)
├── scripts/
│   ├── export_resnet50.py      Exports ResNet-50 to resnet50.onnx with a FLEXIBLE batch axis
│   ├── sweep.sh                Runs the server across configs, writes results CSV (batch sweep AND load sweep)
│   ├── make_table.py           results.csv -> console table + results_table.md
│   └── plot.py                 results.csv/results_load.csv -> throughput_vs_batch.png + latency_vs_throughput.png
├── results/
│   ├── README.md               Results writeup (both engines, side by side)
│   ├── onnx/                    Real ResNet-50 (CPU) sweep outputs: results.csv, results_load.csv, results_table.md, *.png, logs/
│   └── dummy/                   DummyEngine sweep outputs (same file set)
├── resnet50.onnx               The exported model (produced by scripts/export_resnet50.py; ~100 MB, not usually committed)
└── build/                      CMake build directory (generated; contains build/inference_server and build/bench_client)
```

---

## 9. Prerequisites & Installation

Target OS: **Linux or WSL2** (developed on Ubuntu 22.04). You need a C++17
compiler, CMake ≥ 3.16, gRPC + Protocol Buffers (dev libraries + the `protoc`
and `grpc_cpp_plugin` compilers), and — for the real model — ONNX Runtime's C++
package. **CPU-only needs no GPU or NVIDIA driver.**

### 9.1 Compiler + build tools + gRPC/protobuf (apt)

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config \
    protobuf-compiler protobuf-compiler-grpc \
    libprotobuf-dev libgrpc++-dev
```

This gives GCC, CMake, and gRPC/protobuf. Ubuntu ships these as **pkg-config**
modules (not CMake "config packages"), which is exactly what `CMakeLists.txt`
expects by default. Verify:

```bash
cmake --version                 # >= 3.16
protoc --version                # libprotoc 3.x
pkg-config --modversion grpc++  # e.g. 1.30.2
which grpc_cpp_plugin           # /usr/bin/grpc_cpp_plugin
```

> If you instead built gRPC from source or use vcpkg (which *do* ship CMake
> config packages), configure with `-DUSE_CMAKE_CONFIG=ON` to take that path.

### 9.2 ONNX Runtime C++ (CPU build) — the common failure point

ONNX Runtime's C++ package is **not** an apt package. You download a prebuilt
tarball, extract it, and point CMake at it. This is the step most likely to trip
you up, so it's explicit:

```bash
ORT_VER=1.17.1
cd ~   # or wherever you keep dependencies
wget https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-linux-x64-${ORT_VER}.tgz
tar xzf onnxruntime-linux-x64-${ORT_VER}.tgz
# The extracted folder contains include/ and lib/. Remember this path:
export ONNXRUNTIME_ROOT=$PWD/onnxruntime-linux-x64-${ORT_VER}
ls "$ONNXRUNTIME_ROOT/include/onnxruntime_cxx_api.h"   # must exist
ls "$ONNXRUNTIME_ROOT/lib/libonnxruntime.so"           # must exist
```

You pass `-DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT"` to CMake in [§11](#11-building).
(On this machine it lives at `/root/onnxruntime` — a symlink to
`onnxruntime-linux-x64-1.17.1`.)

> **Don't want the real model at all?** Build dummy-only with `-DWITH_ONNX=OFF`
> and skip this section entirely — no ONNX Runtime needed.

### 9.3 Python (only to export the model and make plots)

```bash
python3 -m pip install --upgrade pip
python3 -m pip install matplotlib                      # for scripts/plot.py
# torch/torchvision/onnx are ONLY needed to EXPORT the model (see §10):
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install onnx onnxscript
```

`matplotlib` is the only Python dep needed to regenerate plots. `torch`,
`torchvision`, `onnx`, `onnxscript` are needed *only* to export `resnet50.onnx`
once; after that you can uninstall them to save space.

---

## 10. Getting the Model

The server needs a ResNet-50 in `.onnx` format whose **batch dimension is
flexible**. This is non-negotiable: if the model's batch axis is fixed (e.g.
locked to 1), the batcher literally cannot feed it a batch of 8/16/32 and the
entire project has nothing to demonstrate. The export below marks batch as a
*dynamic axis* so the model accepts any N.

Run the provided export script (needs the Python deps from §9.3):

```bash
python3 scripts/export_resnet50.py resnet50.onnx
```

It writes a single self-contained `resnet50.onnx` (~100 MB) and prints
confirmation that the input shape is `[batch, 3, 224, 224]`. The script is small
and does exactly this:

```python
import os, sys, torch, torchvision
out = sys.argv[1] if len(sys.argv) > 1 else "resnet50.onnx"
model = torchvision.models.resnet50(weights=None).eval()   # random weights OK: this is a THROUGHPUT project
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy, out,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},  # <-- FLEXIBLE batch axis (required)
    opset_version=13,
)
# torch 2.x may spill weights into a sidecar "<out>.data"; consolidate to ONE file,
# and pin IR version to 9 so older ONNX Runtime (<=1.17) can load it.
import onnx
m = onnx.load(out); m.ir_version = 9
onnx.save_model(m, out, save_as_external_data=False)
if os.path.exists(out + ".data"): os.remove(out + ".data")
```

Verify the batch axis is dynamic (should print a *name* like `batch` for dim 0,
not a fixed `1`):

```bash
python3 -c "import onnx;d=onnx.load('resnet50.onnx',load_external_data=False).graph.input[0].type.tensor_type.shape.dim;print([x.dim_param or x.dim_value for x in d])"
# expect: ['batch', 3, 224, 224]
```

> Accuracy note: the export uses **random weights** on purpose. This project
> measures **throughput**, not classification accuracy, so real pretrained
> weights would just be wasted bandwidth. Shapes are what matter.
>
> A direct download of a public ResNet-50 ONNX is possible, but many public files
> are pinned to batch size 1 — which breaks batching. If you download one instead
> of exporting, verify its batch dim is a symbolic name using the command above.

---

## 11. Building

From a clean clone, with ONNX Runtime available (see §9.2):

```bash
cd inference-server
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT"
cmake --build build -j"$(nproc)"
```

This produces two executables:

- `build/inference_server`
- `build/bench_client`

At configure time CMake prints which gRPC/protobuf it found and whether the ONNX
Runtime it linked is a **CPU-only** or **GPU** build — watch for:

```
-- ONNX Runtime library : /.../lib/libonnxruntime.so
-- ONNX Runtime build   : CPU-only (no CUDA provider present)
```

**Dummy-only build** (no ONNX Runtime needed at all):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWITH_ONNX=OFF
cmake --build build -j"$(nproc)"
```

**Clean rebuild** (throw away all build state):

```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT"
cmake --build build -j"$(nproc)"
```

> The ONNX Runtime library directory is baked into the executable's RPATH, so you
> do **not** need to set `LD_LIBRARY_PATH` to run it.

---

## 12. Running the Server

Start the server (dummy engine, default port `0.0.0.0:50051`):

```bash
./build/inference_server --engine=dummy --max-batch=8 --max-wait-ms=5
```

Real ResNet-50 on CPU:

```bash
./build/inference_server --engine=onnx --model-path=resnet50.onnx --max-batch=16 --max-wait-ms=5
```

On startup it prints its configuration and `Inference server listening on
0.0.0.0:50051`. Stop it with `Ctrl-C`.

### Flags

Both `--key=value` and `--key value` forms work. A bare first argument is treated
as the listen address (backward-compatible).

| Flag | Meaning | Default | Example |
|---|---|---|---|
| `[address]` (positional) or `--address=HOST:PORT` | Address to listen on | `0.0.0.0:50051` | `--address=0.0.0.0:50070` |
| `--engine=dummy\|onnx` | Which engine to serve | `dummy` | `--engine=onnx` |
| `--model-path=FILE` | `.onnx` model (used only for `--engine=onnx`) | `resnet50.onnx` | `--model-path=resnet50.onnx` |
| `--provider=cpu\|cuda` | ONNX execution provider (`cuda` = GPU) | `cpu` | `--provider=cuda` |
| `--max-batch=N` | Max requests per batch (`1` = batching off / baseline) | `8` | `--max-batch=32` |
| `--max-wait-ms=N` | Max ms to wait for a batch to fill (the safety-valve timer) | `5` | `--max-wait-ms=10` |
| `--intra-op-threads=N` | ONNX Runtime CPU threads per `Run()` (`0` = auto) | `0` | `--intra-op-threads=8` |
| `-h`, `--help` | Print usage and exit | — | `--help` |

Environment variable: `ORT_LOG_SEVERITY=verbose|info|warning|error` raises ONNX
Runtime's log verbosity (use `verbose` to see per-node execution-provider
placement — handy for confirming CUDA is really in use).

---

## 13. Running the Benchmark Client

With a server already running, in a second terminal:

```bash
# args (all positional):  [target]  [threads]  [duration_sec]  [input_size]
./build/bench_client localhost:50051 64 20 150528
```

| Positional arg | Meaning | Default |
|---|---|---|
| `target` | `HOST:PORT` of the server | `localhost:50051` |
| `threads` | Number of concurrent simulated clients | `8` |
| `duration_sec` | How long to fire requests | `10` |
| `input_size` | Floats per request (`3*224*224 = 150528` for ResNet-50) | `150528` |

Sample output:

```
Benchmark config:
  target     = localhost:50051
  threads    = 64
  duration   = 20 s
  input_size = 150528 floats

Running...

===== Results =====
Successful inferences : <N>
Errors                : 0
Elapsed               : 20.0 s
Throughput            : <T> inferences/sec
Latency p50           : <p50> ms
Latency p95           : <p95> ms
Latency p99           : <p99> ms
```

Each client thread runs closed-loop (send → wait → send). **Throughput** =
successful requests ÷ elapsed. **p50/p95/p99** = the 50th/95th/99th percentile of
all recorded round-trip latencies (sort all samples, index by percentile rank).

---

## 14. Reproducing the Results / Running the Sweeps

Everything needed to regenerate the numbers and plots from zero.

### 14.1 The sweep script

`scripts/sweep.sh` starts the server for each configuration, warms it up, runs
the benchmark client, records one CSV row, and cleanly stops the server before
the next config (a `trap` kills any server on exit, so no orphan processes). It
has **two modes**:

- **`MODE=batch`** (default) → sweeps `max_batch` × `max_wait_ms` at fixed
  concurrency → `results/[SUBDIR/]results.csv`. Shows the batching throughput
  win. `batch=1` is the explicit baseline.
- **`MODE=load`** → fixes the batch, sweeps client concurrency →
  `results/[SUBDIR/]results_load.csv`. Produces the latency/throughput tradeoff
  curve (you need to vary *offered load* to see tail latency rise).

**Tunable parameters** live at the top of `scripts/sweep.sh`; every one is also
overridable as an environment variable on the command line:

| Variable | Meaning | Default |
|---|---|---|
| `MODE` | `batch` or `load` | `batch` |
| `ENGINE` | `onnx` or `dummy` | `onnx` |
| `MODEL_PATH` | `.onnx` file (onnx engine) | `resnet50.onnx` |
| `PROVIDER` | `cpu` or `cuda` | `cpu` |
| `INTRA_OP_THREADS` | ORT CPU threads per Run (0 = auto) | `0` |
| `DURATION` | measured seconds per config | `20` |
| `WARMUP` | discarded warm-up seconds per config | `3` |
| `INPUT_SIZE` | floats per request | `150528` |
| `RESULTS_SUBDIR` | subfolder under `results/` (e.g. `onnx`, `dummy`, `gpu`) | *(none)* |
| `BASE_PORT` | first port (each run uses a fresh one) | `50060` |
| `BATCH_SIZES` | (batch mode) batch sizes to sweep | `1 2 4 8 16 32` |
| `WAIT_MS` | (batch mode) `max_wait_ms` values | `1 5 10` |
| `CONCURRENCY` | (batch mode) fixed client count | `64` |
| `LOAD_BATCH` | (load mode) fixed batch size | `16` |
| `LOAD_WAIT` | (load mode) fixed `max_wait_ms` | `5` |
| `CONCURRENCY_LIST` | (load mode) client counts to sweep | `1 2 4 8 16 32 64 128` |

Edit them in the file, or override inline, e.g.:
`DURATION=30 CONCURRENCY=128 ./scripts/sweep.sh`.

### 14.2 Run each experiment variant

**Real ResNet-50 on CPU** (throughput sweep, then load sweep):

```bash
RESULTS_SUBDIR=onnx ENGINE=onnx MODEL_PATH=resnet50.onnx ./scripts/sweep.sh
RESULTS_SUBDIR=onnx ENGINE=onnx MODEL_PATH=resnet50.onnx MODE=load ./scripts/sweep.sh
```
→ `results/onnx/results.csv` and `results/onnx/results_load.csv`.

**DummyEngine** (the fixed-cost regime; the dramatic-win curve):

```bash
RESULTS_SUBDIR=dummy ENGINE=dummy ./scripts/sweep.sh
RESULTS_SUBDIR=dummy ENGINE=dummy MODE=load ./scripts/sweep.sh
```
→ `results/dummy/results.csv` and `results/dummy/results_load.csv`.

(For the **GPU** variant see [§16](#16-running-on-gpu-nvidia-cuda) →
`results/gpu/`.)

### 14.3 Generate the summary table

```bash
python3 -m pip install matplotlib     # (matplotlib not needed for the table, but is for plots below)
RESULTS_SUBDIR=onnx  python3 scripts/make_table.py     # onnx table  -> console + results/onnx/results_table.md
RESULTS_SUBDIR=dummy python3 scripts/make_table.py     # dummy table -> console + results/dummy/results_table.md
```

`make_table.py` reads `results/[SUBDIR/]results.csv` and writes
`results_table.md` (columns: Config, Throughput, p50, p95, p99, Speedup-vs-baseline)
next to it, and prints it to the console.

### 14.4 Regenerate the plots (the "how do I get the plots again" answer)

The plots come from `scripts/plot.py`, which only needs **matplotlib**:

```bash
python3 -m pip install matplotlib
RESULTS_SUBDIR=onnx  python3 scripts/plot.py
RESULTS_SUBDIR=dummy python3 scripts/plot.py
```

For each subdir this writes two PNGs **next to the CSVs**:

- `results/<subdir>/throughput_vs_batch.png` — throughput vs. `max_batch`, one
  line per `max_wait_ms` (from `results.csv`).
- `results/<subdir>/latency_vs_throughput.png` — the latency/throughput tradeoff.
  It uses `results_load.csv` (the load sweep) when present for the classic
  up-slope; otherwise it falls back to `results.csv`.

That's the complete regeneration path: **sweep → CSV → `make_table.py` →
`plot.py` → PNGs**, all reproducible from zero.

---

## 15. Understanding the Results

### 15.1 What the metrics mean

- **Throughput (inferences/sec):** total successful predictions ÷ elapsed
  seconds. Higher is better; this is the number batching is meant to raise.
- **Latency p50 / p95 / p99 (ms):** the round-trip time a client waits, at the
  50th (median), 95th, and 99th percentile. p99 = "99% of requests were at least
  this fast" — the *tail*, which users feel. The client collects every latency,
  sorts them, and indexes by percentile rank.
- **Little's Law sanity check:** for closed-loop load, `clients ≈ throughput ×
  latency`. If `throughput × p50` reproduces your client count, the measurement
  is trustworthy.

### 15.2 Result tables (placeholders — fill from YOUR runs)

Run the sweeps (§14) and read the generated `results/onnx/results_table.md` and
`results/dummy/results_table.md`. They look like this (numbers are placeholders —
**do not treat these as real**):

| Config | Throughput (inf/s) | p50 (ms) | p95 (ms) | p99 (ms) | Speedup |
|---|---|---|---|---|---|
| batch=1 (baseline) | `<T1>` | `<...>` | `<...>` | `<...>` | 1.00× |
| batch=8 | `<T8>` | `<...>` | `<...>` | `<...>` | `<T8/T1>×` |
| batch=32 | `<T32>` | `<...>` | `<...>` | `<...>` | `<T32/T1>×` |

The actual measured numbers for this machine live in the committed
`results/onnx/results_table.md` and `results/dummy/results_table.md` and in
`results/README.md`.

### 15.3 How to read the curves

- **`throughput_vs_batch.png` — a rising-then-plateauing curve.** The **rise**
  means batching is working: each batch serves more requests per model call. The
  **plateau** means the bottleneck has shifted — you've stopped being limited by
  per-request overhead and are now limited by raw compute or by how many requests
  are actually in flight. Past the plateau, bigger batches don't help.
- **`latency_vs_throughput.png` — an upward slope / a knee.** As you push more
  load, throughput climbs until it **saturates**; past that **knee**, extra load
  can't raise throughput and instead just **piles up in the queue, so p99 latency
  climbs steeply**. This is the fundamental latency-vs-throughput tradeoff.
- **The Dummy-vs-CPU contrast (the headline finding).** The **DummyEngine** curve
  rises sharply — batching wins big — because its cost is *fixed per batch*
  (the accelerator regime). The **real ResNet-50 on CPU** curve is roughly
  **flat** — batching barely helps — because CPU inference is *compute-bound*:
  one image already saturates the cores, so N images cost ~N× as long. Flat here
  is **expected and correct**, not a bug (verified: each run used the right
  `max_batch`, and per-image time is constant across batch sizes). It's the whole
  point — batching's throughput win needs an accelerator-like fixed-cost profile,
  which a GPU provides and a saturated CPU does not.

Plot files referenced: `results/dummy/throughput_vs_batch.png`,
`results/onnx/throughput_vs_batch.png`,
`results/dummy/latency_vs_throughput.png`,
`results/onnx/latency_vs_throughput.png`.

---

## 16. Running on GPU (NVIDIA CUDA)

A complete, self-contained procedure to enable GPU inference later, from scratch.
The code already supports it (`--provider=cuda`); this section is the environment
setup + build + verify.

### 16.1 Context (WSL2 + driver)

This project runs in **WSL2**. On WSL2, the NVIDIA GPU is exposed to Linux via
**passthrough from the Windows driver** — `nvidia-smi` and `/usr/lib/wsl/lib/
libcuda.so` appear automatically. **Do NOT install an NVIDIA driver inside WSL2**
— that breaks the passthrough. Install the *toolkit/runtime libraries* only; the
driver stays on the Windows side. Confirm the GPU is visible first:

```bash
nvidia-smi        # should list your GPU and a max supported CUDA version
```

### 16.2 Pinned versions (version mismatch is the #1 failure)

Match ONNX Runtime's GPU build to the CUDA and cuDNN it was compiled against:

| Component | Exact version |
|---|---|
| ONNX Runtime GPU | **onnxruntime-linux-x64-gpu-1.17.1** |
| CUDA runtime | **11.8** |
| cuDNN | **8.9.2.x for CUDA 11.x** |

### 16.3 Install (runtime-only — small footprint)

The server only needs to **run** on the GPU, not compile CUDA code, so you do
**not** need the full ~5 GB CUDA Toolkit (with `nvcc`, docs, profilers). Install
just the runtime libraries ONNX Runtime dlopen()s, which is far smaller
(~1.5–2 GB total). **Check free disk first** — and remember that on WSL2 the
virtual disk grows on your Windows C: drive:

```bash
df -h /            # ensure a few GB free before installing
```

Add NVIDIA's **WSL-Ubuntu** repo (driverless) and install the CUDA 11.8 runtime
libraries + cuDNN:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
# runtime libraries only (NOT the full 'cuda-toolkit-11-8'):
sudo apt-get install -y --no-install-recommends \
    cuda-cudart-11-8 libcublas-11-8 libcufft-11-8 libcurand-11-8 cuda-nvrtc-11-8
# cuDNN 8.9.2 for CUDA 11.x (if the pinned version isn't in the repo, use the pip
# wheel: python3 -m pip install nvidia-cudnn-cu11==8.9.2.26, then add its
# .../site-packages/nvidia/cudnn/lib to LD_LIBRARY_PATH):
sudo apt-get install -y libcudnn8=8.9.2.*-1+cuda11.8 || \
    python3 -m pip install nvidia-cudnn-cu11==8.9.2.26
```

Then get the ONNX Runtime **GPU** build (ships the extra CUDA provider libs):

```bash
ORT_VER=1.17.1
wget https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-linux-x64-gpu-${ORT_VER}.tgz
tar xzf onnxruntime-linux-x64-gpu-${ORT_VER}.tgz
export ONNXRUNTIME_GPU_ROOT=$PWD/onnxruntime-linux-x64-gpu-${ORT_VER}
ls "$ONNXRUNTIME_GPU_ROOT/lib/libonnxruntime_providers_cuda.so"   # MUST exist (the CPU package lacks it)
```

> After a heavy CUDA install on WSL2, the WSL virtual disk (`ext4.vhdx`) grows on
> C: and does not auto-shrink. If space is tight later, from **Windows
> PowerShell**: `wsl --shutdown` then `wsl --manage <Distro> --set-sparse true`
> (or compact the `ext4.vhdx` with `diskpart`).

### 16.4 Build against the GPU runtime

Point CMake at the **GPU** ONNX Runtime and rebuild:

```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT="$ONNXRUNTIME_GPU_ROOT"
cmake --build build -j"$(nproc)"
```

At configure time you should now see:

```
-- ONNX Runtime build   : GPU (CUDAExecutionProvider available)
```

### 16.5 Run on the GPU + VERIFY it's actually using CUDA

```bash
./build/inference_server --engine=onnx --model-path=resnet50.onnx --provider=cuda --max-batch=16
```

**Confirm CUDA is active and did NOT silently fall back to CPU** — check all of:

1. **Startup log lines** (printed by the engine):
   ```
   ONNX Runtime providers compiled into this build: ... CUDAExecutionProvider ...
   OnnxEngine: requested execution provider = cuda
   OnnxEngine: model loaded; session created with requested provider = cuda
   ```
   The design **fails loudly**: if the CUDA provider can't initialize (version
   mismatch, missing cuDNN, no GPU), the server throws a clear error and exits —
   it does *not* quietly run on CPU. If you see `session created with requested
   provider = cuda`, the CUDA provider initialized.
2. **GPU activity** — while a benchmark runs, in another terminal:
   ```bash
   nvidia-smi     # VRAM usage jumps well above idle and GPU-Util > 0%
   ```
   On CPU the VRAM stays at idle; on CUDA it rises (ResNet-50 uses a few hundred
   MB+). This is the unambiguous check.
3. **Per-node placement (optional, definitive):** run once with
   `ORT_LOG_SEVERITY=verbose ./build/inference_server ... --provider=cuda` and
   look for nodes assigned to `CUDAExecutionProvider` in the log.

### 16.6 GPU sweep

```bash
RESULTS_SUBDIR=gpu ENGINE=onnx PROVIDER=cuda MODEL_PATH=resnet50.onnx \
    BATCH_SIZES="1 2 4 8 16 32" ./scripts/sweep.sh
RESULTS_SUBDIR=gpu python3 scripts/make_table.py
RESULTS_SUBDIR=gpu python3 scripts/plot.py
```
→ `results/gpu/results.csv`, `results/gpu/results_table.md`, and the PNGs,
separate from the CPU results.

### 16.7 Honest expectation

On a small, entry-level GPU (e.g. GTX 1650, 896 cores, 4 GB VRAM) expect a
**moderate rise-then-plateau (~2–5×), plateauing early** — not a huge multiplier,
and cap the sweep at batch 32 to stay within VRAM. The **rising shape** (unlike
the flat CPU curve) is the point, not the size of the number. A large data-center
GPU would show a much bigger rise.

---

## 17. Troubleshooting

| Symptom | Cause & Fix |
|---|---|
| CMake: `WITH_ONNX=ON but ONNX Runtime was not found` | You didn't pass `-DONNXRUNTIME_ROOT`, or it doesn't point at the folder containing `include/` and `lib/`. Set it (see §9.2), or build dummy-only with `-DWITH_ONNX=OFF`. |
| CMake: `find_package(gRPC CONFIG)` fails / gRPC not found | Your distro ships gRPC via **pkg-config**, not CMake config. This is the default path — just install `libgrpc++-dev protobuf-compiler-grpc` (§9.1). Only use `-DUSE_CMAKE_CONFIG=ON` for source/vcpkg builds. |
| Build error: `RepeatedField has no member 'Assign'` | Old protobuf. The code already uses `Reserve`+`Add`; make sure you're on the current source. |
| Server: `Failed to initialize engine: model file not found` | Wrong `--model-path`, or you didn't export the model. Run `python3 scripts/export_resnet50.py` (§10). |
| Server: `expected input of 150528 floats ... got N` | Client sent the wrong tensor size. Use `input_size=150528` (= 3×224×224) in the benchmark client. |
| `--provider=cuda` throws `... has NO CUDAExecutionProvider` | You linked the **CPU** ONNX Runtime. Rebuild against `onnxruntime-linux-x64-gpu-...` (§16.4). |
| `--provider=cuda` throws `failed to register CUDA ... VERSION MISMATCH` | CUDA/cuDNN don't match the ORT GPU build. Install the pinned CUDA 11.8 + cuDNN 8.9.2 for ORT 1.17.1 (§16.2–16.3). |
| GPU "works" but no speedup / suspect silent CPU fallback | Verify with the checks in §16.5 (startup log line + `nvidia-smi` VRAM). This build fails loudly rather than falling back, so a mismatch shows as an error, not silent CPU use. |
| Benchmark throughput is **flat** across batch sizes | **Expected** for a compute-bound model on CPU (§6, §15.3) — not a bug. It *would* be a bug if the server ignored `--max-batch`; confirm each run's server log shows the right `max_batch`. On the DummyEngine or a GPU you should see it rise. |
| WSL2 disk / C: drive filling up during CUDA install | The WSL `ext4.vhdx` grows on C: and doesn't auto-shrink. Delete unneeded files, then from Windows PowerShell `wsl --shutdown` and `wsl --manage <Distro> --set-sparse true` (or `diskpart` → `compact vdisk`). Prefer the runtime-only CUDA install (§16.3). |
| Server won't start: address already in use | A previous server is still bound to the port. Kill it (`pkill -f inference_server`) or use a different `--address` / `BASE_PORT`. |
| `libonnxruntime.so: cannot open shared object file` at runtime | RPATH should handle this; if you moved the ORT folder, rebuild, or set `LD_LIBRARY_PATH=$ONNXRUNTIME_ROOT/lib`. |

---

## 18. Design Decisions & Tradeoffs

- **Why gRPC?** It gives a production-grade, multi-threaded server and typed
  client from one `.proto`, so effort goes into the batching logic, not socket
  code. A unary RPC matches the request/response shape exactly.
- **Why ONNX Runtime?** It runs a trained model with no training framework, reads
  the portable `.onnx` format, and exposes CPU and GPU behind one API — flipping
  the execution provider is the *only* change to go from CPU to GPU.
- **Why a DummyEngine at all?** To build and benchmark the whole serving system
  with zero ML dependencies, and to demonstrate the fixed-cost-per-batch regime
  where batching wins big (see §6). It's both a development tool and a teaching
  device.
- **Why a mutex-guarded queue (not lock-free)?** The critical section is tiny
  (push/pop a pointer-sized `Request`) and the batcher is a single consumer, so
  lock contention is negligible next to a ~20 ms (dummy) or tens-of-ms (ResNet)
  model call. A lock-free queue only matters when lock overhead is comparable to
  the work per item — not the case here. Simpler and correct wins.
- **Why STATIC batching?** This server forms a batch, runs it to completion, then
  forms the next — simple and ideal for fixed-shape vision models like ResNet-50.
  **LLM serving needs *continuous* batching instead:** token-by-token generation
  of varying lengths means you want to add/remove sequences from an in-flight
  batch every step and maintain a per-sequence **KV-cache**. That's a
  substantially larger system and a natural sequel (see §20).

---

## 19. Limitations & Honest Caveats

- **CPU gains are modest-to-flat.** ResNet-50 on a CPU is compute-bound, so
  batching does not raise throughput there. That's a real, demonstrated finding —
  the batching *throughput* win needs an accelerator (GPU) or the fixed-cost
  regime the DummyEngine models.
- **Static batching only.** No continuous batching, no KV-cache — so this is not
  suitable for LLM/token-generation serving.
- **Single model, single node.** One model per server instance; no model
  management, versioning, or horizontal scaling.
- **Educational scope.** No auth, TLS, rate limiting, retries, metrics endpoint,
  or health checks. Random model weights (accuracy is out of scope; throughput is
  the point).
- **Numbers are machine-specific.** All results come from real runs on the
  machine that produced them; re-run the sweeps on your hardware for your own
  numbers.

---

## 20. Possible Extensions

- **GPU inference** — already supported (§16); run the sweep on real hardware.
- **Live metrics endpoint** — expose throughput/latency/queue-depth (e.g.
  Prometheus) for observability.
- **Multi-model serving** — host several models, route by name in the RPC.
- **Docker image** — package server + ONNX Runtime for reproducible deployment.
- **Continuous batching + KV-cache (the LLM sequel)** — the big one: token-level
  scheduling for autoregressive models, the technique behind real LLM servers
  like vLLM/TGI.

---

## 21. License

Inspired by the concepts behind **NVIDIA Triton Inference
Server** and **TensorFlow Serving**, rebuilt from scratch at small scale for
learning.
# AI-Inference-Server
