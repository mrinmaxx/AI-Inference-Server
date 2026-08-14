cmake_minimum_required(VERSION 3.16)
project(inference_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ----- Dependencies -----
#
# Distro packages (Debian/Ubuntu apt) ship gRPC/protobuf as pkg-config modules
# rather than CMake "config packages", so find_package(gRPC CONFIG) does NOT
# work there. We therefore discover them via pkg-config, which IS present. If
# you built gRPC from source or use vcpkg (which DO ship CMake configs), set
# -DUSE_CMAKE_CONFIG=ON to take the config-package path instead.
find_package(Threads REQUIRED)

option(USE_CMAKE_CONFIG "Find gRPC/protobuf via CMake config packages (source/vcpkg builds)" OFF)

if(USE_CMAKE_CONFIG)
  find_package(Protobuf CONFIG REQUIRED)   # provides protobuf::libprotobuf, protobuf::protoc
  find_package(gRPC CONFIG REQUIRED)       # provides gRPC::grpc++, gRPC::grpc_cpp_plugin
  message(STATUS "Using protobuf ${Protobuf_VERSION} (CMake config)")
  message(STATUS "Using gRPC ${gRPC_VERSION} (CMake config)")
  set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
  set(_GRPC_GRPCPP          gRPC::grpc++)
  set(_PROTOBUF_PROTOC      $<TARGET_FILE:protobuf::protoc>)
  set(_GRPC_CPP_PLUGIN      $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
else()
  find_package(PkgConfig REQUIRED)
  # IMPORTED_TARGET creates PkgConfig::PROTOBUF / PkgConfig::GRPCPP with all the
  # include dirs and link flags baked in.
  pkg_check_modules(PROTOBUF REQUIRED IMPORTED_TARGET protobuf)
  pkg_check_modules(GRPCPP   REQUIRED IMPORTED_TARGET grpc++)
  # The code generators are plain executables on PATH (installed by
  # protobuf-compiler and protobuf-compiler-grpc).
  find_program(_PROTOBUF_PROTOC protoc REQUIRED)
  find_program(_GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)
  message(STATUS "Using protobuf ${PROTOBUF_VERSION} (pkg-config)")
  message(STATUS "Using gRPC ${GRPCPP_VERSION} (pkg-config)")
  message(STATUS "protoc            = ${_PROTOBUF_PROTOC}")
  message(STATUS "grpc_cpp_plugin   = ${_GRPC_CPP_PLUGIN}")
  set(_PROTOBUF_LIBPROTOBUF PkgConfig::PROTOBUF)
  set(_GRPC_GRPCPP          PkgConfig::GRPCPP)
endif()

# ----- ONNX Runtime (Phase 3, the OnnxEngine) -----
#
# ONNX Runtime's prebuilt C++ package is NOT an apt/CMake-config package: you
# download a tarball from GitHub releases, extract it, and point CMake at it:
#
#   ONNX_VER=1.17.1
#   wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VER}/onnxruntime-linux-x64-${ONNX_VER}.tgz
#   tar xzf onnxruntime-linux-x64-${ONNX_VER}.tgz
#   cmake -S . -B build -DONNXRUNTIME_ROOT=$PWD/onnxruntime-linux-x64-${ONNX_VER}
#
# ONNXRUNTIME_ROOT is the directory that contains include/ and lib/. Build the
# dummy-only server (no ONNX Runtime needed) with -DWITH_ONNX=OFF.
option(WITH_ONNX "Build the OnnxEngine (requires ONNX Runtime C++)" ON)

if(WITH_ONNX)
  set(ONNXRUNTIME_ROOT "" CACHE PATH
      "Root of an extracted ONNX Runtime prebuilt (contains include/ and lib/)")
  find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}" ENV ONNXRUNTIME_ROOT
    PATH_SUFFIXES include
    PATHS /usr/local/include /usr/include)
  find_library(ORT_LIBRARY onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}" ENV ONNXRUNTIME_ROOT
    PATH_SUFFIXES lib lib64
    PATHS /usr/local/lib /usr/lib)
  if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
    message(FATAL_ERROR
      "WITH_ONNX=ON but ONNX Runtime was not found.\n"
      "  Pass -DONNXRUNTIME_ROOT=<dir> (the folder with include/ and lib/),\n"
      "  or build the dummy-only server with -DWITH_ONNX=OFF.\n"
      "  (ORT_INCLUDE_DIR=${ORT_INCLUDE_DIR}, ORT_LIBRARY=${ORT_LIBRARY})")
  endif()
  get_filename_component(ORT_LIB_DIR "${ORT_LIBRARY}" DIRECTORY)
  message(STATUS "ONNX Runtime headers : ${ORT_INCLUDE_DIR}")
  message(STATUS "ONNX Runtime library : ${ORT_LIBRARY}")
  # The GPU package ships extra provider libs next to libonnxruntime.so
  # (libonnxruntime_providers_cuda.so / _shared.so). We do NOT link them
  # directly -- libonnxruntime.so dlopen()s them at runtime from its own
  # directory, which the RPATH below (ORT_LIB_DIR) makes discoverable. Detect &
  # report which build is linked so a CPU/GPU mixup is obvious at configure time.
  if(EXISTS "${ORT_LIB_DIR}/libonnxruntime_providers_cuda.so")
    message(STATUS "ONNX Runtime build   : GPU (CUDAExecutionProvider available)")
  else()
    message(STATUS "ONNX Runtime build   : CPU-only (no CUDA provider present)")
  endif()
endif()

# ----- Generate C++ from the .proto -----
get_filename_component(proto_file "${CMAKE_CURRENT_SOURCE_DIR}/proto/inference.proto" ABSOLUTE)
get_filename_component(proto_path "${proto_file}" DIRECTORY)

set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${generated_dir}")

set(proto_srcs "${generated_dir}/inference.pb.cc")
set(proto_hdrs "${generated_dir}/inference.pb.h")
set(grpc_srcs  "${generated_dir}/inference.grpc.pb.cc")
set(grpc_hdrs  "${generated_dir}/inference.grpc.pb.h")

add_custom_command(
  OUTPUT "${proto_srcs}" "${proto_hdrs}" "${grpc_srcs}" "${grpc_hdrs}"
  COMMAND ${_PROTOBUF_PROTOC}
  ARGS --grpc_out  "${generated_dir}"
       --cpp_out   "${generated_dir}"
       -I "${proto_path}"
       --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN}"
       "${proto_file}"
  DEPENDS "${proto_file}"
  COMMENT "Generating gRPC/protobuf sources from inference.proto")

# Bundle the generated code into one library both executables link against.
add_library(inference_proto STATIC ${proto_srcs} ${grpc_srcs})
target_link_libraries(inference_proto PUBLIC
  ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF})
# PUBLIC so the generated headers are on the include path of anything linking us.
target_include_directories(inference_proto PUBLIC "${generated_dir}")

# ----- Inference server executable -----
add_executable(inference_server
  src/main.cpp
  src/server.cpp
  src/engine.cpp
  src/batcher.cpp
  src/request_queue.cpp)
target_include_directories(inference_server PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(inference_server PRIVATE
  inference_proto ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF} Threads::Threads)

# Compile the real OnnxEngine (engine.cpp is #ifdef WITH_ONNX) and link ONNX
# Runtime. The RPATH bakes libonnxruntime.so's directory into the executable so
# it runs without setting LD_LIBRARY_PATH by hand.
if(WITH_ONNX)
  target_compile_definitions(inference_server PRIVATE WITH_ONNX)
  target_include_directories(inference_server PRIVATE "${ORT_INCLUDE_DIR}")
  target_link_libraries(inference_server PRIVATE "${ORT_LIBRARY}")
  set_target_properties(inference_server PROPERTIES
    BUILD_RPATH "${ORT_LIB_DIR}"
    INSTALL_RPATH "${ORT_LIB_DIR}")
endif()

# ----- Benchmark client executable -----
add_executable(bench_client
  bench/client.cpp)
target_link_libraries(bench_client PRIVATE
  inference_proto ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF} Threads::Threads)
