set shell := ["bash", "-lc"]

host := "127.0.0.1"
port := "8080"

# Default target prints available recipes.
default:
    @just --list

# Build the project in debug mode.
build: fmt check test
    cargo build

# Build the project in release mode.
build-release: fmt check test
    cargo build --release

# Run the service with optional host/port overrides.
run: fmt check test
    NVIDIA_CUDA_HOST={{host}} NVIDIA_CUDA_PORT={{port}} cargo run

# Run the service in release mode.
run-release: fmt check test
    NVIDIA_CUDA_HOST={{host}} NVIDIA_CUDA_PORT={{port}} cargo run --release

# Execute the test suite (skips silently if CUDA unavailable).
test:
    cargo test

# Fast type-check without running tests.
check:
    cargo check

# Format the Rust sources.
fmt:
    cargo fmt

# Profile the release binary with Nsight Systems (nsys must be in PATH).
profile-nsys: build-release
    nsys profile -o saxpy-http --trace=cuda,nvtx,osrt ./target/release/nvidia_cuda

# Profile with Nsight Compute (requires sudo + ncu).
profile-ncu: build-release
    sudo ncu --target-processes all --set full ./target/release/nvidia_cuda
