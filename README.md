# NVIDIA CUDA Rust Microservice

This project exposes a CUDA-accelerated SAXPY (`out[i] = a * x[i] + y[i]`) kernel as an HTTP microservice built with Actix Web. The GPU work is handled through [`cudarc`](https://crates.io/crates/cudarc) and optional NVTX annotations power profiling with NVIDIA Nsight tools.

---

## Requirements

- NVIDIA GPU with a compatible CUDA driver (tested with compute capability ≥ 8.6)
- Rust toolchain via [rustup](https://rustup.rs/)
- CUDA Toolkit 13.0.1 (provides `nvrtc` and driver libraries)
- Optional: Nsight Systems 2025.3.2 and Nsight Compute 2025.3.1 for profiling

> **Note**: The examples below assume an Arch Linux environment, but any Linux distribution with matching CUDA/NVIDIA versions should work.

---

## Environment Setup (Arch Linux example)

Arch does not bundle Nsight tooling. If you need `nsys`/`ncu`, download the Debian installer and extract it locally:

```bash
sudo pacman -S dpkg
wget https://developer.download.nvidia.com/compute/cuda/13.0.1/local_installers/cuda-repo-debian12-13-0-local_13.0.1-580.82.07-1_amd64.deb
dpkg-deb -x cuda-repo-debian12-13-0-local_13.0.1-580.82.07-1_amd64.deb extracted/
sudo cp -r extracted/opt/* /opt/
export PATH=/opt/nsight-systems/2025.3.2/target-linux-x64:$PATH
export PATH=/opt/nsight-compute/2025.3.1:$PATH
```

Persist the `PATH` exports in your shell profile if you rely on them frequently.

---

## Build & Run

```bash
git clone https://github.com/asoldo/nvidia_cuda
cd nvidia_cuda
cargo run --release
```

The service starts on `127.0.0.1:8080` by default. You can exercise the endpoints with `curl`:

```bash
# Health probe
curl http://127.0.0.1:8080/healthz

# SAXPY request (arrays must share length)
curl -s -X POST http://127.0.0.1:8080/saxpy \
  -H 'Content-Type: application/json' \
  -d '{"a":2.0,"x":[0,1,2,3,4],"y":[0.5,0.5,0.5,0.5,0.5]}' \
  http://127.0.0.1:8080/saxpy
```

Sample response:

```json
[0.5,2.5,4.5,6.5,8.5]
```

---

## API Surface

| Method | Path      | Description                       |
| ------ | --------- | --------------------------------- |
| GET    | `/healthz`| Reports GPU initialization status |
| POST   | `/saxpy`  | Runs SAXPY on supplied payload    |

### `/saxpy` payload schema

```json
{
  "a": 2.0,
  "x": [0.0, 1.0, 2.0, 3.0, 4.0],
  "y": [0.5, 0.5, 0.5, 0.5, 0.5]
}
```

- `a`: scalar multiplier
- `x`, `y`: equal-length float arrays (up to the configured JSON size limit of 8 MiB)
- Response: array of `f32` values representing the SAXPY result

Error conditions return standard HTTP status codes (`400` for length mismatches, `503` if GPU init fails, `500` for runtime errors).

---

## Profiling (optional)

NVTX ranges wrap host-to-device copies, kernel execution, and device-to-host copies. That makes Nsight timelines easy to interpret.

```bash
# Capture with Nsight Systems
nsys profile -o saxpy-nvtx cargo run --release
nsys stats saxpy-nvtx.nsys-rep

# Inspect kernel metrics with Nsight Compute
ncu cargo run --release
```

Each command launches the microservice; invoke the `/saxpy` endpoint from another terminal while profiling is active.

---

## Project Layout

```
src/
  gpu/          // CUDA initialization, NVTX helpers, SAXPY execution
  http/
    handlers.rs // Actix handlers
    models.rs   // Request data structures
    server.rs   // Actix configuration helpers
  lib.rs        // Library exports
  main.rs       // Binary entry point wiring HTTP + GPU layers
```

---

## Notes

- `cudarc` handles runtime NVRTC compilation and kernel launch ergonomics.
- `nvtx` annotations are optional but recommended whenever profiling with Nsight.
- JSON payloads are limited to 8 MiB; adjust `JSON_LIMIT_BYTES` in `src/http/server.rs` if needed.
