use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use cudarc::driver::{CudaContext, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use nvtx::{range_pop, range_push};
use once_cell::sync::Lazy;

const KERNEL: &str = r#"
extern "C" __global__
void saxpy(float a, const float* __restrict__ x,
           const float* __restrict__ y,
           float* __restrict__ out, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a * x[i] + y[i];
    }
}
"#;

pub struct GpuState {
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
    func_name: &'static str,
}

static GPU: Lazy<Result<GpuState>> = Lazy::new(|| {
    let ctx = CudaContext::new(0)?;

    let ptx = compile_ptx(KERNEL)?;
    let module = ctx.load_module(ptx)?;
    let _ = module.load_function("saxpy")?;

    Ok(GpuState {
        ctx,
        module,
        func_name: "saxpy",
    })
});

macro_rules! nvtx_scope {
    ($label:literal, $body:expr) => {{
        range_push!($label);
        let result = ($body)();
        range_pop!();
        result
    }};
}

pub fn ensure_ready() -> Result<&'static GpuState> {
    gpu_state()
}

pub fn saxpy(a: f32, x: &[f32], y: &[f32]) -> Result<Vec<f32>> {
    let state = ensure_ready()?;
    saxpy_with_state(state, a, x, y)
}

pub fn saxpy_with_state(state: &GpuState, a: f32, x: &[f32], y: &[f32]) -> Result<Vec<f32>> {
    if x.len() != y.len() {
        return Err(anyhow!("x and y must be same length"));
    }

    let n = x.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let stream = state.ctx.new_stream().context("create CUDA stream")?;
    let func = state
        .module
        .load_function(state.func_name)
        .context("load saxpy kernel")?;

    let mut dx: CudaSlice<f32> = stream
        .alloc_zeros(n)
        .context("allocate device buffer for x")?;
    let mut dy: CudaSlice<f32> = stream
        .alloc_zeros(n)
        .context("allocate device buffer for y")?;
    let mut dout: CudaSlice<f32> = stream
        .alloc_zeros(n)
        .context("allocate device buffer for output")?;

    nvtx_scope!("H2D copies", || -> Result<()> {
        stream.memcpy_htod(x, &mut dx).context("copy x to device")?;
        stream.memcpy_htod(y, &mut dy).context("copy y to device")?;
        Ok(())
    })?;

    nvtx_scope!("saxpy kernel", || -> Result<()> {
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&a)
                .arg(&dx)
                .arg(&dy)
                .arg(&mut dout)
                .arg(&n)
                .launch(cfg)
                .context("launch saxpy kernel")?;
        }
        Ok(())
    })?;

    let mut out = vec![0f32; n];
    nvtx_scope!("D2H copy", || -> Result<()> {
        stream
            .memcpy_dtoh(&dout, &mut out)
            .context("copy output to host")?;
        Ok(())
    })?;

    stream.synchronize().context("synchronize CUDA stream")?;

    Ok(out)
}

fn gpu_state() -> Result<&'static GpuState> {
    GPU.as_ref().map_err(|e| anyhow!("{e:#}"))
}
