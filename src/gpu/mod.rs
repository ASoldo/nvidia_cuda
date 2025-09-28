use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use cudarc::driver::{CudaContext, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use once_cell::sync::Lazy;
use uuid::Uuid;

#[inline(always)]
fn nvtx_scoped<S, F, T>(label: S, f: F) -> T
where
    S: Into<String>,
    F: FnOnce() -> T,
{
    let label = label.into();
    let _guard = nvtx::range!("{}", &label);
    f()
}

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
    let _g = nvtx::range!("cuda init");
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

pub fn ensure_ready() -> Result<&'static GpuState> {
    GPU.as_ref().map_err(|e| anyhow!("{e:#}"))
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

    let mut dx: CudaSlice<f32> = stream.alloc_zeros(n).context("alloc device x")?;
    let mut dy: CudaSlice<f32> = stream.alloc_zeros(n).context("alloc device y")?;
    let mut dout: CudaSlice<f32> = stream.alloc_zeros(n).context("alloc device out")?;

    let trace_id = Uuid::new_v4();

    nvtx_scoped(format!("H2D copies {}", trace_id), || -> Result<()> {
        stream.memcpy_htod(x, &mut dx).context("copy x to device")?;
        stream.memcpy_htod(y, &mut dy).context("copy y to device")?;
        Ok(())
    })?;

    nvtx_scoped(format!("saxpy kernel {}", trace_id), || -> Result<()> {
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
    nvtx_scoped(format!("D2H copy {}", trace_id), || -> Result<()> {
        stream
            .memcpy_dtoh(&dout, &mut out)
            .context("copy out to host")?;
        Ok(())
    })?;

    stream.synchronize().context("synchronize CUDA stream")?;

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saxpy_matches_cpu_reference() {
        let state = match ensure_ready() {
            Ok(state) => state,
            Err(_) => return, // Skip when CUDA hardware isn't accessible in CI.
        };

        let a = 2.0_f32;
        let x = vec![0.0_f32, 1.0, 2.0, 3.0];
        let y = vec![0.5_f32; x.len()];

        let gpu = saxpy_with_state(state, a, &x, &y).expect("GPU saxpy");
        let cpu: Vec<f32> = x.iter().zip(&y).map(|(&xi, &yi)| a * xi + yi).collect();

        assert_eq!(gpu, cpu);
    }
}
