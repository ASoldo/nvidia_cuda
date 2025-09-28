use anyhow::Result;
use cudarc::driver::PushKernelArg;
use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use nvtx::{range_pop, range_push};

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

fn main() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.new_stream()?;

    let ptx = compile_ptx(KERNEL)?;
    let module = ctx.load_module(ptx)?;
    let func = module.load_function("saxpy")?;

    let n: usize = 1_000_000;
    let a: f32 = 2.0;
    let host_x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let host_y: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();

    let mut dev_x: CudaSlice<f32> = stream.alloc_zeros(n)?;
    let mut dev_y: CudaSlice<f32> = stream.alloc_zeros(n)?;
    let mut dev_out: CudaSlice<f32> = stream.alloc_zeros(n)?;
    range_push!("H2D copies");
    stream.memcpy_htod(&host_x, &mut dev_x)?;
    stream.memcpy_htod(&host_y, &mut dev_y)?;
    range_pop!();

    let cfg = LaunchConfig::for_num_elems(n as u32);
    range_push!("saxpy kernel");
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&a)
            .arg(&dev_x)
            .arg(&dev_y)
            .arg(&mut dev_out)
            .arg(&n)
            .launch(cfg)?;
    }
    range_pop!();

    let mut out = vec![0f32; n];

    range_push!("D2H copy");
    stream.memcpy_dtoh(&dev_out, &mut out)?;
    range_pop!();

    stream.synchronize()?;

    for &i in &[0usize, 1, 12345, n - 1] {
        let expect = a * (i as f32) + (i as f32) * 0.5;
        let got = out[i];
        let diff = (got - expect).abs();
        assert!(diff < 1e-5, "mismatch at {i}: got {got}, expected {expect}");
    }

    println!(
        "SAXPY complete on {} elements. Example out[12345] = {}",
        n, out[12345]
    );
    Ok(())
}
