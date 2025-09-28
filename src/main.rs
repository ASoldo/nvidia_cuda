use actix_web::{App, HttpResponse, HttpServer, Responder, get, post, web};
use anyhow::Result;
use nvtx::{range_pop, range_push};
use once_cell::sync::Lazy;
use serde::Deserialize;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

// -------- CUDA kernel (same as your demo) --------
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

// -------- App state: initialized once and reused --------
struct GpuState {
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
    // cache the function handle too
    func_name: &'static str,
}

static GPU: Lazy<Result<GpuState>> = Lazy::new(|| {
    // 1) Init device/context
    let ctx = CudaContext::new(0)?;

    // 2) Compile PTX and load module once
    let ptx = compile_ptx(KERNEL)?;
    let module = ctx.load_module(ptx)?;
    // probe the symbol now so we fail early if missing
    let _ = module.load_function("saxpy")?;

    Ok(GpuState {
        ctx,
        module,
        func_name: "saxpy",
    })
});

// -------- Payloads --------
#[derive(Deserialize)]
struct SaxpyInput {
    a: f32,
    x: Vec<f32>,
    y: Vec<f32>,
}

// -------- Handlers --------
#[post("/saxpy")]
async fn saxpy_api(body: web::Json<SaxpyInput>) -> impl Responder {
    let state = match &*GPU {
        Ok(s) => s,
        Err(e) => {
            return HttpResponse::ServiceUnavailable().body(format!("GPU init failed: {e:#}"));
        }
    };

    if body.x.len() != body.y.len() {
        return HttpResponse::BadRequest().body("x and y must be same length");
    }
    let n = body.x.len();
    if n == 0 {
        return HttpResponse::Ok().json(Vec::<f32>::new());
    }

    // Per-request CUDA stream
    let stream = match state.ctx.new_stream() {
        Ok(s) => s,
        Err(e) => return HttpResponse::InternalServerError().body(format!("stream: {e:#}")),
    };

    // Kernel handle from preloaded module
    let func = match state.module.load_function(state.func_name) {
        Ok(f) => f,
        Err(e) => return HttpResponse::InternalServerError().body(format!("function: {e:#}")),
    };

    // Device buffers
    let mut dx: CudaSlice<f32> = match stream.alloc_zeros(n) {
        Ok(b) => b,
        Err(e) => return HttpResponse::InternalServerError().body(format!("alloc dx: {e:#}")),
    };
    let mut dy: CudaSlice<f32> = match stream.alloc_zeros(n) {
        Ok(b) => b,
        Err(e) => return HttpResponse::InternalServerError().body(format!("alloc dy: {e:#}")),
    };
    let mut dout: CudaSlice<f32> = match stream.alloc_zeros(n) {
        Ok(b) => b,
        Err(e) => return HttpResponse::InternalServerError().body(format!("alloc dout: {e:#}")),
    };

    // H2D copies
    range_push!("H2D copies");
    if let Err(e) = stream.memcpy_htod(&body.x, &mut dx) {
        range_pop!();
        return HttpResponse::InternalServerError().body(format!("htod x: {e:#}"));
    }
    if let Err(e) = stream.memcpy_htod(&body.y, &mut dy) {
        range_pop!();
        return HttpResponse::InternalServerError().body(format!("htod y: {e:#}"));
    }
    range_pop!();

    // Kernel launch
    let cfg = LaunchConfig::for_num_elems(n as u32);
    range_push!("saxpy kernel");
    unsafe {
        if let Err(e) = stream
            .launch_builder(&func)
            .arg(&body.a)
            .arg(&dx)
            .arg(&dy)
            .arg(&mut dout)
            .arg(&n)
            .launch(cfg)
        {
            range_pop!();
            return HttpResponse::InternalServerError().body(format!("launch: {e:#}"));
        }
    }
    range_pop!();

    // D2H copy
    let mut out = vec![0f32; n];
    range_push!("D2H copy");
    if let Err(e) = stream.memcpy_dtoh(&dout, &mut out) {
        range_pop!();
        return HttpResponse::InternalServerError().body(format!("dtoh: {e:#}"));
    }
    range_pop!();

    if let Err(e) = stream.synchronize() {
        return HttpResponse::InternalServerError().body(format!("sync: {e:#}"));
    }

    HttpResponse::Ok().json(out)
}

#[get("/healthz")]
async fn healthz() -> impl Responder {
    match &*GPU {
        Ok(_) => HttpResponse::Ok().body("ok"),
        Err(e) => HttpResponse::ServiceUnavailable().body(format!("gpu init failed: {e:#}")),
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Fail fast on bad CUDA init
    if let Err(e) = &*GPU {
        eprintln!("GPU initialization failed: {e:#}");
        std::process::exit(1);
    }

    HttpServer::new(|| {
        App::new()
            .app_data(web::JsonConfig::default().limit(8 * 1024 * 1024))
            .service(healthz)
            .service(saxpy_api)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
