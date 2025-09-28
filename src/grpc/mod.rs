use std::{env, net::SocketAddr};

use anyhow::{Context, Result};
use tonic::{Request, Response, Status, transport::Server};

use crate::gpu;

pub mod proto {
    tonic::include_proto!("saxpy.v1");
}

use proto::saxpy_service_server::{SaxpyService, SaxpyServiceServer};
use proto::{SaxpyRequest, SaxpyResponse};

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 50051;
const HOST_ENV: &str = "NVIDIA_CUDA_GRPC_HOST";
const PORT_ENV: &str = "NVIDIA_CUDA_GRPC_PORT";

pub fn bind_address() -> Result<SocketAddr> {
    let host = env::var(HOST_ENV).unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let port = env::var(PORT_ENV)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PORT);
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .context("parse gRPC bind address")?;
    Ok(addr)
}

pub async fn serve(addr: SocketAddr) -> Result<()> {
    let service = SaxpyServiceServer::new(SaxpyGrpc);
    Server::builder()
        .add_service(service)
        .serve(addr)
        .await
        .context("run gRPC server")
}

#[derive(Debug, Default)]
struct SaxpyGrpc;

#[tonic::async_trait]
impl SaxpyService for SaxpyGrpc {
    async fn compute(
        &self,
        request: Request<SaxpyRequest>,
    ) -> Result<Response<SaxpyResponse>, Status> {
        let payload = request.into_inner();
        if payload.x.len() != payload.y.len() {
            return Err(Status::invalid_argument("x and y must be same length"));
        }

        let state = gpu::ensure_ready()
            .map_err(|e| Status::unavailable(format!("GPU init failed: {e:#}")))?;

        let result = gpu::saxpy_with_state(state, payload.a, &payload.x, &payload.y)
            .map_err(|e| Status::internal(format!("saxpy failed: {e:#}")))?;

        Ok(Response::new(SaxpyResponse { out: result }))
    }
}
