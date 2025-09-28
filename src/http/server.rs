use std::env;

use actix_web::web;

use super::handlers::{healthz, saxpy_api};

const JSON_LIMIT_BYTES: usize = 8 * 1024 * 1024;

pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const DEFAULT_PORT: u16 = 8080;
const HOST_ENV: &str = "NVIDIA_CUDA_HOST";
const PORT_ENV: &str = "NVIDIA_CUDA_PORT";

pub fn json_config() -> web::JsonConfig {
    web::JsonConfig::default().limit(JSON_LIMIT_BYTES)
}

pub fn configure_services(cfg: &mut web::ServiceConfig) {
    cfg.service(healthz).service(saxpy_api);
}

pub fn bind_address() -> (String, u16) {
    let host = env::var(HOST_ENV).unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let port = env::var(PORT_ENV)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PORT);
    (host, port)
}
