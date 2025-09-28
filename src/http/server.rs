use actix_web::web;

use super::handlers::{healthz, saxpy_api};

const JSON_LIMIT_BYTES: usize = 8 * 1024 * 1024;

pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const DEFAULT_PORT: u16 = 8080;

pub fn json_config() -> web::JsonConfig {
    web::JsonConfig::default().limit(JSON_LIMIT_BYTES)
}

pub fn configure_services(cfg: &mut web::ServiceConfig) {
    cfg.service(healthz).service(saxpy_api);
}
