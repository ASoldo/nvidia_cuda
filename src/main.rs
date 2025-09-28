use actix_web::{App, HttpServer};

use nvidia_cuda::{gpu, http::server};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    if let Err(e) = gpu::ensure_ready() {
        eprintln!("GPU initialization failed: {e:#}");
        std::process::exit(1);
    }

    HttpServer::new(|| {
        App::new()
            .app_data(server::json_config())
            .configure(server::configure_services)
    })
    .bind((server::DEFAULT_HOST, server::DEFAULT_PORT))?
    .run()
    .await
}
