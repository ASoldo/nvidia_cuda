use actix_web::{App, HttpServer};

use nvidia_cuda::{gpu, http::server};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    if let Err(e) = gpu::ensure_ready() {
        eprintln!("GPU initialization failed: {e:#}");
        std::process::exit(1);
    }

    let (host, port) = server::bind_address();
    println!("Listening on {host}:{port}");

    HttpServer::new(|| {
        App::new()
            .app_data(server::json_config())
            .configure(server::configure_services)
    })
    .bind((host.as_str(), port))?
    .run()
    .await
}
