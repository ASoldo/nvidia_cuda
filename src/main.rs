use std::io::{Error as IoError, ErrorKind};

use actix_web::{App, HttpServer};

use nvidia_cuda::{gpu, grpc, http::server};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    if let Err(e) = gpu::ensure_ready() {
        eprintln!("GPU initialization failed: {e:#}");
        std::process::exit(1);
    }

    let (host, port) = server::bind_address();
    println!("HTTP listening on {host}:{port}");

    let grpc_addr = match grpc::bind_address() {
        Ok(addr) => addr,
        Err(e) => {
            eprintln!("Invalid gRPC bind address: {e:#}");
            std::process::exit(1);
        }
    };
    println!("gRPC listening on {grpc_addr}");

    let http_future = HttpServer::new(|| {
        App::new()
            .app_data(server::json_config())
            .configure(server::configure_services)
    })
    .bind((host.as_str(), port))?
    .run();

    let grpc_future = async move {
        grpc::serve(grpc_addr)
            .await
            .map_err(|e| IoError::new(ErrorKind::Other, e))
    };

    match tokio::try_join!(http_future, grpc_future) {
        Ok(_) => Ok(()),
        Err(e) if e.kind() == ErrorKind::Interrupted => Ok(()),
        Err(e) => Err(e),
    }
}
