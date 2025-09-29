use actix_web::{App, HttpServer};
use nvidia_cuda::{gpu, grpc, http::server};
use std::io::{Error as IoError, ErrorKind};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    if let Err(e) = gpu::ensure_ready() {
        eprintln!("GPU initialization failed: {e:#}");
        std::process::exit(1);
    }

    let (host, port) = server::bind_address();
    let grpc_addr = grpc::bind_address().map_err(|e| {
        eprintln!("Invalid gRPC bind address: {e:#}");
        IoError::new(ErrorKind::Other, e)
    })?;

    println!("HTTP listening on {host}:{port}");
    println!("gRPC listening on {grpc_addr}");

    let http_srv = HttpServer::new(|| {
        App::new()
            .app_data(server::json_config())
            .configure(server::configure_services)
    })
    .bind((host.as_str(), port))?
    .shutdown_timeout(5)
    .run();
    let http_handle = http_srv.handle();

    let grpc_task = tokio::spawn(async move {
        grpc::serve(grpc_addr)
            .await
            .map_err(|e| IoError::new(ErrorKind::Other, e))
    });

    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            eprintln!("Ctrl+C received, shutting down...");
        }
        res = http_srv => {
            if let Err(e) = res {
                eprintln!("HTTP server error: {e:#}");
                return Err(e);
            }
        }
    }

    http_handle.stop(true).await;
    grpc_task.abort();

    let _ = grpc_task.await;

    Ok(())
}
