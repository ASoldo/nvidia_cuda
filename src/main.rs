use actix_web::{App, HttpServer, rt};

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

    let grpc_handle = rt::spawn(async move {
        if let Err(e) = grpc::serve(grpc_addr).await {
            eprintln!("gRPC server exited: {e:#}");
        }
    });

    let server = HttpServer::new(|| {
        App::new()
            .app_data(server::json_config())
            .configure(server::configure_services)
    })
    .bind((host.as_str(), port))?
    .run();

    match server.await {
        Ok(()) => {
            let _ = grpc_handle.abort();
            Ok(())
        }
        Err(e) => {
            let _ = grpc_handle.abort();
            Err(e)
        }
    }
}
