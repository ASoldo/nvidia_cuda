use actix_web::{App, HttpServer};
use nvidia_cuda::{gpu, grpc, http::server};
use std::io::{Error as IoError, ErrorKind};
use tokio::{net::TcpListener, sync::oneshot};

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

    let http_server = HttpServer::new(|| {
        App::new()
            .app_data(server::json_config())
            .configure(server::configure_services)
    })
    .bind((host.as_str(), port))?
    .shutdown_timeout(5)
    .run();

    let http_handle = http_server.handle();
    let (http_done_tx, http_done_rx) = oneshot::channel();

    tokio::spawn(async move {
        let result = http_server.await;
        let _ = http_done_tx.send(result);
    });

    let grpc_listener = TcpListener::bind(grpc_addr).await.map_err(|e| {
        eprintln!("Failed to bind gRPC address {grpc_addr}: {e:#}");
        IoError::new(ErrorKind::Other, e)
    })?;
    let grpc_local_addr = grpc_listener
        .local_addr()
        .map_err(|e| IoError::new(ErrorKind::Other, e))?;

    println!("HTTP listening on {host}:{port}");
    println!("gRPC listening on {grpc_local_addr}");

    let (grpc_shutdown_tx, grpc_shutdown_rx) = oneshot::channel::<()>();
    let (grpc_done_tx, grpc_done_rx) = oneshot::channel();

    tokio::spawn(async move {
        let shutdown = async {
            let _ = grpc_shutdown_rx.await;
        };
        let result = grpc::serve(grpc_listener, shutdown)
            .await
            .map_err(|e| IoError::new(ErrorKind::Other, e));
        let _ = grpc_done_tx.send(result);
    });

    let mut http_done_rx = http_done_rx;
    let mut grpc_done_rx = grpc_done_rx;

    enum Event {
        CtrlC,
        Http(std::io::Result<()>),
        Grpc(std::io::Result<()>),
    }

    let event = tokio::select! {
        _ = tokio::signal::ctrl_c() => Event::CtrlC,
        res = &mut http_done_rx => {
            let res = res.unwrap_or_else(|_| Err(IoError::new(ErrorKind::Other, "HTTP server task dropped before completion")));
            Event::Http(res)
        }
        res = &mut grpc_done_rx => {
            let res = res.unwrap_or_else(|_| Err(IoError::new(ErrorKind::Other, "gRPC server task dropped before completion")));
            Event::Grpc(res)
        }
    };

    let mut http_result: Option<std::io::Result<()>> = None;
    let mut grpc_result: Option<std::io::Result<()>> = None;
    let mut grpc_shutdown_tx = Some(grpc_shutdown_tx);

    match event {
        Event::CtrlC => {
            eprintln!("Ctrl+C received, shutting down...");
            let _ = http_handle.stop(true).await;
            if let Some(tx) = grpc_shutdown_tx.take() {
                let _ = tx.send(());
            }
        }
        Event::Http(res) => {
            http_result = Some(res);
            if let Some(tx) = grpc_shutdown_tx.take() {
                let _ = tx.send(());
            }
        }
        Event::Grpc(res) => {
            grpc_result = Some(res);
            let _ = http_handle.stop(true).await;
        }
    }

    if http_result.is_none() {
        http_result = Some((&mut http_done_rx).await.unwrap_or_else(|_| {
            Err(IoError::new(
                ErrorKind::Other,
                "HTTP server task dropped before completion",
            ))
        }));
    }

    if grpc_result.is_none() {
        grpc_result = Some((&mut grpc_done_rx).await.unwrap_or_else(|_| {
            Err(IoError::new(
                ErrorKind::Other,
                "gRPC server task dropped before completion",
            ))
        }));
    }

    if let Some(Err(e)) = http_result {
        eprintln!("HTTP server error: {e:#}");
        return Err(e);
    }

    if let Some(Err(e)) = grpc_result {
        eprintln!("gRPC server error: {e:#}");
        return Err(e);
    }

    Ok(())
}
