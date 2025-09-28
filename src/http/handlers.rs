use actix_web::{HttpResponse, Responder, get, post, web};

use crate::gpu;

use super::models::SaxpyInput;

#[get("/healthz")]
pub async fn healthz() -> impl Responder {
    match gpu::ensure_ready() {
        Ok(_) => HttpResponse::Ok().body("ok"),
        Err(e) => HttpResponse::ServiceUnavailable().body(format!("gpu init failed: {e:#}")),
    }
}

#[post("/saxpy")]
pub async fn saxpy_api(body: web::Json<SaxpyInput>) -> impl Responder {
    let state = match gpu::ensure_ready() {
        Ok(state) => state,
        Err(e) => {
            return HttpResponse::ServiceUnavailable().body(format!("GPU init failed: {e:#}"));
        }
    };

    let payload = body.into_inner();
    if payload.x.len() != payload.y.len() {
        return HttpResponse::BadRequest().body("x and y must be same length");
    }

    match gpu::saxpy_with_state(state, payload.a, &payload.x, &payload.y) {
        Ok(output) => HttpResponse::Ok().json(output),
        Err(e) => HttpResponse::InternalServerError().body(format!("saxpy failed: {e:#}")),
    }
}
