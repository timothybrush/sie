use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info;
use utoipa::ToSchema;

use crate::http_error::{code as err_code, json_detail};
use crate::server::AppState;
use crate::state::pool_manager::DEFAULT_POOL_NAME;

#[derive(Debug, Deserialize, ToSchema)]
pub struct CreatePoolRequest {
    pub name: String,
    #[serde(default)]
    pub gpus: HashMap<String, u32>,
    #[serde(default)]
    pub gpu_caps: HashMap<String, u32>,
    #[serde(default)]
    pub bundle: Option<String>,
    #[serde(default)]
    pub ttl_seconds: Option<u64>,
    /// Per-pool warm floor (minimum machines kept warm via KEDA). Default 0
    /// keeps scale-from-zero. See `PoolSpec::minimum_worker_count`.
    #[serde(default)]
    pub minimum_worker_count: u32,
}

#[utoipa::path(
    post,
    path = "/v1/pools",
    tag = "pools",
    request_body = CreatePoolRequest,
    responses(
        (status = 201, description = "Pool created, renewed, or updated", body = crate::types::pool::Pool),
        (status = 400, description = "Invalid pool request", body = crate::openapi::StandardApiError)
    )
)]
pub async fn create_pool(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreatePoolRequest>,
) -> impl IntoResponse {
    if req.name.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json_detail(
                err_code::INVALID_REQUEST,
                "Pool name is required",
            )),
        )
            .into_response();
    }

    if req.gpus.is_empty() && req.gpu_caps.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json_detail(
                err_code::INVALID_REQUEST,
                "GPU requirements or caps are required",
            )),
        )
            .into_response();
    }

    match state
        .pool_manager
        .create_pool_with_caps(
            &req.name,
            req.gpus,
            req.gpu_caps,
            req.bundle,
            req.ttl_seconds,
            req.minimum_worker_count,
        )
        .await
    {
        Ok(pool) => {
            info!(event = "pool.create", pool = %req.name, status = 201u16, "audit");
            (StatusCode::CREATED, Json(json!(pool))).into_response()
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(json_detail(err_code::INVALID_REQUEST, e.to_string())),
        )
            .into_response(),
    }
}

#[utoipa::path(
    get,
    path = "/v1/pools",
    tag = "pools",
    responses((status = 200, description = "Pools visible to this gateway replica", body = crate::openapi::PoolListResponse))
)]
pub async fn list_pools(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let pools = state.pool_manager.list_pools().await;
    (StatusCode::OK, Json(json!({"pools": pools})))
}

#[utoipa::path(
    get,
    path = "/v1/pools/{name}",
    tag = "pools",
    params(("name" = String, Path, description = "Pool name")),
    responses(
        (status = 200, description = "Pool detail", body = crate::types::pool::Pool),
        (status = 404, description = "Pool not found", body = crate::openapi::StandardApiError)
    )
)]
pub async fn get_pool(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    match state.pool_manager.get_pool(&name).await {
        Some(pool) => (StatusCode::OK, Json(json!(pool))).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(json_detail(
                err_code::POOL_NOT_FOUND,
                format!("Pool '{}' not found", name),
            )),
        )
            .into_response(),
    }
}

#[utoipa::path(
    delete,
    path = "/v1/pools/{name}",
    tag = "pools",
    params(("name" = String, Path, description = "Pool name")),
    responses(
        (status = 200, description = "Pool deleted", body = crate::openapi::MessageResponse),
        (status = 403, description = "Pool cannot be deleted", body = crate::openapi::StandardApiError),
        (status = 404, description = "Pool not found", body = crate::openapi::StandardApiError)
    )
)]
pub async fn delete_pool(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    if name.eq_ignore_ascii_case(DEFAULT_POOL_NAME) {
        return (
            StatusCode::FORBIDDEN,
            Json(json_detail(
                err_code::DEFAULT_POOL_DELETE_FORBIDDEN,
                "Cannot delete the default pool",
            )),
        )
            .into_response();
    }

    match state.pool_manager.delete_pool(&name).await {
        Ok(true) => {
            info!(event = "pool.delete", pool = %name, status = 200u16, "audit");
            (StatusCode::OK, Json(json!({"message": "Pool deleted"}))).into_response()
        }
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(json_detail(
                err_code::POOL_NOT_FOUND,
                format!("Pool '{}' not found", name),
            )),
        )
            .into_response(),
        Err(e) => (
            StatusCode::FORBIDDEN,
            Json(json_detail(
                err_code::POOL_OPERATION_FORBIDDEN,
                e.to_string(),
            )),
        )
            .into_response(),
    }
}

#[utoipa::path(
    post,
    path = "/v1/pools/{name}/renew",
    tag = "pools",
    params(("name" = String, Path, description = "Pool name")),
    responses(
        (status = 200, description = "Pool renewed", body = crate::openapi::MessageResponse),
        (status = 404, description = "Pool not found", body = crate::openapi::StandardApiError)
    )
)]
pub async fn renew_pool(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    if state.pool_manager.renew_pool(&name).await {
        info!(event = "pool.renew", pool = %name, status = 200u16, "audit");
        (StatusCode::OK, Json(json!({"message": "Pool renewed"})))
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(json_detail(
                err_code::POOL_NOT_FOUND,
                format!("Pool '{}' not found", name),
            )),
        )
    }
}
