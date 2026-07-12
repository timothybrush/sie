use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};

use crate::http_error::{code as err_code, json_detail};

#[allow(dead_code)]
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("no healthy workers available")]
    NoWorkerAvailable,

    #[error("GPU provisioning in progress: {0}")]
    GpuProvisioning(String),

    #[error("worker connection error: {0}")]
    UpstreamConnection(String),

    #[error("internal error: {0}")]
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let is_gpu_provisioning = matches!(self, AppError::GpuProvisioning(_));
        let (status, code, message) = match &self {
            AppError::NoWorkerAvailable => (
                StatusCode::SERVICE_UNAVAILABLE,
                err_code::QUEUE_UNAVAILABLE,
                self.to_string(),
            ),
            AppError::GpuProvisioning(_) => (
                StatusCode::SERVICE_UNAVAILABLE,
                err_code::PROVISIONING,
                self.to_string(),
            ),
            AppError::UpstreamConnection(_) => (
                StatusCode::BAD_GATEWAY,
                err_code::INTERNAL_ERROR,
                self.to_string(),
            ),
            AppError::Internal(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                err_code::INTERNAL_ERROR,
                self.to_string(),
            ),
        };

        let body = json_detail(code, message);
        let mut response = (status, axum::Json(body)).into_response();
        if is_gpu_provisioning {
            response.headers_mut().insert(
                HeaderName::from_static("retry-after"),
                HeaderValue::from_static("60"),
            );
            response.headers_mut().insert(
                HeaderName::from_static("x-sie-error-code"),
                HeaderValue::from_static(err_code::PROVISIONING),
            );
            response.headers_mut().insert(
                HeaderName::from_static("x-sie-version"),
                HeaderValue::from_static(env!("CARGO_PKG_VERSION")),
            );
            response.headers_mut().insert(
                HeaderName::from_static("x-sie-server-version"),
                HeaderValue::from_static(env!("CARGO_PKG_VERSION")),
            );
        }
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response_status(err: AppError) -> StatusCode {
        err.into_response().status()
    }

    #[test]
    fn test_no_worker_available_is_503() {
        assert_eq!(
            response_status(AppError::NoWorkerAvailable),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[test]
    fn test_gpu_provisioning_is_503() {
        assert_eq!(
            response_status(AppError::GpuProvisioning("l4".into())),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[test]
    fn test_gpu_provisioning_has_retry_headers() {
        let response = AppError::GpuProvisioning("l4".into()).into_response();
        assert_eq!(response.headers().get("retry-after").unwrap(), "60");
        assert_eq!(
            response.headers().get("x-sie-error-code").unwrap(),
            err_code::PROVISIONING
        );
        assert!(response.headers().get("x-sie-version").is_some());
        assert!(response.headers().get("x-sie-server-version").is_some());
    }

    #[test]
    fn test_upstream_connection_is_502() {
        assert_eq!(
            response_status(AppError::UpstreamConnection("refused".into())),
            StatusCode::BAD_GATEWAY
        );
    }

    #[test]
    fn test_internal_is_500() {
        assert_eq!(
            response_status(AppError::Internal("oops".into())),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[test]
    fn test_error_display() {
        assert_eq!(
            AppError::NoWorkerAvailable.to_string(),
            "no healthy workers available"
        );
        assert_eq!(
            AppError::GpuProvisioning("l4".into()).to_string(),
            "GPU provisioning in progress: l4"
        );
    }
}
