//! Optional same-origin delivery for the prebuilt HTML/JavaScript/CSS client.
//!
//! The web toolchain creates static files before launch; this module serves
//! those files from the Rust gateway and gives extensionless client routes the
//! application's index document. API misses remain 404 responses.

use std::path::{Component, Path, PathBuf};

use axum::body::Body;
use axum::extract::{Extension, OriginalUri};
use axum::http::{header, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use tokio::fs;

#[derive(Debug, Clone)]
pub struct WebRoot(pub PathBuf);

/// Serve one file from the configured static root, with an index fallback for
/// client routes. Canonical paths are checked so symlinks cannot escape root.
pub async fn serve(
    method: Method,
    Extension(web_root): Extension<WebRoot>,
    OriginalUri(uri): OriginalUri,
) -> Response {
    let path = uri.path();
    if path == "/api" || path.starts_with("/api/") || path == "/api/v1" {
        return StatusCode::NOT_FOUND.into_response();
    }
    if method != Method::GET && method != Method::HEAD {
        return StatusCode::METHOD_NOT_ALLOWED.into_response();
    }

    let Ok(canonical_root) = fs::canonicalize(&web_root.0).await else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let Some(relative) = safe_relative_path(path) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let target = canonical_root.join(relative);
    if let Some(response) = file_response(&canonical_root, &target).await {
        return response;
    }

    // Vite's browser routes are client-side routes. Only an extensionless
    // non-API route falls back to index.html; a missing asset remains a 404.
    let has_extension = target.extension().is_some();
    if !has_extension {
        let index = canonical_root.join("index.html");
        if let Some(response) = file_response(&canonical_root, &index).await {
            return response;
        }
    }
    StatusCode::NOT_FOUND.into_response()
}

fn safe_relative_path(path: &str) -> Option<PathBuf> {
    if !path.starts_with('/') || path.contains(['%', '\\', '\0']) {
        return None;
    }
    let mut relative = PathBuf::new();
    for segment in path.split('/').filter(|segment| !segment.is_empty()) {
        if segment == "." || segment == ".." || segment.contains(':') {
            return None;
        }
        relative.push(segment);
    }
    if relative
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return None;
    }
    Some(relative)
}

async fn file_response(root: &Path, target: &Path) -> Option<Response> {
    let canonical_target = fs::canonicalize(target).await.ok()?;
    if !canonical_target.starts_with(root) {
        return None;
    }
    let metadata = fs::metadata(&canonical_target).await.ok()?;
    if !metadata.is_file() {
        return None;
    }
    let bytes = fs::read(&canonical_target).await.ok()?;
    let content_type = content_type(&canonical_target);
    Some(
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, content_type)
            .header(header::CACHE_CONTROL, "no-cache")
            .header(header::X_CONTENT_TYPE_OPTIONS, "nosniff")
            .body(Body::from(bytes))
            .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response()),
    )
}

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js" | "mjs") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("ico") => "image/x-icon",
        Some("json" | "map") => "application/json",
        Some("woff") => "font/woff",
        Some("woff2") => "font/woff2",
        Some("ttf") => "font/ttf",
        Some("wasm") => "application/wasm",
        Some("txt") => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::{serve, WebRoot};
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use axum::routing::get;
    use axum::{Extension, Router};
    use std::fs;
    use tower::ServiceExt;

    async fn request(router: Router, path: &str) -> (StatusCode, String, Vec<u8>) {
        let response = router
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let content_type = response
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        let bytes = to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap()
            .to_vec();
        (status, content_type, bytes)
    }

    #[tokio::test]
    async fn serves_assets_and_client_routes_but_keeps_api_misses_and_traversal_closed() {
        let directory = tempfile::tempdir().unwrap();
        fs::create_dir(directory.path().join("assets")).unwrap();
        fs::write(directory.path().join("index.html"), "<main>client</main>").unwrap();
        fs::write(directory.path().join("assets/app.js"), "export {}; ").unwrap();
        let router = Router::new()
            .fallback(get(serve))
            .layer(Extension(WebRoot(directory.path().to_path_buf())));

        let (status, content_type, body) = request(router.clone(), "/").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(content_type, "text/html; charset=utf-8");
        assert_eq!(body, b"<main>client</main>");

        let (status, content_type, body) = request(router.clone(), "/assets/app.js").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(content_type, "text/javascript; charset=utf-8");
        assert_eq!(body, b"export {}; ");

        let (status, content_type, body) = request(router.clone(), "/live/table").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(content_type, "text/html; charset=utf-8");
        assert_eq!(body, b"<main>client</main>");

        let (status, _, body) = request(router.clone(), "/api/v1/missing").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(body.is_empty());

        let (status, _, body) = request(router.clone(), "/assets/missing.js").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(body.is_empty());

        let (status, _, body) = request(router, "/%2e%2e/secret").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(body.is_empty());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn refuses_symlinks_that_escape_the_static_root() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("secret.txt"), "private").unwrap();
        symlink(outside.path(), directory.path().join("escape")).unwrap();
        let router = Router::new()
            .fallback(get(serve))
            .layer(Extension(WebRoot(directory.path().to_path_buf())));

        let (status, _, body) = request(router, "/escape/secret.txt").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(body.is_empty());
    }
}
