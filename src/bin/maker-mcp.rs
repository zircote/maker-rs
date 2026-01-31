//! MAKER MCP Server Binary
//!
//! Runs the MAKER framework as an MCP server with stdio transport.
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin maker-mcp
//! ```
//!
//! # Environment Variables
//!
//! - `RUST_LOG` - Log level filter (e.g., `info`, `debug`, `maker=debug`)
//! - `OPENAI_API_KEY` - OpenAI API key (for OpenAI provider)
//! - `ANTHROPIC_API_KEY` - Anthropic API key (for Anthropic provider)

use maker::mcp::MakerServer;
use rmcp::{transport::stdio, ServiceExt};
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing with env filter
    tracing_subscriber::registry()
        .with(fmt::layer().with_target(true))
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,maker=debug")),
        )
        .init();

    info!(
        version = env!("CARGO_PKG_VERSION"),
        "Starting MAKER MCP server"
    );

    // Create the server
    let server = MakerServer::new();

    // Set up graceful shutdown
    let shutdown = async {
        let ctrl_c = async {
            tokio::signal::ctrl_c()
                .await
                .expect("Failed to install Ctrl+C handler");
        };

        #[cfg(unix)]
        let terminate = async {
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("Failed to install SIGTERM handler")
                .recv()
                .await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => {
                info!("Received Ctrl+C, shutting down");
            }
            _ = terminate => {
                info!("Received SIGTERM, shutting down");
            }
        }
    };

    // Start the server with stdio transport
    info!("Listening on stdio transport");

    let service = server.serve(stdio()).await.inspect_err(|e| {
        error!(error = %e, "Failed to start MCP server");
    })?;

    // Wait for shutdown signal or service completion
    tokio::select! {
        result = service.waiting() => {
            match result {
                Ok(_) => info!("MCP service completed"),
                Err(e) => warn!(error = %e, "MCP service error"),
            }
        }
        _ = shutdown => {
            info!("Graceful shutdown complete");
        }
    }

    info!("MAKER MCP server stopped");
    Ok(())
}
