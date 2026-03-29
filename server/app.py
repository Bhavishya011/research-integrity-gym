"""
OpenEnv server entry point.

This module provides the main() function for the 'server' script entry point
required by OpenEnv multi-mode deployment.
"""
import uvicorn


def main():
    """Run the FastAPI server."""
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )


if __name__ == "__main__":
    main()
