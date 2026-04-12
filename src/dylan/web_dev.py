"""Serve ``web/`` on loopback so the console shows ``http://127.0.0.1/…`` instead of IPv6 ``[::]``."""

from __future__ import annotations

import argparse
import http.server
import socketserver
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    """Run a static file server for the repository ``web/`` directory."""
    parser = argparse.ArgumentParser(
        description="Serve web/ over HTTP on 127.0.0.1 (readable localhost URL for Pyodide dev).",
    )
    parser.add_argument("--port", type=int, default=8000, help="TCP port (default: 8000)")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    root = repo_root / "web"
    if not root.is_dir():
        print(f"Directory not found: {root}", file=sys.stderr)
        raise SystemExit(1)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a: object, **kw: object) -> None:
            super().__init__(*a, directory=str(root), **kw)

    with socketserver.TCPServer((args.host, args.port), Handler) as httpd:
        print(f"Serving {root}")
        print(f"  http://127.0.0.1:{args.port}/")
        print(f"  http://localhost:{args.port}/")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
