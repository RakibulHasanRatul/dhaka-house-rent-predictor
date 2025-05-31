from .handler.web.http import HttpHandler
from .handler.web.server import Server


def serve_ui() -> None:
    PORT = 5000
    with Server(("", PORT), HttpHandler) as server:
        print(f"\nServing on http://localhost:{PORT}/\n")
        print("Press Ctrl+C to stop the server.\n")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
            print("\nServer stopped.")
