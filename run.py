"""Entry point for running the Loom web application."""

from loom import LoomApp


if __name__ == "__main__":
    app = LoomApp()
    app.run(host="0.0.0.0", port=8001)
