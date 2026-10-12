"""Serve only explicit public UI files; never mount the repository or its data."""
from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1] / "frontend"
ASSETS = {"app.js": "text/javascript", "core.js": "text/javascript",
          "render.js": "text/javascript", "styles.css": "text/css"}


def mount_frontend(app):
    def public_file(name, media_type):
        path = ROOT / name
        if not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise HTTPException(status_code=503, detail="Frontend asset unavailable")
        return FileResponse(path, media_type=media_type, headers={
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'self' https://colab.research.google.com https://*.googleusercontent.com",
        })

    @app.get("/", include_in_schema=False)
    def index():
        return public_file("index.html", "text/html")

    @app.get("/static/{name}", include_in_schema=False)
    def asset(name: str):
        if name not in ASSETS:
            raise HTTPException(status_code=404, detail="Not found")
        return public_file(name, ASSETS[name])
