from __future__ import annotations

import json
import urllib.request
from typing import Dict, Any

class LiveStreamClient:
    """Tiny HTTP client to post Trainer events to FastAPI /api/stream without extra deps.
    """
    def __init__(self, stream_url: str = "http://127.0.0.1:8000/api/stream") -> None:
        self.stream_url = stream_url

    def send(self, event: Dict[str, Any]) -> None:
        try:
            data = json.dumps(event).encode("utf-8")
            req = urllib.request.Request(
                self.stream_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                _ = resp.read()
        except Exception:
            # Non-fatal: streaming is best-effort; ignore failures during training
            pass
