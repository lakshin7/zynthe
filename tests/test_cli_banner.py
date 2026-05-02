from __future__ import annotations

from app.main import banner_text


def test_banner_text_contains_large_zynthe() -> None:
    banner = banner_text()

    assert "ZYNTHE" in banner
    assert len(banner.splitlines()) >= 3
