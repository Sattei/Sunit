from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from backend import main


client = TestClient(main.app)


def image_bytes(size: tuple[int, int] = (128, 128)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color=(120, 100, 80)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_output_url_never_exposes_path_outside_job(tmp_path: Path) -> None:
    url = main._output_url_for_result(
        "safe-job",
        {"output_path": str(tmp_path / "secret.png")},
    )

    assert url == "/outputs/safe-job/output/relighted.png"
    assert str(tmp_path) not in url


def test_auto_endpoint_rejects_invalid_light_vector() -> None:
    response = client.post(
        "/api/relight-auto",
        files={"image": ("portrait.png", image_bytes(), "image/png")},
        data={"new_x": "0", "new_y": "0", "new_z": "0"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Light direction cannot be zero."


def test_auto_endpoint_rejects_disguised_text_file() -> None:
    response = client.post(
        "/api/relight-auto",
        files={"image": ("portrait.jpg", b"plain text", "image/jpeg")},
    )

    assert response.status_code == 422
    assert "decodable image" in response.json()["detail"]


def test_auto_endpoint_returns_clean_queue_contract() -> None:
    fake_job = SimpleNamespace(id="queued-job")

    with patch.object(main.relight_queue, "enqueue", return_value=fake_job):
        response = client.post(
            "/api/relight-auto",
            files={"image": ("portrait.png", image_bytes(), "image/png")},
            data={
                "new_x": "0.55",
                "new_y": "-0.2",
                "new_z": "0.8",
                "preset": "natural",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "queued-job",
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "status_url": "/api/jobs/queued-job",
        "result_url": "/api/jobs/queued-job/result",
    }
