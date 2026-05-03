from fastapi.testclient import TestClient
from PIL import Image
import io


def make_jpeg_bytes():
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_detect_returns_label_and_score():
    from main import app
    client = TestClient(app)
    resp = client.post(
        "/detect",
        files={"file": ("test.jpg", make_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "label" in body
    assert "score" in body
    assert isinstance(body["score"], float)
    assert 0.0 <= body["score"] <= 1.0
