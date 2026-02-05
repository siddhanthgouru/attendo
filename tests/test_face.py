"""
Basic smoke test for the face recognition service.

To run:
    1. Place any photo with a face at tests/test_photo.jpg
    2. Run: pytest tests/test_face.py -v
"""
import os

import numpy as np
import pytest


@pytest.fixture
def sample_image():
    """Load a test image if available, otherwise create a blank one."""
    from app.services.face import load_image

    test_path = os.path.join(os.path.dirname(__file__), "test_photo.jpg")
    if os.path.exists(test_path):
        return load_image(test_path)
    pytest.skip("No test_photo.jpg found in tests/ — place a selfie there to run this test.")


def test_detect_and_embed(sample_image):
    from app.services.face import detect_and_embed

    results = detect_and_embed(sample_image)
    assert len(results) >= 1, "Should detect at least one face"

    face = results[0]
    assert "bbox" in face
    assert "embedding" in face
    assert "confidence" in face
    assert len(face["bbox"]) == 4
    assert face["embedding"].shape == (512,)
    assert face["confidence"] > 0.5


def test_get_single_embedding(sample_image):
    from app.services.face import get_single_embedding

    embedding = get_single_embedding(sample_image)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (512,)
    # Verify it's L2-normalized (magnitude ≈ 1)
    magnitude = np.linalg.norm(embedding)
    assert abs(magnitude - 1.0) < 0.01
