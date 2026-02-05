import cv2
import numpy as np
from PIL import Image, ExifTags
from insightface.app import FaceAnalysis

# Initialize the InsightFace model once at module level.
# "buffalo_l" bundles SCRFD (face detector) + ArcFace (face recognizer).
# It downloads model weights automatically on first run (~300 MB).
_face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
_face_app.prepare(ctx_id=-1, det_size=(640, 640))


def detect_and_embed(image: np.ndarray) -> list[dict]:
    """
    Detect all faces in an image and return their embeddings.

    Args:
        image: BGR image as a numpy array (OpenCV format).

    Returns:
        List of dicts, each containing:
            - bbox: [x1, y1, x2, y2] bounding box
            - embedding: 512-dimensional numpy array
            - confidence: detection confidence score
    """
    faces = _face_app.get(image)
    results = []
    for face in faces:
        results.append({
            "bbox": face.bbox.tolist(),
            "embedding": face.normed_embedding,  # already L2-normalized, 512-d
            "confidence": float(face.det_score),
        })
    return results


def get_single_embedding(image: np.ndarray) -> np.ndarray:
    """
    Detect exactly one face in an image and return its embedding.
    Raises ValueError if zero or multiple faces are found.

    Args:
        image: BGR image as a numpy array.

    Returns:
        512-dimensional embedding as a numpy array.
    """
    faces = detect_and_embed(image)
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")
    if len(faces) > 1:
        raise ValueError(f"Expected 1 face, found {len(faces)}. Please upload a clear selfie with only your face.")
    return faces[0]["embedding"]


def _apply_exif_rotation(pil_image: Image.Image) -> Image.Image:
    """Rotate a PIL image according to its EXIF orientation tag."""
    try:
        exif = pil_image._getexif()
        if exif is None:
            return pil_image
        orientation_key = next(k for k, v in ExifTags.TAGS.items() if v == "Orientation")
        orientation = exif.get(orientation_key)
        rotations = {3: 180, 6: 270, 8: 90}
        if orientation in rotations:
            pil_image = pil_image.rotate(rotations[orientation], expand=True)
    except (StopIteration, AttributeError):
        pass
    return pil_image


def load_image(file_path: str) -> np.ndarray:
    """Load an image from disk as a BGR numpy array, respecting EXIF rotation."""
    pil_img = Image.open(file_path)
    pil_img = _apply_exif_rotation(pil_img)
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Load an image from raw bytes, respecting EXIF rotation."""
    import io
    pil_img = Image.open(io.BytesIO(data))
    pil_img = _apply_exif_rotation(pil_img)
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
