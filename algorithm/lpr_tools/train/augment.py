"""OpenCV/NumPy character augmentation; all randomness belongs to the caller."""

import cv2
import numpy as np


def augment_char(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """32x32 uint8 -> independent 32x32 uint8 (development document 8.2).

    Always apply bounded affine jitter. Independently apply blur (30%), noise
    (50%), one-pixel morphology (20%), and small occlusion (20%).
    """
    if image.shape != (32, 32) or image.dtype != np.uint8:
        raise ValueError("image must be 32 x 32 uint8")
    angle = rng.uniform(-8, 8)
    scale = rng.uniform(0.9, 1.1)
    rotation = np.eye(3, dtype=np.float32)
    rotation[:2] = cv2.getRotationMatrix2D((15.5, 15.5), angle, scale)
    shear = rng.uniform(-0.1, 0.1)
    skew = np.array([[1, shear, -15.5 * shear], [0, 1, 0], [0, 0, 1]], np.float32)
    matrix = skew @ rotation
    matrix[:2, 2] += rng.uniform(-2, 2, 2)
    out = cv2.warpAffine(image, matrix[:2], (32, 32), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if rng.random() < 0.3:
        sigma = rng.uniform(0.2, 1.2)
        out = cv2.GaussianBlur(out, (5, 5), sigmaX=sigma)
    out = out.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15)
    if rng.random() < 0.5:
        out += rng.normal(0, rng.uniform(1, 8), out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).round().astype(np.uint8)
    if rng.random() < 0.2:
        kernel = np.ones((3, 3), np.uint8)  # radius 1 pixel
        operation = cv2.erode if rng.random() < 0.5 else cv2.dilate
        out = operation(out, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    if rng.random() < 0.2:
        width, height = rng.integers(2, 6, 2)
        x, y = rng.integers(0, 33 - width), rng.integers(0, 33 - height)
        out[y:y + height, x:x + width] = 0
    return out
