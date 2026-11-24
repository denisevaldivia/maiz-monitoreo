import numpy as np
import pytest
from PIL import Image

from src import config
from src.data.preprocessing import (
    InvalidImageError,
    extract_patches,
    load_image,
    preprocess_image,
)
from tests.conftest import make_image


def test_output_shape():
    x = preprocess_image(Image.new("RGB", (300, 200), (60, 160, 60)))
    assert x.ndim == 4
    assert 1 <= x.shape[0] <= config.EVAL_K
    assert x.shape[1:] == (3, config.IMG_SIZE, config.IMG_SIZE)


def test_normalization_matches_imagenet_stats():
    x = preprocess_image(Image.new("RGB", (256, 256), (255, 255, 255)))
    expected = [(1 - m) / s for m, s in zip(config.MEAN, config.STD, strict=True)]
    got = x[0, :, 0, 0].tolist()
    assert got == pytest.approx(expected, abs=1e-3)


def test_patches_avoid_black_background():
    arr = np.full((256, 256, 3), (60, 160, 60), dtype=np.uint8)
    arr[:, :128] = 0  # mitad izquierda negra
    patches = extract_patches(Image.fromarray(arr))
    assert patches
    for p in patches:
        assert np.asarray(p).max(axis=2).min() >= config.BLACK_THR


def test_all_black_image_still_returns_patches():
    patches = extract_patches(Image.new("RGB", (256, 256), (0, 0, 0)))
    assert len(patches) == config.EVAL_K
    assert patches[0].size == (config.PATCH, config.PATCH)


def test_load_image_accepts_valid_bytes(good_image_bytes):
    assert load_image(good_image_bytes).mode == "RGB"


def test_load_image_rejects_garbage():
    with pytest.raises(InvalidImageError):
        load_image(b"no soy una imagen")


def test_load_image_rejects_tiny_and_wrong_format():
    with pytest.raises(InvalidImageError):
        load_image(make_image(size=(20, 20)))
    with pytest.raises(InvalidImageError):
        load_image(make_image(fmt="BMP"))
