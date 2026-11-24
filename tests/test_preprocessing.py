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


def test_patches_have_expected_size_and_count():
    patches = extract_patches(Image.new("RGB", (256, 256), (60, 160, 60)))
    assert len(patches) == config.EVAL_K
    assert all(p.size == (config.PATCH, config.PATCH) for p in patches)


def test_small_and_non_square_images_still_give_patches():
    for size in [(64, 64), (300, 200), (200, 900)]:
        patches = extract_patches(Image.new("RGB", size, (60, 160, 60)))
        assert 1 <= len(patches) <= config.EVAL_K


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
