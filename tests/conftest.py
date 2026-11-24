import io

import pytest
from PIL import Image


def make_image(size=(256, 256), color=(60, 160, 60), fmt="JPEG"):
    """Imagen sintetica en bytes."""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def good_image_bytes():
    return make_image()
