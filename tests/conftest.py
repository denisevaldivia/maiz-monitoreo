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


@pytest.fixture
def image_dir(tmp_path):
    """Carpeta con 2 imagenes buenas, una corrupta y una muy chica."""
    (tmp_path / "ok1.jpg").write_bytes(make_image())
    (tmp_path / "ok2.png").write_bytes(make_image(color=(90, 120, 40), fmt="PNG"))
    (tmp_path / "corrupta.jpg").write_text("esto no es una imagen")
    (tmp_path / "chica.jpg").write_bytes(make_image(size=(20, 20)))
    return tmp_path
