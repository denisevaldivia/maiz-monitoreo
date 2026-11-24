import pandas as pd

from src import config
from src.data import validation as v


def test_valid_images_have_no_problems(image_dir):
    assert v.validate_image_file(image_dir / "ok1.jpg") == []
    assert v.validate_image_file(image_dir / "ok2.png") == []


def test_corrupt_and_tiny_images_are_reported(image_dir):
    assert v.validate_image_file(image_dir / "corrupta.jpg")
    assert v.validate_image_file(image_dir / "chica.jpg")
    assert v.validate_image_file(image_dir / "no_existe.jpg")


def test_labels_must_be_known_classes():
    ok = pd.DataFrame({"label": config.CLASSES})
    bad = pd.DataFrame({"label": ["healthy", "tomato_blight", None]})
    assert v.validate_labels(ok) == []
    assert len(v.validate_labels(bad)) == 2  # una etiqueta invalida y una vacia


def test_clean_split_has_no_leakage():
    df = pd.DataFrame(
        {
            "path": ["a.jpg", "b.jpg", "c.jpg"],
            "split": ["train", "val", "test"],
            "group": [1, 2, 3],
        }
    )
    assert v.find_leakage(df) == []


def test_same_image_in_two_splits_is_detected():
    df = pd.DataFrame({"path": ["a.jpg", "a.jpg"], "split": ["train", "test"], "group": [1, 1]})
    assert v.find_leakage(df)


def test_duplicate_group_across_splits_is_detected():
    df = pd.DataFrame({"path": ["a.jpg", "b.jpg"], "split": ["train", "test"], "group": [7, 7]})
    assert any("grupo" in p for p in v.find_leakage(df))


def test_identical_content_across_splits_is_detected(image_dir):
    (image_dir / "copia.jpg").write_bytes((image_dir / "ok1.jpg").read_bytes())
    df = pd.DataFrame({"path": ["ok1.jpg", "copia.jpg"], "split": ["train", "test"]})
    assert v.find_identical_files_across_splits(df, image_dir)
