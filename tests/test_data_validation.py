from pathlib import Path

import pandas as pd
import pytest

from src import config
from src.data import validation as v

SPLIT = Path("data/split_v1.csv")
PATCHES = Path("data/patch_index_v1.csv")


@pytest.fixture(scope="module")
def split():
    return pd.read_csv(SPLIT)


@pytest.fixture(scope="module")
def patches():
    return pd.read_csv(PATCHES)


# SPLIT


def test_total_images(split):
    assert len(split) == 3852  # maiz de PlantVillage


def test_no_image_listed_twice(split):
    assert split["s3_key"].is_unique
    assert split["seg_key"].is_unique


def test_labels_are_valid(split):
    assert v.validate_labels(split, label_col="label") == []


def test_label_matches_folder(split):
    # maize/raw/<clase>/archivo.jpg -> la carpeta debe ser la etiqueta
    folder = split["s3_key"].str.split("/").str[2]
    assert (folder == split["label"]).all()


def test_no_leakage_between_splits(split):
    assert v.find_leakage(split, "s3_key", "split", "group") == []


def test_split_names(split):
    assert set(split["split"]) == {"train", "val", "test"}


def test_split_proportions(split):
    frac = split["split"].value_counts(normalize=True)
    assert frac["train"] == pytest.approx(0.70, abs=0.04)
    assert frac["val"] == pytest.approx(0.15, abs=0.04)
    assert frac["test"] == pytest.approx(0.15, abs=0.04)


def test_every_class_in_every_split(split):
    counts = split.groupby(["split", "label"]).size().unstack()
    assert not counts.isna().any().any()


# INDEX


def test_patches_only_from_images_in_split(patches, split):
    assert set(patches["seg_key"]) <= set(split["seg_key"])


def test_patches_keep_split_label_and_group(patches, split):
    cols = ["seg_key", "split", "label", "group"]
    m = patches.merge(split[cols], on="seg_key", suffixes=("", "_img"))
    assert len(m) == len(patches)
    for col in ("split", "label", "group"):
        assert (m[col] == m[col + "_img"]).all(), f"{col} distinto entre parche e imagen"


def test_patch_size_and_positions(patches):
    assert (patches["size"] == config.PATCH).all()
    limit = config.SOURCE_SIZE - config.PATCH
    assert patches["y"].between(0, limit).all()
    assert patches["x"].between(0, limit).all()
    assert (patches["y"] % config.STRIDE == 0).all()
    assert (patches["x"] % config.STRIDE == 0).all()


def test_no_repeated_patches(patches):
    assert not patches.duplicated(["seg_key", "y", "x"]).any()


def test_every_class_has_patches_in_every_split(patches):
    counts = patches.groupby(["split", "label"]).size().unstack()
    assert not counts.isna().any().any()
