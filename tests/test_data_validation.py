from pathlib import Path

import pandas as pd
import pytest

from src.data import validation as v

SPLIT = Path("data/split_v1.csv")


@pytest.fixture(scope="module")
def split():
    return pd.read_csv(SPLIT)


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
