import ee
import pytest


from src.helpers.rsd import DataCube


AOI = ee.FeatureCollection("projects/nb-lidar/assets/aoi_nb_south").geometry()


def test_filter_bounds():
    instance = DataCube().filter_bounds(AOI)
    assert isinstance(instance.dataset, ee.ImageCollection)


def test_select_spectral_bands():
    instance = DataCube().select_spectral_bands().dataset.first().bandNames().size()
    assert instance.getInfo() == 30


def test_composite():
    instance = DataCube().filter_bounds(AOI).compsite()
    assert isinstance(instance.dataset, ee.Image)


def test_add_ndvi():
    dataset = DataCube().filter_bounds(AOI).compsite().add_ndvi()
    actual = dataset.dataset.select("NDVI.*").bandNames().size().getInfo()
    expected = 3
    assert expected == actual


def test_add_savi():
    dataset = DataCube().filter_bounds(AOI).compsite().add_savi()
    actual = dataset.dataset.select("SAVI.*").bandNames().size().getInfo()
    expected = 3
    assert expected == actual


def test_add_tasseled_cap():
    pattern = "brightness.*|wetness.*|greenness.*"
    dataset = DataCube().filter_bounds(AOI).compsite().add_tasseled_cap()
    actual = dataset.dataset.select(pattern).bandNames().size().getInfo()
    expected = 3 * 3
    assert expected == actual


def test_processor_all():
    instance = (
        DataCube()
        .filter_bounds(AOI)
        .select_spectral_bands()
        .compsite()
        .add_ndvi()
        .add_savi()
        .add_tasseled_cap()
        .build()
    )
    assert isinstance(instance, ee.Image)
