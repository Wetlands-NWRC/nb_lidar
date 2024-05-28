import ee
import pytest
from src.helpers.calcs import RasterCalc

TEST_IMAGE = ee.Image(list(range(1, 7)))
calc = RasterCalc()


def test_compute_ndvi():
    ndvi = calc.compute_ndvi(TEST_IMAGE, 3, 4)
    expected = ["NDVI"]
    actual = ndvi.bandNames().getInfo()
    assert expected == actual


def test_compute_savi():
    savi = calc.compute_savi(TEST_IMAGE, 3, 4)
    expected = ["SAVI"]
    actual = savi.bandNames().getInfo()
    assert expected == actual


def test_compute_tasseled_cap():
    tc = calc.compute_tassele_cap(TEST_IMAGE, 0, 1, 2, 3, 4, 5)
    expected = ["brightness", "greenness", "wetness"]
    actual = tc.bandNames().getInfo()
    assert expected == actual


def test_compute_ratio():
    ratio = calc.compute_ratio(TEST_IMAGE, 0, 1)
    expected = ["0_1"]
    actual = ratio.bandNames().getInfo()
    assert expected == actual
