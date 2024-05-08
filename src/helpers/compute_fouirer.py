from math import pi
from typing import Callable
import ee
import ee.batch



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Dataset
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
class Sentinel2TOA(ee.ImageCollection):
    def __init__(self):
        super().__init__("COPERNICUS/S2_HARMONIZED")

    def addNDVI(self):
        return self.map(self.compute_and_add_ndvi)
    
    def applyCloudMask(self):
        return self.map(self.cloud_mask)
    
    @staticmethod
    def compute_and_add_ndvi(image: ee.Image):
        return image.addBands(image.normalizedDifference(['B8', 'B4']).rename('NDVI'))

    @staticmethod
    def cloud_mask(image: ee.Image):
        qa = image.select("QA60")
        # Bits 10 and 11 are clouds and cirrus, respectively.
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        # Both flags should be set to zero, indicating clear conditions.
        mask = (
            qa.bitwiseAnd(cloud_bit_mask)
            .eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )

        return image.updateMask(mask)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Helpers
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def get_names(prefix: str, frequencies: list[int]) -> list[str]:
    return [f"{prefix}_{freq}" for freq in frequencies]


def add_constant(image) -> ee.Image:
    return image.addBands(ee.Image(1))


def add_time(image: ee.Image) -> ee.Image:
    date = image.date()
    years = date.difference(ee.Date("1970-01-01"), "year")
    time_radians = ee.Image(years.multiply(2 * pi))
    return image.addBands(time_radians.rename("t").float())


def add_harmonics(
    freqs: list[int], cos_names: list[str], sin_names: list[str]
) -> Callable:
    def wrapper(image: ee.Image):
        frequencies = ee.Image.constant(freqs)
        time = ee.Image(image).select("t")
        cosine = time.multiply(frequencies).cos().rename(cos_names)
        sines = time.multiply(frequencies).sin().rename(sin_names)
        return image.addBands(cosine).addBands(sines)

    return wrapper


def compute_phase(cos: str, sin: str):
    name = f'phase_{cos.split("_")[-1]}'

    def wrapper(image: ee.Image) -> ee.Image:
        return image.addBands(
            image.select(cos).atan2(image.select(sin)).unitScale(-pi, pi).rename(name)
        )

    return wrapper


def compute_amplitude(cos: str, sin: str):
    name = f'amplitude_{cos.split("_")[-1]}'

    def wrapper(image: ee.Image):
        return image.addBands(image.select(cos).hypot(image.select(sin)).rename(name))

    return wrapper


def compute_fourier_transform(aoi, start, end, modes: int = 3):
    frequencies = list(range(1, modes + 1))
    cos_names = get_names("cos", frequencies)
    sin_names = get_names("sin", frequencies)
    
    independents = ["t", "constant"] + cos_names + sin_names
    dependent = 'NDVI'
    
    dataset = (
        Sentinel2TOA()
        .filterDate(start, str(int(end) + 1))
        .filterBounds(aoi)
        .applyCloudMask()
        .addNDVI()
        .map(add_constant)
        .map(add_time)
        .map(add_harmonics(frequencies, cos_names, sin_names))
    )
    
    trend =  dataset.select(independents + [dependent]).reduce(
        ee.Reducer.linearRegression(len(independents), 1)
    )
    
    coefficients = (
        trend.select("coefficients").arrayProject([0]).arrayFlatten([independents])
    )
    
    # add coefficients to each image in the dataset
    dataset = dataset.select(dependent).map(lambda x: x.addBands(coefficients))
    
    for cos, sin in zip(cos_names, sin_names):
        dataset = dataset.map(compute_phase(cos, sin)).map(compute_amplitude(cos, sin))

    # transform and return
    return dataset.median().unitScale(-1, 1)