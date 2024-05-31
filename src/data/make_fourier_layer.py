import ee

from src.helpers.compute_fouirer import compute_fourier_transform

_add_ndvi = lambda x: x.addBands(x.normalizedDifference(["B8", "B4"]).rename("NDVI"))


def make_fourier_transform(dataset, dependent, modes):
    # check to make sure that the destination asset dosnet conflict
    modes = 3
    dataset = (
        Sentinel2TOA()
        .filterBounds()
        .filterDate()
        .applyCloudMask()
        .map(_add_ndvi)
        .select("NDVI")
    )

    ft = compute_fourier_transform(
        dataset,
    )
