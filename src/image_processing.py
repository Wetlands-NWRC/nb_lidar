import ee


from src.helpers import rsd


## Optical Raster Calcs
def _compute_ndvi(image, nir, red):
    return image.normalizedDifference([nir, red]).rename("NDVI")


def _compute_savi(image, nir, red, L: float = 0.5) -> ee.Image:
    return image.expression(
        "(1 + L) * (NIR - RED) / (NIR + RED + L)",
        {"NIR": image.select(nir), "RED": image.select(red), "L": L},
    ).rename("SAVI")


def _compute_tasseled_cap(image, *bands) -> ee.Image:
    if not bands:
        bands = list(range(0, 6))
    else:
        bands = list(bands)

    if len(bands) != 6:
        raise ValueError("Must Provide exactly 6 bands")

    coefficients = ee.Array(
        [
            [0.3029, 0.2786, 0.4733, 0.5599, 0.508, 0.1872],
            [-0.2941, -0.243, -0.5424, 0.7276, 0.0713, -0.1608],
            [0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559],
            [-0.8239, 0.0849, 0.4396, -0.058, 0.2013, -0.2773],
            [-0.3294, 0.0557, 0.1056, 0.1855, -0.4349, 0.8085],
            [0.1079, -0.9023, 0.4119, 0.0575, -0.0259, 0.0252],
        ]
    )
    array_image = image.select(bands).toArray()
    array_image_2d = array_image.toArray(1)

    components = (
        ee.Image(coefficients)
        .matrixMultiply(array_image_2d)
        .arrayProject([0])
        .arrayFlatten(
            [["brightness", "greenness", "wetness", "fourth", "fifth", "sixth"]]
        )
    )

    return components.select(["brightness", "greenness", "wetness"])


def fetch_and_process_data_cube(aoi) -> ee.Image:
    """fetch and apply processing steps to Data Cube Composites"""
    dataset = (
        rsd.DataCube().filterBounds(aoi).mosaic().select(".*_b[0][2-8].*|.*b[1][1-2].*")
    )

    # standardize bands
    band_names = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    prefixs = ["a_spri.*", "b_summ.*", "c_fall.*"]

    output_dataset = None
    for prefix in prefixs:
        input_dataset = dataset.select(prefix).rename(band_names)
        ndvi = _compute_ndvi(input_dataset, "B8", "B4")
        savi = _compute_savi(input_dataset, "B8", "B4")
        tc = _compute_tasseled_cap(input_dataset, "B2", "B3", "B4", "B8", "B11", "B12")
        input_dataset = input_dataset.addBands(ndvi).addBands(savi).addBands(tc)

        if output_dataset is None:
            output_dataset = input_dataset
            continue
        output_dataset = output_dataset.addBands(input_dataset)

    return output_dataset


def fetch_and_proecss_s1_seasonal(aoi):
    """processing for Sentinel 1 seasonal composites"""
    s1_dataset = (
        rsd.Sentinel1()
        .filterDate("2019-04-01", "2019-09-21")
        .filterBounds(aoi)
        .filterIWMode()
        .filterDV()
        .applyEdgeMask()
        .median()
        .select("V.*")
        .convolve(ee.kernel.Kernel.square(1))
    )
    return s1_dataset


def fetch_and_proeces_s2_sr_seasonal(
    aoi, ndvi: bool = False, savi: bool = False, tasseled_cap: bool = False
) -> ee.Image:
    """processing chain for s2 sr seasonal composite"""
    s2sr_dataset = (
        rsd.Sentinel2SR()
        .filterBounds(aoi)
        .filterDate("2019", "2022")
        .filter(ee.Filter.dayOfYear())
        .filterClouds(1)
        .sort("CLOUDY_PIXEL_PERCENTAGE", False)
        .median()
        .select("B[2-9].*|B1[1-2]")
    )

    if ndvi:
        s2sr_dataset = s2sr_dataset.addBands(_compute_ndvi(s2sr_dataset, "B8", "B4"))

    if savi:
        s2sr_dataset = s2sr_dataset.addBands(_compute_savi(s2sr_dataset, "B8", "B4"))

    if tasseled_cap:
        s2sr_dataset = s2sr_dataset.addBands(
            _compute_tasseled_cap(s2sr_dataset, "B2", "B3", "B4", "B8", "B11", "B12")
        )

    return s2sr_dataset


def fetch_and_process_alos() -> ee.Image:
    return (
        rsd.AlosPalsar()
        .filterDate("2018-01-01", "2022-01-01")
        .median()
        .convolve(ee.Kernel.square(1))
        .select("H.*")
    )


def fetch_fourier_transform() -> ee.Image:
    return rsd.FourierTransform().select("cos.*|sin.*|pha.*|amp.*")


def fetch_dtm_ta() -> ee.Image:
    return rsd.TerrainDTM()


def fetch_dsm_ta() -> ee.Image:
    return rsd.TerrainDSM()


def fetch_srtm_ta() -> ee.Image:
    return rsd.TerrainSRTM()
