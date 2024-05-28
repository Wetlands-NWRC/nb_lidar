import ee


from src.helpers import rsd
from src.helpers import calcs


## Optical Raster Calcs
def _compute_ndvi(image, nir, red):
    return image.normalizedDifference([nir, red]).rename("NDVI")


def _compute_savi(image, nir, red) -> ee.Image:
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
        rsd.DataCubeCollection()
        .filterBounds(aoi)
        .mosaic()
        .select(".*_b[0][2-8].*|.*b[1][1-2].*")
    )

    # standardize bands
    band_names = []

    ## spring
    spring_dataset = dataset.select("a_spri.*").rename(band_names)


def process_and_stack_images(aoi, terrain_type: str):
    s1 = (
        rsd.Sentinel1()
        .filter_bounds(aoi)
        .apply_edgemask()
        .select()
        .split_season()
        .add_boxcar()
        .add_ratio()
        .build()
    )

    dc = (
        rsd.DataCube()
        .filter_bounds(aoi)
        .compsite()
        .select_spectral_bands()
        .add_ndvi()
        .add_savi()
        .add_tasseled_cap()
        .build()
    )

    al = (
        rsd.AlosPalsar()
        .filter_date(2018, 2020)
        .composite()
        .add_boxcar()
        .select()
        .add_ratio()
        .build()
    )

    ft = rsd.FourierTransform().select("cos.*|sin.*|pha.*|amp.*")
    ta = rsd.TerrrainProdcutLayersFactory().get_layer(terrain_type)

    return ee.Image.cat(s1, dc, al, ft, ta)
