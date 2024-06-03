import ee


def extract_features(
    image: ee.Image,
    input_data: ee.FeatureCollection,
    properties: list[str] | ee.List = None,
    scale: int = 10,
    projection=None,
    tileScale: int = 16,
    geometries: bool = False,
):
    """extracts features from the input image"""
    return image.sampleRegions(
        collection=input_data,
        properties=properties,
        scale=scale,
        projection=projection,
        tileScale=tileScale,
        geometries=geometries,
    )
