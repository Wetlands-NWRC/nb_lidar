import ee

import ee.batch
from tagee import terrainAnalysis

import smoothening as smthn


def compute_bbox_from_geometry(
    table: ee.Geometry | ee.FeatureCollection,
) -> ee.Geometry:
    if isinstance(table, ee.FeatureCollection):
        geom = table.geometry()

    coords = geom.bounds().coordinates()

    listCoords = ee.Array.cat(coords, 1)
    xCoords = listCoords.slice(1, 0, 1)
    yCoords = listCoords.slice(1, 1, 2)

    xMin = xCoords.reduce("min", [0]).get([0, 0])
    xMax = xCoords.reduce("max", [0]).get([0, 0])
    yMin = yCoords.reduce("min", [0]).get([0, 0])
    yMax = yCoords.reduce("max", [0]).get([0, 0])

    return ee.Geometry.Rectangle(xMin, yMin, xMax, yMax)


def compute_elevation_w_gussian(elevation, bbox) -> ee.Image:
    bands = ["Elevation", "Slope", "GaussianCurvature"]
    elevation_w_filter = smthn.apply_guassian_kernel(elevation.select(0)).rename(
        "elevation"
    )
    return terrainAnalysis(elevation_w_filter, bbox).select(bands)


def compute_elevation_w_pm(elevation, bbox) -> ee.Image:
    bands = [
        "HorizontalCurvature",
        "VerticalCurvature",
        "MeanCurvature",
    ]
    elevation_w_filter = smthn.apply_peronal_malik(elevation.select(0)).rename(
        "elevation"
    )
    return terrainAnalysis(elevation_w_filter, bbox).select(bands)
