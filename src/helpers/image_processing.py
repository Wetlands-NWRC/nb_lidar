import ee


from src.helpers import rsd


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
