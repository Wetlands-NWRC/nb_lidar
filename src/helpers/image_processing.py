import ee


from src.helpers import rsd


def _compute_ndvi(image, nir: str | int, red: str | int) -> ee.Image:
    pass


def _compute_savi():
    pass


def _compute_tasselecd_cap():
    pass


def fetch_and_process_data_cube(aoi) -> ee.Image:
    """ fetch and apply processing steps to Data Cube Composites """
    dataset = rsd.DataCubeCollection().filterBounds(aoi).mosaic().select("<>")
    
    # standardize bands
    band_names = []
    
    ## spring
    spring_dataset = dataset.select('a_spri.*').rename(band_names)
    # add ndvi 
    ndvi_spring = dataset.addBands(_compute_ndvi(dataset, ))
    savi = dataset.addBands(_compute_savi)
    tc = add_tasseled_cap()(image.select())



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
