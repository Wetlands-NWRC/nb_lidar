# Remote Sensing Datasets
import ee

from src.helpers.calcs import RasterCalc
from src.helpers.smoothening import make_boxcar


class DataCube:
    def __init__(self) -> None:
        self.dataset = ee.ImageCollection(
            "projects/fpca-336015/assets/cnwi-datasets/aoi_newbrunswick/datacube"
        )

    def filter_bounds(self, aoi):
        self.dataset = self.dataset.filterBounds(aoi)
        return self

    def select_spectral_bands(self):
        self.dataset = self.dataset.select(".*_b[0][2-8].*|.*b[1][1-2].*")
        return self

    def compsite(self):
        self.dataset = self.dataset.mosaic()
        return self

    def add_ndvi(self):
        nir = 6
        red = 2
        for _ in range(0, 3):
            self.dataset = self.dataset.addBands(
                RasterCalc().compute_ndvi(self.dataset, nir, red)
            )
            nir += 10
            red += 10
        return self

    def add_savi(self):
        nir = 6
        red = 2
        for _ in range(0, 3):
            self.dataset = self.dataset.addBands(
                RasterCalc().compute_savi(self.dataset, nir, red)
            )
            nir += 10
            red += 10
        return self

    def add_tasseled_cap(self):
        band_idxs = [0, 1, 2, 6, 8, 9]
        for _ in range(0, 3):
            self.dataset = self.dataset.addBands(
                RasterCalc().compute_tassele_cap(self.dataset, *band_idxs)
            )
            band_idxs = map(lambda x: x + 10, band_idxs)
        return self

    def build(self):
        return self.dataset


s1_assets = [
    "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190618T221053_20190618T221118_027740_03219A_D15A",
    "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190618T221118_20190618T221143_027740_03219A_E09B",
    "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190618T221143_20190618T221208_027740_03219A_0877",
    "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190618T221208_20190618T221233_027740_03219A_ECED",
    "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190805T221056_20190805T221121_028440_0336C5_8F55",
    "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190805T221121_20190805T221146_028440_0336C5_322A",
    "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190805T221146_20190805T221211_028440_0336C5_0053",
    "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190805T221211_20190805T221236_028440_0336C5_042F",
    "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190611T221902_20190611T221931_027638_031E91_DCA1",
    "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190611T221931_20190611T221956_027638_031E91_1D95",
    "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190611T221956_20190611T222021_027638_031E91_1E80",
    "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190611T222021_20190611T222046_027638_031E91_EA9A",
    "COPERNICUS/S1_GRD/S1B_IW_GRDH_1SDV_20190804T221815_20190804T221840_017442_020CE3_4EE2",
    "COPERNICUS/S1_GRD/S1B_IW_GRDH_1SDV_20190804T221840_20190804T221904_017442_020CE3_7783",
    "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190628T222725_20190628T222750_027886_0325FF_90F6",
    "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190628T222750_20190628T222815_027886_0325FF_DA8A",
    "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190628T222815_20190628T222840_027886_0325FF_30DE",
    "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190803T222728_20190803T222753_028411_0335EA_76AD",
    "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190803T222753_20190803T222818_028411_0335EA_0A51",
    "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190803T222818_20190803T222843_028411_0335EA_6745",
]


class Sentinel1:
    def __init__(self) -> None:
        self.dataset = ee.ImageCollection(s1_assets)

    def select(self):
        self.dataset = self.dataset.select("V.*")
        return self

    def filter_bounds(self, aoi):
        self.dataset = self.dataset.filterBounds(aoi)
        return self

    def split_season(
        self, spring_range: tuple[str, str] = None, summer_range: tuple[str, str] = None
    ):
        spring_image = self.dataset.filterDate("2019-03-21", "2019-06-20").mosaic()
        summer_image = self.dataset.filterDate("2019-06-21", "2019-09-21").mosaic()
        self.dataset = spring_image.addBands(summer_image)
        return self

    def add_ratio(self):
        spring_ratio = RasterCalc().compute_ratio(self.dataset, "VV", "VH")
        summer_ratio = RasterCalc().compute_ratio(self.dataset, "VV_1", "VH_1")
        self.dataset = self.dataset.addBands(spring_ratio).addBands(summer_ratio)
        return self

    def apply_edgemask(self):
        self.dataset = self.dataset.map(self.edge_mask)
        return self

    def add_boxcar(self):
        if not isinstance(self.dataset, ee.Image):
            raise TypeError(
                "Data set must me transformed to image before adding boxcar"
            )
        self.dataset = self.dataset.convolve(make_boxcar())
        return self

    def build(self):
        return self.dataset

    @staticmethod
    def edge_mask(image: ee.Image):
        edge = image.lt(-30.0)
        masked_image = image.mask().And(edge.Not())
        return image.updateMask(masked_image)


class AlosPalsar:
    def __init__(self) -> None:
        self.dataset = ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH")

    def select(self):
        self.dataset = self.dataset.select("H.*")
        return self

    def filter_date(self, start_yyyy: int, end_yyyy: int):
        self.dataset = self.dataset.filterDate(str(start_yyyy), str(end_yyyy + 1))
        return self

    def add_ratio(self):
        if not isinstance(self.dataset, ee.Image):
            raise TypeError("Data set must me transformed to image before adding ratio")
        ratio = RasterCalc().compute_ratio(self.dataset, "HH", "HV")
        self.dataset = self.dataset.addBands(ratio)
        return self

    def add_boxcar(self):
        if not isinstance(self.dataset, ee.Image):
            raise TypeError(
                "Data set must me transformed to image before adding boxcar"
            )
        self.dataset = self.dataset.convolve(make_boxcar())
        return self

    def composite(self):
        self.dataset = self.dataset.median()
        return self

    def build(self) -> ee.Image:
        return self.dataset


class AuxLayer(ee.Image):
    def __init__(self, args: str):
        if not isinstance(args, str):
            raise TypeError("Must be a string to an asset")
        super().__init__(args, None)


class FourierTransform(AuxLayer):
    def __init__(self):
        super().__init__("projects/nb-lidar/assets/s2_nb_south_ft")


class TerrainDTM(AuxLayer):
    def __init__(self):
        super().__init__("projects/nb-lidar/assets/dtm_nb_ta")


class TerrainDSM(AuxLayer):
    def __init__(self):
        super().__init__("projects/nb-lidar/assets/dsm_nb_ta")


class TerrainSRTM(AuxLayer):
    def __init__(self):
        super().__init__("projects/nb-lidar/assets/srtm_nb_south_ta")


class TerrrainProdcutLayersFactory:
    def get_layer(self, dataset_type: str) -> AuxLayer | ee.Image:
        datasets = {"dtm": TerrainDTM, "dsm": TerrainDSM, "srtm": TerrainSRTM}

        dataset = datasets.get(dataset_type.lower())
        if dataset is None:
            raise ValueError("Type must me dtm, dsm, srtm")
        return dataset()


