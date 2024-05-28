# Remote Sensing Datasets
import ee


class DataCube(ee.ImageCollection):
    def __init__(self):
        super().__init__(
            "projects/fpca-336015/assets/cnwi-datasets/aoi_newbrunswick/datacube"
        )


class Sentinel1(ee.ImageCollection):
    def __init__(self, args: list[str] = None):
        args = args or "COPERNICUS/S1_GRD"
        super().__init__(args)

    def applyEdgeMask(self):
        return self.map(self.edge_mask)

    def filterDV(self):
        return self.filter(
            ee.Filter.listContains("transmitterReceiverPolarisation", "VV")
        ).filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))

    def filterIWMode(self):
        return self.filter(ee.Filter.eq("instrumentMode", "IW"))

    def filterDesc(self):
        return self.filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))

    def filterAsc(self):
        return self.filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))

    @staticmethod
    def edge_mask(image: ee.Image):
        edge = image.lt(-30.0)
        masked_image = image.mask().And(edge.Not())
        return image.updateMask(masked_image)


class Sentinel2(ee.ImageCollection):
    def __init__(self, args):
        super().__init__(args)

    def applyCloudMask(self):
        def mask(image):
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
            return image.updateMask(mask).divide(10000)

        return self.map(mask)

    def filterClouds(self, percent):
        return self.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", percent))


class Sentinel2SR(Sentinel2):
    def __init__(self):
        super().__init__("COPERNICUS/S2_SR_HARMONIZED")


class Sentinel2TOA(Sentinel2):
    def __init__(self):
        super().__init__("COPERNICUS/S2_HARMONIZED")


class AlosPalsar(ee.ImageCollection):
    def __init__(self) -> None:
        super().__init__("JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH")


class FourierTransform(ee.Image):
    def __init__(self):
        super().__init__("projects/nb-lidar/assets/s2_nb_south_ft")


class TerrainDTM(ee.Image):
    def __init__(self):
        super().__init__("projects/nb-lidar/assets/dtm_nb_ta")


class TerrainDSM(ee.Image):
    def __init__(self):
        super().__init__("projects/nb-lidar/assets/dsm_nb_ta")


class TerrainSRTM(ee.Image):
    def __init__(self):
        super().__init__("projects/nb-lidar/assets/srtm_nb_south_ta")
