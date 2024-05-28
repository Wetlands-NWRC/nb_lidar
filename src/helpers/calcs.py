import ee


class RasterCalc:

    def __call__(self, image: ee.Image) -> ee.Any:
        return self.add(image)

    def add(self, image: ee.Image):
        return image.addBands(self.compute(image))

    def compute(self, image: ee.Image):
        # compute logic goes here
        pass


class NDVI(RasterCalc):
    def __init__(self, nir: str | int, red: str | int) -> None:
        self.nir = nir
        self.red = red
        super().__init__()

    def compute(self, image: ee.Image):
        if isinstance(self.nir, int):
            self.nir = image.bandNames().get(self.nir)
        if isinstance(self.red, int):
            self.red = self.red.bandNames().get(self.red)
        return image.normalizedDifference([self.nir, self.red]).rename("NDVI")


class RasterCalc:
    def compute_ndvi(self, image: ee.Image, nir: str | int = 0, red: str | int = 1):
        if isinstance(nir, int):
            nir = image.bandNames().get(nir)
        if isinstance(red, int):
            red = image.bandNames().get(red)
        return image.normalizedDifference([nir, red]).rename("NDVI")

    def compute_savi(
        self, image: ee.Image, nir: str | int, red: str | int, L: float = 0.5
    ):
        return image.expression(
            "(1 + L) * (NIR - RED) / (NIR + RED + L)",
            {"NIR": image.select(nir), "RED": image.select(red), "L": L},
        ).rename("SAVI")

    def compute_tassele_cap(
        self,
        image: ee.Image,
        b: str | int = 0,
        g: str | int = 1,
        r: str | int = 2,
        nir: str | int = 3,
        swir_1: str | int = 4,
        swir_2: str | int = 5,
        *args,
    ):
        if not args:
            args = [b, g, r, nir, swir_1, swir_2]
        else:
            args = list(args)

        if len(args) != 6:
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
        array_image = image.select(args).toArray()
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

    def compute_ratio(self, image: ee.Image, b1: str | int = 0, b2: str | int = 1):
        return image.select(b1).divide(image.select(b2)).rename(f"{b1}_{b2}")
