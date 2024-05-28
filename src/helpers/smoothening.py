import ee


def apply_guassian_kernel(
    image: ee.Image, radius: int = 3, sigma: int = 2
) -> ee.kernel.Kernel:
    kernel = ee.Kernel.gaussian(
        radius=radius, sigma=sigma, units="pixels", normalize=True, magnitude=1.0
    )
    return image.convolve(kernel)


def apply_peronal_malik(img: ee.Image, K=3.5, iterations=10, method=2) -> ee.Image:
    dxW = ee.Kernel.fixed(3, 3, [[0, 0, 0], [1, -1, 0], [0, 0, 0]])
    dxE = ee.Kernel.fixed(3, 3, [[0, 0, 0], [0, -1, 1], [0, 0, 0]])
    dyN = ee.Kernel.fixed(3, 3, [[0, 1, 0], [0, -1, 0], [0, 0, 0]])
    dyS = ee.Kernel.fixed(3, 3, [[0, 0, 0], [0, -1, 0], [0, 1, 0]])
    lamb = 0.2

    k1 = ee.Image(-1.0 / K)
    k2 = ee.Image(K).multiply(ee.Image(K))

    for _ in range(0, iterations):
        dI_W = img.convolve(dxW)
        dI_E = img.convolve(dxE)
        dI_N = img.convolve(dyN)
        dI_S = img.convolve(dyS)

        if method == 1:
            cW = dI_W.multiply(dI_W).multiply(k1).exp()
            cE = dI_E.multiply(dI_E).multiply(k1).exp()
            cN = dI_N.multiply(dI_N).multiply(k1).exp()
            cS = dI_S.multiply(dI_S).multiply(k1).exp()

            img = img.add(
                ee.Image(lamb).multiply(
                    cN.multiply(dI_N)
                    .add(cS.multiply(dI_S))
                    .add(cE.multiply(dI_E))
                    .add(cW.multiply(dI_W))
                )
            )

        else:
            cW = ee.Image(1.0).divide(ee.Image(1.0).add(dI_W.multiply(dI_W).divide(k2)))
            cE = ee.Image(1.0).divide(ee.Image(1.0).add(dI_E.multiply(dI_E).divide(k2)))
            cN = ee.Image(1.0).divide(ee.Image(1.0).add(dI_N.multiply(dI_N).divide(k2)))
            cS = ee.Image(1.0).divide(ee.Image(1.0).add(dI_S.multiply(dI_S).divide(k2)))

            img = img.add(
                ee.Image(lamb).multiply(
                    cN.multiply(dI_N)
                    .add(cS.multiply(dI_S))
                    .add(cE.multiply(dI_E))
                    .add(cW.multiply(dI_W))
                )
            )

    return img


def make_boxcar(radius: int = 1):
    return ee.Kernel.square(1)
