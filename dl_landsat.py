import os

import ee
import geemap
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure

LANDSAT_5 = "LANDSAT/LT05/C02/T1_L2"
LANDSAT_5_START = 1984
LANDSAT_5_END = 2000

LANDSAT_7 = "LANDSAT/LE07/C02/T1_L2"
LANDSAT_7_START = 2000
LANDSAT_7_END = 2014

LANDSAT_8 = "LANDSAT/LC08/C02/T1_L2"
LANDSAT_8_START = 2014
LANDSAT_8_END = 2022


def apply_scale_factors_57(image):
    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBand = image.select('ST_B6').multiply(0.00341802).add(149.0)

    image = (image.addBands(opticalBands, None, True)
             .addBands(thermalBand, None, True))
    image = (image.select(["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "ST_B6", "QA_PIXEL"])
             .rename(["B1", "B2", "B3", "B4", "B5", "B7", "ST", "QA_PIXEL"]))
    return image


def apply_scale_factors_8(image):
    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBand = image.select('ST_B.*').multiply(0.00341802).add(149.0)

    image = (image.addBands(opticalBands, None, True)
             .addBands(thermalBand, None, True))
    image = (image.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10", "QA_PIXEL"])
             .rename(["B1", "B2", "B3", "B4", "B5", "B7", "ST", "QA_PIXEL"]))
    return image


def mask_clouds(image):
    qa = image.select("QA_PIXEL")

    mask = (qa.bitwiseAnd(1 << 3).And(qa.bitwiseAnd(1 << 9))
            .Or(qa.bitwiseAnd(1 << 4).And(qa.bitwiseAnd(1 << 11)))
            .Or(qa.bitwiseAnd(1 << 5).And(qa.bitwiseAnd(1 << 13)))
            )

    image = image.updateMask(mask.Not())

    return image


def create_timelaps_data(point, years, bands=None, verbose=True):
    if bands is None:
        bands = ["B1", "B2", "B3", "B4", "B5", "B7"]
    resulting_images = []
    for year in years:
        if verbose:
            print(f"At Year: {year}")
        bounding_box = create_bounding_box(point, 15300)  # We cant export more ATM
        ls_read = (ee.ImageCollection(LANDSAT_8)
                   .map(apply_scale_factors_8)
                   .map(mask_clouds)
                   .filterBounds(point)
                   .filterDate(f'{year}-01-01', f'{year}-12-31')
                   .map(mask_clouds)
                   )
        ls_median = ls_read.median()
        ls_read_export = ls_median.reproject(ls_read.first().select("B1").projection())
        image = geemap.ee_to_numpy(ls_read_export, region=bounding_box, bands=bands, default_value=0.0)

        if verbose:
            img = np.dstack([image[:, :, 2], image[:, :, 1], image[:, :, 0]])
            img = exposure.equalize_adapthist(img, clip_limit=0.02)
            # plt.imshow(img)
            # plt.show()
        resulting_images.append(image)
    return np.asarray(resulting_images)


def create_bounding_box(point: ee.Geometry.Point, diameter: float):
    intermediate_circle = point.buffer(diameter / 2.0, 0.0)
    return intermediate_circle.bounds()


def create_center_bounding_boxes(lat, lon, diameter, r_earth=6371.0):
    d_scale = diameter * 0.25
    bounding_boxes = []
    for dx, dy in [(1, -1), (1, 1), (-1, 1), (-1, -1)]:
        dx = dx * d_scale
        dy = dy * d_scale
        p_lat = lat + (dy / r_earth) * (180 / np.pi)
        p_lon = lon + (dx / r_earth) * ((180 / np.pi) / np.cos(lat * (np.pi / 180)))

        point = ee.Geometry.Point([p_lon, p_lat])
        bb = create_bounding_box(point, diameter / 2.0)
        bounding_boxes.append(bb)
    return bounding_boxes


def main(city, years=None, base_path=None, auth=False):
    # Trigger the authentication flow.
    if auth:
        ee.Authenticate()

    # Initialize the library.
    ee.Initialize()

    city = ee.Geometry.Point(city)  # Coord format: E, N
    if years is None:
        years = list(range(2013, 2022))
    if base_path is None:
        base_path = "data/city_dataset/"
    if not os.path.isdir(base_path):
        os.makedirs(base_path)

    data_set = create_timelaps_data(city, years)
    print(data_set.shape)
    print(base_path)
    np.save(f"{base_path}/data_raw", data_set)
    np.save(f"{base_path}/train_raw", data_set[:-1])
    np.save(f"{base_path}/test_raw", data_set[-1:])


# Pohang-si: 129.3145, 36.0030 (Works)
# Bangkok: 100.5018, 13.7563 (Sadly Poor data in 1984)
# Hanoi: 105.8342 21.0278
# Shenzhen: 114.0596, 22.5429
# HongKong: 114.1694, 22.3193

if __name__ == '__main__':
    point = [129.3145, 36.0030]
    main(point, base_path="data/pohang_si_dataset")
