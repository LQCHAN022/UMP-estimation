import ee
from shapely.geometry.polygon import Polygon

from dl_landsat import create_bounding_box

ee.Initialize()
import geemap
import json
from ipyleaflet import GeoJSON, Marker, MarkerCluster
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from skimage import exposure
import cv2
import os

from osgeo import gdal
import os
import glob

from terracatalogueclient import Catalogue


def create_projection_window(bounding_box):
    coords = np.asarray(bounding_box.getInfo()["coordinates"])[0]
    lat_max = np.max(coords[:, 0])
    lat_min = np.min(coords[:, 0])

    lon_max = np.max(coords[:, 1])
    lon_min = np.min(coords[:, 1])
    return [lat_min, lon_max, lat_max, lon_min]


def main(coordinates, base_dir, raw_data_dir):
    image = np.load(raw_data_dir)
    print(image.shape)
    point = ee.Geometry.Point(coordinates)
    bounds = create_bounding_box(point, 20000)



    geometry = tuple([tuple(x) for x in bounds.coordinates().getInfo()[0]])
    geometry = Polygon(geometry)
    #
    catalogue = Catalogue().authenticate()

    products = catalogue.get_products("urn:eop:VITO:ESA_WorldCover_10m_2020_V1", geometry=geometry)
    catalogue.download_products(products, f"{base_dir}")
    file_list = glob.glob(f"{base_dir}/*/*Map.tif")

    proj_window = create_projection_window(bounds)
    my_vrt = gdal.BuildVRT(f'{base_dir}/combined.vrt', file_list)
    ds = gdal.Translate(f'{base_dir}/worldcover.vrt', my_vrt, projWin=proj_window,
                        width=image.shape[2],
                        height=image.shape[3])
    ground_truth_data = ds.ReadAsArray()
    np.save(f"{base_dir}/worldcover", ground_truth_data)

    GROUND_TRUTH_COMBINATION = {
        10: 10,
        20: 10,
        30: 10,
        90: 10,
        95: 10,
        100: 10,

        60: 40,
        40: 40,

        70: 80,
        80: 80,

        50: 50

    }

    for k, v in GROUND_TRUTH_COMBINATION.items():
        ground_truth_data[ground_truth_data == k] = v
    np.save(F"{base_dir}/worldcover_adj_classes", ground_truth_data)
    print(ground_truth_data.shape)
    plt.matshow(ground_truth_data)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    coords = [36.74905523581975, -1.2815372605877613]
    main(coords, 'data/nairobi_images_summer', 'data/nairobi_images_summer/data_raw.npy')
