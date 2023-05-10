import ee
import rasterio
from rasterio.merge import merge

import os
import glob
from pathlib import Path
from natsort import natsorted

from abc import ABC, abstractmethod
import itertools
import numpy as np
from tqdm.auto import tqdm

from concurrent.futures import as_completed
from requests_futures.sessions import FuturesSession

# from utils import logger

try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()

class BaseDownloader(ABC):
    """
    Helper class used for downloading images from GEE
    """
    def __init__(self, root:str, dataset_name:str= "Sentinel"):
        """
        # Parameters
        - `str` root: The path to the folder to store the images
        - `str` dataset_name: The name of the dataset for identification/naming purposes of the files
        """
        self.set_root(root)
        self.dataset_name = dataset_name

    def set_root(self, root):
        self.root = root
        self.cache_dir = os.path.join(self.root, "cache")
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def merge_patches(self, base_filename):
        # retrieve all patches
        filenames = natsorted(glob.glob(os.path.join(self.cache_dir, f"{base_filename}_*.tiff")))
        patches = []
        for filename in filenames:
            patch = rasterio.open(filename)
            patches.append(patch)

        # merge
        mosaic, output = merge(patches)
        output_meta = patches[0].meta.copy()
        output_meta.update(
            {"driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": output,
            }
        )

        filename = f"{self.dataset_name}_{Path(filenames[0]).name.split('_')[-2]}.tiff"
        outfile = os.path.join(self.root, filename)
        with rasterio.open(outfile, "w", **output_meta) as f:
            f.write(mosaic)

        print(f"Successfully merge patches, saved at {outfile}")

    def download_image(self, image, initial_bounds, bands=None, scale=10, format="GEOTIFF", base_filename="dummy"):
        initial_x, initial_y, final_x, final_y = initial_bounds
        step = 0.05
        x_list = np.arange(initial_x, final_x, step)
        y_list = np.arange(initial_y, final_y, step)

        # generate urls for small image patches
        patch_bounds = list(itertools.product(x_list, y_list))
        patch_boxes = [[x, y, min(x + step, final_x), min(y + step, final_y)] for x, y in patch_bounds]
        save_params_list = [{
            'bands': image.bandNames().getInfo() if bands is None else bands,
            'region': ee.Geometry.BBox(*box),
            'scale': scale,
            'format': format
        } for box in patch_boxes]

        print("Downloading image patches...")
        session = FuturesSession(max_workers=20)
        futures = []
        for i, save_params in enumerate(save_params_list):
            url = image.getDownloadUrl(save_params)
            future = session.get(url)
            future.filename = os.path.join(self.cache_dir, f"{self.dataset_name}_{base_filename}_{i}.tiff")
            futures.append(future)

        for future in as_completed(futures):
            response = future.result()
            with open(future.filename, 'wb') as fd:
                fd.write(response.content)
            
            print(f"Downloaded and saved at {future.filename}")

        # merge
        self.merge_patches(f"{self.dataset_name}_{base_filename}")
