"""
The modules provides for utility functions with respect to data from the prepared datasets
"""

# Import libraries
import glob

import math
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset

from osgeo import gdal


def plotArray(arr, n_channels= None, 
band_names= [
        'B1: Aerosols',
        'B2: Blue',
        'B3: Green',
        'B4: Red',
        'B5: Red Edge 1',
        'B6: Red Edge 2',
        'B7: Red Edge 3',
        'B8: NIR',
        'B8A: Red Edge 4',
        'B9: Water Vapor',
        'B11: SWIR 1',
        'B12: SWIR 2'
    ]):
    """
    Plot the all the channels in the 3D array (C x H x W)

    # Parameters\n
    - arr: np array\n
    - n_channels: The first n channels will be plotted, if None then plots all
    - band_names: Used for labelling the plots, if any
    
    """
    # Visualise all
    if n_channels is None or n_channels > arr.shape[0]:
        n_channels = arr.shape[0]

    band_count = 0

    fig, axs = plt.subplots(math.ceil(n_channels/2), 2, figsize= (8, math.ceil(n_channels/2)*4))
    for col in range(math.ceil(n_channels/2)):
        for row in range(2):
            f = plt.subplot(math.ceil(n_channels/2), 2, band_count+1)
            f_img = plt.imshow(arr[band_count, :, :])

            plt.title(band_names[band_count])
            band_count += 1
            # fig.colorbar(f_img, ax= axs[col, row])
            plt.colorbar()
    plt.show()


class UMPDataset(Dataset):
    """Urban Morphological Parameters - Sentinel Dataset"""

    def __init__(self, ump_df, dir_path, file_ext= ".tiff", transform= None):
        """
        # Parameters:
            - ump_df: DataFrame containing the geometry of each cell with its corresponding UMPs.
            - dir_path: Directory in which all tiff files will be matched with it's corresponding cells in ump_df.
                - No trailing slashes for dir. eg. "data/osaka"
                - Naming convention for files: Sentinel_{'_'.join(map(str, map(int, cell.bounds)))}.tiff for cell in ump_df["geometry"]
            - file_ext: Extension of the file eg. .tiff / .tif
        """
        self.ump_df = ump_df
        self.dir_path = dir_path
        self.file_ext = file_ext
        # Look in both current and sub-directories
        self.files = glob.glob(f"{self.dir_path}/**/*{self.file_ext}") + glob.glob(f"{self.dir_path}/*{self.file_ext}")
        self.transform = transform

        # Generate the max and min for each channel and each UMP
        self.channel_max = [0 for _ in range(12)]
        self.UMP_max = [0 for _ in range(10)]
        # Using the roundabout way because there seems to be a bug with iterating directly
        for entry in range(len(self)):
            data_piece = self[entry]
            image = data_piece[0]
            UMPs = data_piece[1]
            for channel in range(len(image)):
                # The image
                cur_max = image[channel].max()
                if cur_max > self.channel_max[channel]:
                    self.channel_max[channel] = cur_max
            
            for ump in range(len(UMPs)):
                cur_max = UMPs[ump]
                if cur_max > self.UMP_max[ump]:
                    self.UMP_max[ump] = cur_max        
    
    def __len__(self):
        return len(self.ump_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        poly = self.ump_df["geometry"][idx]
        target = f"Sentinel_{'_'.join(map(str, map(int, poly.bounds)))}.tiff"
        # Finds and return the first match in a string
        file_name = next((s for s in self.files if target in s), None)
        if file_name is None:
            raise KeyError(f"{target} not found in {self.dir_path}")
        
        image = gdal.Open(file_name, gdal.GA_ReadOnly).ReadAsArray().astype("f")

        if image.shape != (12, 90, 90):
            print("Non-standard image shape:", image.shape)
            if image.shape[0] != 12:
                raise ValueError(f"Incorrect number of channels, {image.shape[0]} channels present when 12 expected")
        
        umps = np.array(self.ump_df.iloc[idx][[
            "AverageHeightArea", 
            "AverageHeightBuilding", 
            "AverageHeightTotalArea", 
            "Displacement", 
            "FrontalAreaIndex",
            "MaximumHeight",
            "PercentileHeight",
            "PlanarAreaIndex",
            "RoughnessLength",
            "StandardDeviation",
        ]].astype("f"))
        
        out = [image, umps]

        if self.transform:
            out = self.transform(out)
        
        return out
        

