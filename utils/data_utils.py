"""
The modules provides for utility functions with respect to data from the prepared datasets
"""

# Import libraries
import glob
import tqdm

import math
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset

from osgeo import gdal
from scipy.ndimage import rotate

UMP = ["AverageHeightArea", 
            "AverageHeightBuilding", 
            "AverageHeightTotalArea", 
            "Displacement", 
            "FrontalAreaIndex",
            "MaximumHeight",
            "PercentileHeight",
            "PlanarAreaIndex",
            "RoughnessLength",
            "StandardDeviation"]

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

def plotUMP(gdf, band_names= UMP, n_col= 2, scale= None):
    """
    Plot all the UMPs on a Map

    # Parameters\n
    - gdf: GeoDataFrame\n
    - band_names: Used for labelling the plots, if any
    - n_col: The number of columns (for presentation purposes)
    - scale: List[(number, number)] that denotes the v_min and v_max of each UMP
    """
    scale = [(None, None) for _ in range(len(band_names))] if (not scale) else scale
    ump_count = 0
    n_ump = len(band_names) # -1 for the geometry column
    fig, axs = plt.subplots(math.ceil(n_ump/n_col), n_col, figsize= (n_col*5, math.ceil(n_ump/n_col)*6))
    for col in range(math.ceil(n_ump/n_col)):
        for row in range(n_col):
            # f = plt.subplot(5, 2, ump_count+1)
            if math.ceil(n_ump/n_col) > 1 and n_col > 1:
                f_img = gdf.plot(column= band_names[ump_count], alpha= 0.5, ax= axs[col, row], legend= True, vmin= scale[ump_count][0], vmax= scale[ump_count][1]) # Note row/col here is wrong should be flipped
            elif n_col == 1:
                f_img = gdf.plot(column= band_names[ump_count], alpha= 0.5, ax= axs[col], legend= True, vmin= scale[ump_count][0], vmax= scale[ump_count][1])
            else:
                f_img = gdf.plot(column= band_names[ump_count], alpha= 0.5, ax= axs[row], legend= True, vmin= scale[ump_count][0], vmax= scale[ump_count][1])
            f_img.set_title(band_names[ump_count])
            # plt.title(band_names[ump_count])
            ump_count += 1
    plt.show()


class UMPDataset(Dataset):
    """Urban Morphological Parameters - Sentinel Dataset"""

    def __init__(self, ump_df, dir_path, tgt_ump, file_ext= ".tiff", transform= None, return_coords= False):
        """
        # Parameters:
            - ump_df: DataFrame containing the geometry of each cell with its corresponding UMPs.
            - dir_path: Directory in which all tiff files will be matched with it's corresponding cells in ump_df.
                - No trailing slashes for dir. eg. "data/osaka"
                - Naming convention for files: Sentinel_{'_'.join(map(str, map(int, cell.bounds)))}.tiff for cell in ump_df["geometry"]
            - tgt_ump: List of UMPs that should correspond to the column names of ump_df
            - file_ext: Extension of the file eg. .tiff / .tif
        """
        self.tgt_ump = tgt_ump
        self.ump_df = ump_df
        self.dir_path = dir_path
        self.file_ext = file_ext
        # Look in both current and sub-directories
        self.files = glob.glob(f"{self.dir_path}/**/*{self.file_ext}") + glob.glob(f"{self.dir_path}/*{self.file_ext}")
        self.transform = transform
        self.return_coords = return_coords

        # Generate the max and min for each channel and each UMP
        self.channel_max = [0 for _ in range(12)]
        # Using the roundabout way because there seems to be a bug with iterating directly
        for entry in tqdm.tqdm(range(len(self)), desc= "Checking through image files"):
            data_piece = self[entry]
            image = data_piece[0]
            UMPs = data_piece[1]
            for channel in range(len(image)):
                # The image
                cur_max = image[channel].max()
                if cur_max > self.channel_max[channel]:
                    self.channel_max[channel] = cur_max
        
        # Generate the max for each UMP
        self.UMP_max = [0 for _ in range(len(self.tgt_ump))]
        for param in range(len(self.tgt_ump)):
            param_stat = dict(self.ump_df[self.tgt_ump[param]].describe())
            self.UMP_max[param] = param_stat["max"]


            # for ump in range(len(UMPs)):
            #     cur_max = UMPs[ump]
            #     if cur_max > self.UMP_max[ump]:
            #         self.UMP_max[ump] = cur_max        
    
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

        # if image.shape != (12, 90, 90):
        #     print("Non-standard image shape:", image.shape)
        #     if image.shape[0] != 12:
        #         raise ValueError(f"Incorrect number of channels, {image.shape[0]} channels present when 12 expected")
        
        rot = self.ump_df["Rotation"][idx]
        image = rotate(image, rot, axes= (2, 1)) # axes is 2, 1 cause axis 0 is the batch

        umps = np.array(self.ump_df.iloc[idx][self.tgt_ump].astype("f"))
        
        if self.return_coords:
            out = [image, umps, np.array(self.ump_df["geometry"][idx].bounds)]
        else:
            out = [image, umps]

        if self.transform:
            out = self.transform(out)
        
        return out

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return map(self.__getitem__, range(self.__len__()))

        per_worker = int(math.ceil((self.__len__()) / float(worker_info.num_workers)))
        worker_id = worker_info.id
        iter_start = worker_id * per_worker
        iter_end = min(iter_start + per_worker, self.__len__())
        return map(self.__getitem__, range(iter_start, iter_end))

    def set_UMPs(self, tgt_umps):
        """
        Changes the output UMP of the dataset
        # Parameters:
        - List[str] tgt_umps: List of strings, corresponding to the columns in UMPDataset.ump_df 
        """
        self.tgt_ump = tgt_umps
        self.UMP_max = [0 for _ in range(len(self.tgt_ump))]
        for param in range(len(self.tgt_ump)):
            param_stat = dict(self.ump_df[self.tgt_ump[param]].describe())
            self.UMP_max[param] = param_stat["max"]

        

