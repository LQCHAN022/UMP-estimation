# Ensures runtime code is updated when source code of libraries are updated as well

# Import of necessary libraries

import os
os.environ['USE_PYGEOS'] = '0'
import pandas as pd
import numpy as np
import pickle
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import gc
import ssl
import tqdm

from osgeo import gdal
import geopandas as gpd
import shapely

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.preprocessing import Normalizer

# FastAI
from fastai.vision.all import *
from fastai.text.all import *
from fastai.collab import *
from fastai.tabular.all import *

import utils.data_utils as du
import utils.sp_utils as sp


# User modules

from utils.models import senet
from utils.models import modules
from utils.models import net
# Check if gpu/cuda is available
import torch
torch.cuda.is_available()
# Constants
CHECKPT_PATH = "pretrained_model/im2elevation/Block0_skip_model_110.pth.tar"
UMP = ["AverageHeightArea", 
            "AverageHeightBuilding", 
            "AverageHeightTotalArea", 
            # "Displacement", 
            "FrontalAreaIndex",
            "MaximumHeight",
            "PercentileHeight",
            "PlanarAreaIndex",
            # "RoughnessLength",
            "StandardDeviation"]

## Build Dataset
Y_tokyo = gpd.read_feather("data/Y_UMP/Y_tokyo_4.feather")
ds_tokyo = du.UMPDataset(Y_tokyo, "data/X_sentinel/tokyo")
Y_osaka = gpd.read_feather("data/Y_UMP/Y_osaka_4.feather")
ds_osaka = du.UMPDataset(Y_osaka, "data/X_sentinel/osaka")
ds_osaka[0][0].dtype, ds_osaka[0][1].dtype
#### Old Dataset
with open("data/x_train_tokyo.pkl", "rb") as f:
    old_x_train = pickle.load(f)

with open("data/x_val_tokyo.pkl", "rb") as f:
    old_x_val = pickle.load(f)

with open("data/y_train_tokyo.pkl", "rb") as f:
    old_y_train = pickle.load(f)

with open("data/y_val_tokyo.pkl", "rb") as f:
    old_y_val = pickle.load(f)

# Reorder old_y_
old_y_train = old_y_train[:, [1, 0, 2, 7, 4, 5, 6, 3]]
old_y_val = old_y_val[:, [1, 0, 2, 7, 4, 5, 6, 3]]
## Old
old_dls = DataLoaders().from_dsets(list(zip(old_x_train, old_y_train)), list(zip(old_x_val, old_y_val)), bs= 8, device=torch.device('cuda'))
# Adapting the old dataset to the new dataset format
# Generate the max and min for each channel and each UMP

old_ds_channel_max = [0 for _ in range(12)]
old_ds_UMP_max = [0 for _ in range(8)]
# Using the roundabout way because there seems to be a bug with iterating directly
for entry in range(len(old_x_train)):
    image = old_x_train[entry]
    UMPs = old_y_train[entry]
    for channel in range(len(image)):
        # The image
        cur_max = image[channel].max()
        if cur_max > old_ds_channel_max[channel]:
            old_ds_channel_max[channel] = cur_max
    
    for ump in range(len(UMPs)):
        cur_max = UMPs[ump]
        if cur_max > old_ds_UMP_max[ump]:
            old_ds_UMP_max[ump] = cur_max

old_ds_channel_max, old_ds_UMP_max

dl = DataLoaders().from_dsets(ds_tokyo, ds_osaka, bs= 8, device=torch.device('cuda'))

# Load weights from IM2ELEVATION and delete unnecessary layers
checkpoint = torch.load(CHECKPT_PATH) # The original IM2ELEVATION weights
# checkpoint = torch.load("trained_models/model_customhead_w_64_40.pth")

to_delete = []
# for layer in checkpoint.keys():
for layer in checkpoint["state_dict"].keys():
    if any([word in layer.upper() for word in ["HARM", "R.CONV4", "R.BN4", "R2"]]):
    # if any([word in layer.upper() for word in ["HARM", "R.CONV3", "R.BN3", "R.CONV4", "R.BN4"]]):
        to_delete.append(layer)
print(to_delete)

for i in to_delete:
    # checkpoint.pop(i)
    checkpoint["state_dict"].pop(i)

# Load Weights
ssl._create_default_https_context = ssl._create_unverified_context

original_model = senet.senet154()

Encoder = modules.E_senet(original_model, dl.train_ds.channel_max) # For new ds
# Encoder = modules.E_senet(original_model, old_ds_channel_max) # For old ds
# model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

model = net.model_n12(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])



# Duplicate the weights for the encoders
for k in list(checkpoint["state_dict"].keys()):
    if "E." in k:
        for i in range(4):
            checkpoint["state_dict"][f"E{i}." + k[2:]] = checkpoint["state_dict"][k]
# Load weights
model.load_state_dict(checkpoint["state_dict"], strict=False)

# Clear memory
del checkpoint
gc.collect()
torch.cuda.empty_cache()

# Loss Function

def mse_weighted(pred, actual, UMP_max= old_ds_UMP_max):
# def mse_weighted(pred, actual, UMP_max= dl.train_ds.UMP_max):
    """
    Weighted loss function that normalises the predictions based on the parameters used to normalise the actual during training
    """
    loss = tensor(0).float()
    loss.requires_grad_(True)
    for ump in range(pred.shape[1]):
        # loss = torch.add(loss, F.mse_loss(pred[:, ump], actual[:, ump]))
        loss = torch.add(loss, torch.div(F.mse_loss(pred[:, ump], actual[:, ump]), UMP_max[ump]**2))
    if loss.isnan().sum() > 1:
        raise ValueError([pred, actual])
    return loss.float()

# List of metrics
"""
"AverageHeightArea", 
"AverageHeightBuilding", 
"AverageHeightTotalArea", 
"Displacement", 
"FrontalAreaIndex",
"MaximumHeight",
"PercentileHeight",
"PlanarAreaIndex",
"RoughnessLength",
"StandardDeviation"
"""
def AverageHeightArea_RMSE(pred, actual):
    return math.sqrt(F.mse_loss(pred[:, 0], actual[:, 0]))

def AverageHeightBuilding_RMSE(pred, actual):
    return math.sqrt(F.mse_loss(pred[:, 1], actual[:, 1]))

def AverageHeightTotalArea_RMSE(pred, actual):
    return math.sqrt(F.mse_loss(pred[:, 2], actual[:, 2]))

# def Displacement_RMSE(pred, actual):
#     return math.sqrt(F.mse_loss(pred[:, 3], actual[:, 3]))

def FrontalAreaIndex_RMSE(pred, actual):
    return math.sqrt(F.mse_loss(pred[:, 3], actual[:, 3]))
    # return math.sqrt(F.mse_loss(pred[:, 4], actual[:, 4]))

def MaximumHeight_RMSE(pred, actual):
    return math.sqrt(F.mse_loss(pred[:, 4], actual[:, 4]))
    # return math.sqrt(F.mse_loss(pred[:, 5], actual[:, 5]))

def PercentileHeight_RMSE(pred, actual):
    return math.sqrt(F.mse_loss(pred[:, 5], actual[:, 5]))
    # return math.sqrt(F.mse_loss(pred[:, 6], actual[:, 6]))

def PlanarAreaIndex_RMSE(pred, actual):
    return math.sqrt(F.mse_loss(pred[:, 6], actual[:, 6]))
    # return math.sqrt(F.mse_loss(pred[:, 7], actual[:, 7]))

# def RoughnessLength_RMSE(pred, actual):
#     return math.sqrt(F.mse_loss(pred[:, 8], actual[:, 8]))

def StandardDeviation_RMSE(pred, actual):
    return math.sqrt(F.mse_loss(pred[:, 7], actual[:, 7]))
    # return math.sqrt(F.mse_loss(pred[:, 9], actual[:, 9]))

metrics = [
    AverageHeightArea_RMSE, 
    AverageHeightBuilding_RMSE,
    AverageHeightTotalArea_RMSE,
    # Displacement_RMSE,
    FrontalAreaIndex_RMSE,
    MaximumHeight_RMSE,
    PercentileHeight_RMSE,
    PlanarAreaIndex_RMSE,
    # RoughnessLength_RMSE,
    StandardDeviation_RMSE
]
### Train Model
# Need better loss due to scale difference
model.train()
learn = Learner(old_dls, model, loss_func= mse_weighted, metrics= metrics, cbs=[MixedPrecision, FP16TestCallback])
# learn = Learner(dl, model, loss_func= mse_weighted, metrics= metrics, cbs=[MixedPrecision, FP16TestCallback])
learn.fine_tune(5, 0.001, freeze_epochs= 1, cbs= [ShowGraphCallback()])
# Plateaus around 30
name = "v5_oldds_ump_cut_12ch_8_30" # <description>_<batch_size>_<epochs>
torch.save(model.state_dict(), f"trained_models/model_weight_{name}.pth")
# torch.save(model, f"trained_models/model_{name}.pth")

with open(f"trained_models/model_records_{name}.pkl", "wb") as f:
    pickle.dump(learn.recorder.values, f)