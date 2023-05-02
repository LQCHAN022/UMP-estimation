# Import of necessary libraries

import os
os.environ['USE_PYGEOS'] = '0'
import sys
import argparse
from datetime import datetime
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
from scipy.ndimage import rotate

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
from functools import partial

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

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type= str, default= "Tokyo_Osaka", help= "<Train>_<Valid>, right now the available options are Tokyo_Osaka, Tokyo_Tokyo")
parser.add_argument("--model", type= str, default= "12ch_light", help= "Available options: 3ch, 12ch_light, 12ch_mod, 12ch_full")
parser.add_argument("--epoch", type= int, default= 10)
parser.add_argument("--batchsize", type= int, default= 64)
parser.add_argument("--root", type= str, default= "overnight_results", help= "Place to store results, ie. Weights and plots")

args = parser.parse_args()


def mse_weighted(pred, actual, UMP_max):
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

def main():
# Prepare datasets

    with open("data/ds_tokyo.pkl", "rb") as f:
        ds_tokyo = pickle.load(f)

    with open("data/ds_osaka.pkl", "rb") as f:
        ds_osaka = pickle.load(f)

    with open("data/ds_tokyo_distinct.pkl", "rb") as f:
        ds_tokyo_distinct = pickle.load(f)

    # Create dataloader based on arg and batchsize
    if args.dataset == "Toyko_Osaka":
        dl = DataLoaders().from_dsets(ds_tokyo, ds_osaka, bs= args.batchsize, device=torch.device('cuda'))
    elif args.dataset == "Tokyo_Tokyo":
        dl = DataLoaders().from_dsets(*torch.utils.data.random_split(ds_tokyo_distinct, [0.8, 0.2]), bs= args.batchsize, device=torch.device('cuda'))
    # Not going to vary for now since both are trained on tokyo
    channel_max = ds_tokyo.channel_max
    UMP_max = ds_tokyo.UMP_max


    # Instantiate model

    # Load weights from IM2ELEVATION and delete unnecessary layers
    checkpoint = torch.load(CHECKPT_PATH) # The original IM2ELEVATION weights
    to_delete = []
    for layer in checkpoint["state_dict"].keys():
        if any([word in layer.upper() for word in ["HARM", "R.CONV4", "R.BN4", "R2"]]):
            to_delete.append(layer)
    for i in to_delete:
        checkpoint["state_dict"].pop(i)

    # Load Weights
    ssl._create_default_https_context = ssl._create_unverified_context
    original_model = senet.senet154()
    Encoder = modules.E_senet(original_model, channel_max) # For new ds

    ## 3ch, 12ch_light, 12ch_mod, 12ch_full
    ### 3ch
    # Gonna assume that input dataloaders are all 12 channels
    if args.model == "3ch":
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048], slice_input= True)
    ### 12ch_light
    elif args.model == "12ch_light":
        model = net.model_n12_light(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    elif args.model == "12ch_mod":
        Encoder.channels = list(range(12))
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
        # Modify weights
        conv1 = next(next(next(next(model.children()).children()).children()).children())
        in_chans = 12
        conv1_weight = conv1.weight.float()
        conv1_type = conv1_weight.dtype
        repeat = int(math.ceil(in_chans / 3))
        conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
        conv1_weight *= (3 / float(in_chans))
        conv1_weight = conv1_weight.to(conv1_type)
        checkpoint["state_dict"]["E.base.0.conv1.weight"] = conv1_weight
        # Modify channel size
        old_weight = conv1.weight.data
        new_weight = torch.zeros((64, 12, 3, 3))
        new_weight[:, :old_weight.shape[1], :, :] = old_weight
        conv1.weight.data = new_weight
    elif args.model == "12ch_full":
        model = net.model_n12(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    else:
        print(f"Model {args.model} does not exist, please choose from [3ch, 12ch_light, 12ch_mod, 12ch_full]")
        return
        
    # Load weights
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    # Clear memory
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    # Define loss function based on the dataset (since Y changes)
    loss_func = partial(mse_weighted, UMP_max= UMP_max)

    learn = Learner(dl, model, loss_func= loss_func, metrics= metrics, cbs=[MixedPrecision, FP16TestCallback])

    learn.fine_tune(args.epoch, 0.001, freeze_epochs= 1, cbs= [ShowGraphCallback()])

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M")
    name = f"{args.model}_{args.dataset}_{args.batchsize}_{args.epoch}"

    # Save the weights
    torch.save(model.state_dict(), f"{args.root}/{name}.pth")

    # Save the results
    with open(f"{args.root}/recorder_{name}.pkl", "wb") as f:
        pickle.dump(learn.recorder, f)


if __name__ == "__main__":
    main()