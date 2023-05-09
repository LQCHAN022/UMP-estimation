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
import argparse

from osgeo import gdal
import geopandas as gpd
import shapely
from scipy.ndimage import rotate

from torch.utils.tensorboard import SummaryWriter
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

parser = argparse.ArgumentParser()

# parser.add_argument("--dataset", type= str, default= "Tokyo_Osaka", help= "<Train>_<Valid>, right now the available options are Tokyo_Osaka, Tokyo_Tokyo")
# parser.add_argument("--model", type= str, default= "12ch_light", help= "Available options: 3ch, 12ch_light, 12ch_mod, 12ch_full")
parser.add_argument("--epoch", type= int, default= 50)
parser.add_argument("--batchsize", type= int, default= 64)
parser.add_argument("--name", type= str, default= "default", help= "Name of the model/setup")
parser.add_argument("--extend", type= int, default= 5, help= "Extends the max epoch by the value if the current best is the last epoch")
parser.add_argument("--patience", type= int, default= 5, help= "Stops the training when vloss increases for <patience> consecutive times, min=0")
# parser.add_argument("--root", type= str, default= "overnight_results", help= "Place to store results, ie. Weights and plots")

args = parser.parse_args()
print(args)

####### Change the sigmoid_idx accordingly #######

# Constants
CHECKPT_PATH = "pretrained_model/im2elevation/Block0_skip_model_110.pth.tar"
UMP = [
        # "AverageHeightArea", 
        # "AverageHeightBuilding", 
        # "AverageHeightTotalArea", 
        # "Displacement", 
        "FrontalAreaIndex",
        # "MaximumHeight",
        # "PercentileHeight",
        # "PlanarAreaIndex",
        # "RoughnessLength",
        # "StandardDeviation"
        ]

with open("data/ds_tokyo_3.pkl", "rb") as f:
    ds_tokyo = pickle.load(f)

with open("data/ds_osaka_3.pkl", "rb") as f:
    ds_osaka = pickle.load(f)

ds_tokyo.set_UMPs(UMP)
ds_osaka.set_UMPs(UMP)

dl = DataLoaders().from_dsets(ds_tokyo, ds_osaka, bs= args.batchsize, device=torch.device('cuda'))

# Load weights from IM2ELEVATION and delete unnecessary layers
checkpoint = torch.load(CHECKPT_PATH) # The original IM2ELEVATION weights
# checkpoint = torch.load("trained_models/model_customhead_w_64_40.pth")

to_delete = []
# for layer in checkpoint.keys():
for layer in checkpoint["state_dict"].keys():
    if any([word in layer.upper() for word in ["HARM", "R.CONV4", "R.BN4", "R2"]]):
    # if any([word in layer.upper() for word in ["HARM", "R.CONV3", "R.BN3", "R.CONV4", "R.BN4"]]):
        to_delete.append(layer)
# print(to_delete)

for i in to_delete:
    # checkpoint.pop(i)
    checkpoint["state_dict"].pop(i)

# Load Weights
ssl._create_default_https_context = ssl._create_unverified_context

original_model = senet.senet154()

Encoder = modules.E_senet(original_model, dl.train_ds.channel_max) # For new ds
# Encoder = modules.E_senet(original_model, old_ds_channel_max) # For old ds
# model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

# model = net.model_n12(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
model = net.model_n12_light(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048], n_out= len(UMP), sigmoid_idx= [0])

# Load weights
model.load_state_dict(checkpoint["state_dict"], strict=False)

# Clear memory
del checkpoint
gc.collect()
torch.cuda.empty_cache()

def mse_weighted(pred, actual, UMP_max= dl.train_ds.UMP_max):
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
def AverageHeightArea_RMSE(pred, actual, idx):
    return math.sqrt(F.mse_loss(pred[:, idx], actual[:, idx]))

def AverageHeightBuilding_RMSE(pred, actual, idx):
    return math.sqrt(F.mse_loss(pred[:, idx], actual[:, idx]))

def AverageHeightTotalArea_RMSE(pred, actual, idx):
    return math.sqrt(F.mse_loss(pred[:, idx], actual[:, idx]))

def Displacement_RMSE(pred, actual, idx):
    return math.sqrt(F.mse_loss(pred[:, idx], actual[:, idx]))

def FrontalAreaIndex_RMSE(pred, actual, idx):
    return math.sqrt(F.mse_loss(pred[:, idx], actual[:, idx]))

def MaximumHeight_RMSE(pred, actual, idx):
    return math.sqrt(F.mse_loss(pred[:, idx], actual[:, idx]))

def PercentileHeight_RMSE(pred, actual, idx):
    return math.sqrt(F.mse_loss(pred[:, idx], actual[:, idx]))

def PlanarAreaIndex_RMSE(pred, actual, idx):
    return math.sqrt(F.mse_loss(pred[:, idx], actual[:, idx]))

def RoughnessLength_RMSE(pred, actual, idx):
    return math.sqrt(F.mse_loss(pred[:, idx], actual[:, idx]))

def StandardDeviation_RMSE(pred, actual, idx):
    return math.sqrt(F.mse_loss(pred[:, idx], actual[:, idx]))

metrics = [
    # partial(AverageHeightArea_RMSE, idx= 0), 
    # partial(AverageHeightBuilding_RMSE, idx= 1),
    # partial(AverageHeightTotalArea_RMSE, idx= 0),
    # partial(Displacement_RMSE, idx= 0),
    partial(FrontalAreaIndex_RMSE, idx= 0),
    # partial(MaximumHeight_RMSE, idx= 4),
    # partial(PercentileHeight_RMSE, idx= 5),
    # partial(PlanarAreaIndex_RMSE, idx= 2),
    # partial(RoughnessLength_RMSE, idx= 1),
    # partial(StandardDeviation_RMSE, idx= 3),
]


from torch.cuda.amp import GradScaler
# Optimizers specified in the torch.optim package
model.train()
model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# Creates a GradScaler once at the beginning of training.
scaler = GradScaler()
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(tqdm.tqdm(dl.train, desc= f"Epoch: {epoch_index + 1}", dynamic_ncols= True)):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        with torch.autocast(device_type= "cuda", dtype= torch.float16):
            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = mse_weighted(outputs, labels, dl.train_ds.UMP_max)

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()
        # loss.backward()

        # Adjust learning weights
        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)
        # optimizer.step()

        # Updates the scale for next iteration.
        scaler.update()

        # Gather data and report
        running_loss += loss.detach().item()
        last_loss = running_loss / (i + 1) # loss per batch (avg loss)
        tb_x = epoch_index * len(dl[0]) + i + 1
        tb_writer.add_scalar('Loss/train', last_loss, tb_x)

        tqdm.tqdm.write(f"Running Loss: {last_loss}", end= "\r")

    # print(f'Epoch {} loss: {}')
    return last_loss

# PyTorch TensorBoard support
# Initializing in a separate cell so we can easily add more epochs to the same run
name = args.name
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/sentinel_{}_{}'.format(name, timestamp))
epoch_number = 0

EPOCHS = args.epoch

best_vloss = 1_000_000.

# Early Stopping
prev_vloss = 1_000_000.
patience = 0


while epoch_number < EPOCHS:
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    # model.train(True)
    # model.cuda()
    avg_loss = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    model.train(False)

    # Initialise loss and metrics
    running_vloss = 0.0
    val_metrics = {}
    for i, metric in enumerate(metrics):
        val_metrics[metric.func.__name__] = 0

    with torch.no_grad():
        for i, vdata in enumerate(dl[1]):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            # Calcualte the loss
            vloss = mse_weighted(voutputs, vlabels, dl.train_ds.UMP_max)
            running_vloss += vloss

            # Calculate the metrics
            for i, metric in enumerate(metrics):
                val_metrics[metric.func.__name__] += metric(voutputs, vlabels) / len(dl[1])

        

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Log the metrics as well
    writer.add_scalars('Validation Metrics',
                    val_metrics,
                    epoch_number + 1)
    writer.flush()

    print(val_metrics)


    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        # Save this to var instead of writing directly
        best_model = model.state_dict()
        best_epoch = epoch_number

        if epoch_number == EPOCHS - 1:
            EPOCHS += args.extend

    # If loss increase or plateau
    if avg_vloss > prev_vloss or abs(avg_loss - prev_vloss) < 0.01 * abs(prev_vloss):
        patience += 1
        if patience > args.patience:
            print(f"Early Stopping at Epoch {epoch_number} with {patience} consecutive vloss increase or plateau")
            break
    else:
        patience = 0
    prev_vloss = avg_vloss

    epoch_number += 1

# Save model
model_path = 'overnight_results/model_{}_{}_{}'.format(name, timestamp, best_epoch)
torch.save(best_model, model_path)