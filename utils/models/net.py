import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils.models import modules
from torchvision import utils
import math
import copy

import cv2
class im2elevation_model(nn.Module):
    """
    The original IM2ELEVATION model
    """
    def __init__(self, Encoder, num_features, block_channel):

        super(im2elevation_model, self).__init__()

        self.E = Encoder
        self.D2 = modules.D2(num_features = num_features)
        self.MFF = modules.MFF(block_channel)
        self.R = modules.R(block_channel)


    def forward(self, x):
        x_block0, x_block1, x_block2, x_block3, x_block4 = self.E(x)
  
        # x= x_block0.view(-1,250,250)

        # x = x.cpu().detach().numpy()
        
        #for idx in range(0,len(x)):
        #    x[idx] = x[idx]*100000
        #    np.clip(x[idx], 0, 50000).astype(np.uint16)
        #    filename = str(idx)+'.png'
        #    cv2.imwrite(filename, x[idx]) 
         
        
        x_decoder = self.D2(x_block0, x_block1, x_block2, x_block3, x_block4) 


        
        x_mff = self.MFF(x_block0, x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)]) 


 
        out = self.R(torch.cat((x_decoder, x_mff), 1)) 
        return out


class model(nn.Module):
    """
    Final model that aims to put a head to the IM2ELEVATION model to for UMP prediction
    """
    def __init__(self, Encoder, num_features, block_channel):
        """
        # Parameters:\n
        - body: The main body of the model, in this case generally refers to original IM2ELEVATION model\n
        - head: The head of the model, takes in output from the body and outputs it as the output of this model\n
        - cut: The index to cut the body after, ie. body = body[:cut]. Defaults to None. \n

        """

        super(model, self).__init__()
        
        self.E = Encoder
        self.D2 = modules.D2(num_features = num_features)
        self.MFF = modules.MFF(block_channel)
        self.R = modules.R1()
        self.R2 = modules.R2()


    def forward(self, x):

        x_block0, x_block1, x_block2, x_block3, x_block4 = self.E(x)

        x_decoder = self.D2(x_block0, x_block1, x_block2, x_block3, x_block4) 

        x_mff = self.MFF(x_block0, x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)]) 

        x_R1 = self.R(torch.cat((x_decoder, x_mff), 1))

        out = self.R2(x_R1) 

        return out


class model_n12(nn.Module):
    """
    Final model that aims to put a head to the IM2ELEVATION model to for UMP prediction
    """
    def __init__(self, Encoder, num_features, block_channel):
        """
        # Parameters:\n
        - body: The main body of the model, in this case generally refers to original IM2ELEVATION model\n
        - head: The head of the model, takes in output from the body and outputs it as the output of this model\n
        - cut: The index to cut the body after, ie. body = body[:cut]. Defaults to None. \n

        """

        super(model_n12, self).__init__()
        
        # self.E0, self.E1, self.E2, self.E3 = Encoders
        # Instantiate Encoders and set their channels
        self.E0 = Encoder
        self.E1 = copy.deepcopy(Encoder)
        self.E2 = copy.deepcopy(Encoder)
        self.E3 = copy.deepcopy(Encoder)

        # Set their respective channels for normalisation
        self.E0.set_channels(list(range(0, 3)))
        self.E1.set_channels(list(range(3, 6)))
        self.E2.set_channels(list(range(6, 9)))
        self.E3.set_channels(list(range(9, 12)))

        # Mergers for the different channels
        self.M0 = nn.Conv2d(256, 64, 1, 1)
        self.M1 = nn.Conv2d(1024, 256, 1, 1)
        self.M2 = nn.Conv2d(2048, 512, 1, 1)
        self.M3 = nn.Conv2d(4096, 1024, 1, 1)
        self.M4 = nn.Conv2d(8192, 2048, 1, 1)

        self.D2 = modules.D2(num_features = num_features)
        self.MFF = modules.MFF(block_channel)
        self.R = modules.R1()
        self.R2 = modules.R2()


    def forward(self, x):
        
        
        # First Encoder (Channels 1-3)
        x_block0, x_block1, x_block2, x_block3, x_block4 = self.E0(x[:, 0:3])

        try:
            if x_block0.isnan().sum() > 0:
                raise ValueError
        except:
            pass

        # Second Encoder (Channels 4-6)
        x_block0_temp, x_block1_temp, x_block2_temp, x_block3_temp, x_block4_temp = self.E1(x[:, 3:6])
        x_block0 = torch.cat([x_block0, x_block0_temp], 1)
        x_block1 = torch.cat([x_block1, x_block1_temp], 1)
        x_block2 = torch.cat([x_block2, x_block2_temp], 1)
        x_block3 = torch.cat([x_block3, x_block3_temp], 1)
        x_block4 = torch.cat([x_block4, x_block4_temp], 1)

        try:
            if x_block0.isnan().sum() > 0:
                raise ValueError
        except:
            pass

        # Third Encoder (Channels 7-9)
        x_block0_temp, x_block1_temp, x_block2_temp, x_block3_temp, x_block4_temp = self.E2(x[:, 6:9])
        x_block0 = torch.cat([x_block0, x_block0_temp], 1)
        x_block1 = torch.cat([x_block1, x_block1_temp], 1)
        x_block2 = torch.cat([x_block2, x_block2_temp], 1)
        x_block3 = torch.cat([x_block3, x_block3_temp], 1)
        x_block4 = torch.cat([x_block4, x_block4_temp], 1)

        try:
            if x_block0.isnan().sum() > 0:
                raise ValueError
        except:
            pass

        # Fourth Encoder (Channels 10-12)
        x_block0_temp, x_block1_temp, x_block2_temp, x_block3_temp, x_block4_temp = self.E3(x[:, 9:12])
        x_block0 = torch.cat([x_block0, x_block0_temp], 1)
        x_block1 = torch.cat([x_block1, x_block1_temp], 1)
        x_block2 = torch.cat([x_block2, x_block2_temp], 1)
        x_block3 = torch.cat([x_block3, x_block3_temp], 1)
        x_block4 = torch.cat([x_block4, x_block4_temp], 1)

        try:
            if x_block0.isnan().sum() > 0:
                raise ValueError
        except:
            pass

        # Clear Memory
        # del x_block0_temp, x_block1_temp, x_block2_temp, x_block3_temp, x_block4_temp
        # torch.cuda.empty_cache()

        x_block0 = self.M0(x_block0)
        x_block1 = self.M1(x_block1)
        x_block2 = self.M2(x_block2)
        x_block3 = self.M3(x_block3)
        x_block4 = self.M4(x_block4)
       

        x_decoder = self.D2(x_block0, x_block1, x_block2, x_block3, x_block4) 

        x_mff = self.MFF(x_block0, x_block1, x_block2, x_block3, x_block4, [x_decoder.size(2), x_decoder.size(3)]) 

        x_R1 = self.R(torch.cat((x_decoder, x_mff), 1))

        out = self.R2(x_R1) 

        try:
            if out.isnan().sum() > 0:
                raise ValueError
        except:
            pass

        return out


class model_n12_light(nn.Module):
    """
    Light version, which uses the merger in the beginning to merge the channels, without duplicating the Encoder
    """
    def __init__(self, Encoder, num_features, block_channel):
        """
        # Parameters:\n
        - body: The main body of the model, in this case generally refers to original IM2ELEVATION model\n
        - head: The head of the model, takes in output from the body and outputs it as the output of this model\n
        - cut: The index to cut the body after, ie. body = body[:cut]. Defaults to None. \n

        """

        super(model_n12_light, self).__init__()
        
        # Mergers for the different channels
        self.M0 = nn.Conv2d(12, 3, 1, 1)

        # Instantiate Encoders and set their channels
        self.E = Encoder

        # Set their respective channels for normalisation
        self.E.set_channels(list(range(0, 3)))


        self.D2 = modules.D2(num_features = num_features)
        self.MFF = modules.MFF(block_channel)
        self.R = modules.R1()
        self.R2 = modules.R2()


    def forward(self, x):
        
        # Merge the channels
        x = self.M0(x)

        # First Encoder (Channels 1-3)
        x_block0, x_block1, x_block2, x_block3, x_block4 = self.E(x[:, 0:3])

        try:
            if x_block0.isnan().sum() > 0:
                raise ValueError
        except:
            pass

       

        x_decoder = self.D2(x_block0, x_block1, x_block2, x_block3, x_block4) 

        x_mff = self.MFF(x_block0, x_block1, x_block2, x_block3, x_block4, [x_decoder.size(2), x_decoder.size(3)]) 

        x_R1 = self.R(torch.cat((x_decoder, x_mff), 1))

        out = self.R2(x_R1) 

        try:
            if out.isnan().sum() > 0:
                raise ValueError
        except:
            pass

        return out


class model_n12_visualise(nn.Module):
    """
    Final model that aims to put a head to the IM2ELEVATION model to for UMP prediction
    """
    def __init__(self, Encoder, num_features, block_channel):
        """
        # Parameters:\n
        - body: The main body of the model, in this case generally refers to original IM2ELEVATION model\n
        - head: The head of the model, takes in output from the body and outputs it as the output of this model\n
        - cut: The index to cut the body after, ie. body = body[:cut]. Defaults to None. \n

        """

        super(model_n12_visualise, self).__init__()
        
        # self.E0, self.E1, self.E2, self.E3 = Encoders
        # Instantiate Encoders and set their channels
        self.E0 = Encoder
        self.E1 = copy.deepcopy(Encoder)
        self.E2 = copy.deepcopy(Encoder)
        self.E3 = copy.deepcopy(Encoder)

        # Set their respective channels for normalisation
        self.E0.set_channels(list(range(0, 3)))
        self.E1.set_channels(list(range(3, 6)))
        self.E2.set_channels(list(range(6, 9)))
        self.E3.set_channels(list(range(9, 12)))

        # Mergers for the different channels
        self.M0 = nn.Conv2d(256, 64, 1, 1)
        self.M1 = nn.Conv2d(1024, 256, 1, 1)
        self.M2 = nn.Conv2d(2048, 512, 1, 1)
        self.M3 = nn.Conv2d(4096, 1024, 1, 1)
        self.M4 = nn.Conv2d(8192, 2048, 1, 1)

        self.D2 = modules.D2(num_features = num_features)
        self.MFF = modules.MFF(block_channel)
        self.R = modules.R1()
        self.R2 = modules.R2()


    def forward(self, x):
        
        
        # First Encoder (Channels 1-3)
        x_block0, x_block1, x_block2, x_block3, x_block4 = self.E0(x[:, 0:3])

        try:
            if x_block0.isnan().sum() > 0:
                raise ValueError
        except:
            pass

        # Second Encoder (Channels 4-6)
        x_block0_temp, x_block1_temp, x_block2_temp, x_block3_temp, x_block4_temp = self.E1(x[:, 3:6])
        x_block0 = torch.cat([x_block0, x_block0_temp], 1)
        x_block1 = torch.cat([x_block1, x_block1_temp], 1)
        x_block2 = torch.cat([x_block2, x_block2_temp], 1)
        x_block3 = torch.cat([x_block3, x_block3_temp], 1)
        x_block4 = torch.cat([x_block4, x_block4_temp], 1)

        try:
            if x_block0.isnan().sum() > 0:
                raise ValueError
        except:
            pass

        # Third Encoder (Channels 7-9)
        x_block0_temp, x_block1_temp, x_block2_temp, x_block3_temp, x_block4_temp = self.E2(x[:, 6:9])
        x_block0 = torch.cat([x_block0, x_block0_temp], 1)
        x_block1 = torch.cat([x_block1, x_block1_temp], 1)
        x_block2 = torch.cat([x_block2, x_block2_temp], 1)
        x_block3 = torch.cat([x_block3, x_block3_temp], 1)
        x_block4 = torch.cat([x_block4, x_block4_temp], 1)

        try:
            if x_block0.isnan().sum() > 0:
                raise ValueError
        except:
            pass

        # Fourth Encoder (Channels 10-12)
        x_block0_temp, x_block1_temp, x_block2_temp, x_block3_temp, x_block4_temp = self.E3(x[:, 9:12])
        x_block0 = torch.cat([x_block0, x_block0_temp], 1)
        x_block1 = torch.cat([x_block1, x_block1_temp], 1)
        x_block2 = torch.cat([x_block2, x_block2_temp], 1)
        x_block3 = torch.cat([x_block3, x_block3_temp], 1)
        x_block4 = torch.cat([x_block4, x_block4_temp], 1)

        try:
            if x_block0.isnan().sum() > 0:
                raise ValueError
        except:
            pass

        # Clear Memory
        # del x_block0_temp, x_block1_temp, x_block2_temp, x_block3_temp, x_block4_temp
        # torch.cuda.empty_cache()

        x_block0 = self.M0(x_block0)
        x_block1 = self.M1(x_block1)
        x_block2 = self.M2(x_block2)
        x_block3 = self.M3(x_block3)
        x_block4 = self.M4(x_block4)
       

        x_decoder = self.D2(x_block0, x_block1, x_block2, x_block3, x_block4) 

        x_mff = self.MFF(x_block0, x_block1, x_block2, x_block3, x_block4, [x_decoder.size(2), x_decoder.size(3)]) 

        x_R1 = self.R(torch.cat((x_decoder, x_mff), 1))

        # out = self.R2(x_R1) 

        # try:
        #     if out.isnan().sum() > 0:
        #         raise ValueError
        # except:
        #     pass

        return x_R1
        # return out








# class parallel_model(nn.Module):
#     """
#     Model that increases the number of input channels (split_start = 0) by running the model in parallel then merging\n
#     the layers at the split_end layer. 
#     """
#     def __init__(self, models, split_end, head= None, split_start= 0, n= 1):
#         """
#         # Parameters:\n
#         - models: The list of model(s) to run in parallel. If len(models) == 1: model will be split into n parallel models.
#             Else len(models) == n\n
#         - split_end: The layer to end the parallel modelling, output of this layer will be merged additionally
#         - head: The head of the model, takes in output from the last layer(s, if parallel) and outputs it as the output of this model
#             If None (Default), then the output will be concatenated\n
#         - cut: The index to cut the body after, ie. body = body[:cut]. Defaults to None. \n
#         - split_start: the layer to split, if split_start = 0, then number of input channels increases accordingly\n
#         - n: The number of splits\n

#         """

#         super(parallel_model, self).__init__()
        
#         # Consistency check
#         if len(models) != 1 and len(models) != n:
#             raise ValueError(f"Number of splits ({n}) must be consistent with number of models {len(models)}")
        
#         # Assign attributes
#         self.split_start = split_start
#         self.split_end = split_end
#         self.head = head
#         self.n = n
        
#         # Set up parallel models
#         if len(models) > 1:
#             self.models = [nn.Sequential(*list(model.children())[split_start:split_end]) for model in models]
#             self.split_input_channels = [next(list(model.children())[self.split_start].parameters()).shape[1] for model in self.models]
#             self.split_output_channels = [next(list(model.children())[self.split_end].parameters()).shape[1] for model in self.models]
#         elif len(models) == 1:
#             self.models = [nn.Sequential(*list(copy.deepcopy(models[0]).children())[split_start:split_end]) for _ in range(n)]
#             self.split_input_channels = [next(list(models[0].children())[self.split_start].parameters()).shape[1] for _ in range(n)]
#             self.split_output_channels = [next(list(models[0].children())[self.split_end].parameters()).shape[1] for _ in range(n)]
#         else:
#             raise ValueError("There must be at least one model in 'models'")

#         # Checks if there are still layers after the split_end
#         self.premature_cut = self.split_end != -1 or self.split_end != len(list(self.models.children())) - 1
#         if self.premature_cut:
#             self.merger = nn.Conv2d(sum(self.split_output_channels), 
#                 next(list(models[0].children())[self.split_end+1].parameters()).shape[1],
#                 1, 1)
#             self.remainder_model = nn.Sequential(*list(models[0].children())[split_end:])


#     def forward(self, x):

#         # For now assuming split is done at the start
#         # Initialise variables to keep track of splitting and its results
#         splits = torch.tensor([[]])
#         split_count = 0
#         # Ensure that the total number of input channels matches that of the input x
#         assert sum(self.split_input_channels) == x.shape[1]
#         # Split the x accordingly for each model in parallel and stack/cat them
#         for split in range(math.ceil(x.shape[1] / self.split_input_channels[0])):
#             model_cur = self.models[split]
#             channel_start = split_count
#             channel_end = split_count + self.split_input_channels[split]
#             out_cur = model_cur(x[:, channel_start:channel_end])
#             if split == 1:
#                 splits = out_cur
#             else:
#             splits = torch.cat([splits, out_cur], 1)

#         # Run through head if any

#         if not self.head is None:
#             out = self.head(splits)
        
#         # If the end split is before the last layer of the first 
#         # (assumption here is that the first model will be used to continue if at all)
#         if self.premature_cut:
#             # 1x1 conv to reduce channels
#             out = self.merger(self.premature_cut)
#             out = self.remainder_model(out)
        
#         return out



