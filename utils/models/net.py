import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils.models import modules
from torchvision import utils

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
        self.R = modules.R2()


    def forward(self, x):

        x_block0, x_block1, x_block2, x_block3, x_block4 = self.E(x)

        x_decoder = self.D2(x_block0, x_block1, x_block2, x_block3, x_block4) 

        x_mff = self.MFF(x_block0, x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)]) 

        out = self.R(torch.cat((x_decoder, x_mff), 1)) 

        return out



