from __future__ import print_function
import torch
import numpy as np
from network import joint_net_vgg16
from network import anchor_net_vgg16

#==============================================================================
# joint move predictor
#==============================================================================
class joint_predictor():
    def __init__(self, test_model,
                 sil_ver = False,
                 gpu = True
                ):
        # device config
        self.device = torch.device("cuda:0" if gpu else "cpu")

        # make the network, reload the trained model
        if sil_ver is False:
            in_channels = 4
        else:
            in_channels = 2
        out_channels = 2
        
        self.joint_net = joint_net_vgg16(init_weights = False,
                                         in_channels = in_channels,
                                         num_classes = out_channels,
                                        ).eval().to(self.device)
        if gpu is True:
            self.joint_net.load_state_dict(torch.load(test_model))
        else:
            self.joint_net.load_state_dict(torch.load(test_model, 
                                                      map_location='cpu'))
    # Important: predict_one is not an efficient way to predict, we recommend
    #            to use predict_list to predict large amount of data.
    def predict_one(self, input_arr):
        input_tsr = torch.tensor(input_arr).unsqueeze(0)
        input_tsr = input_tsr.to(self.device).float()
        pred_para = self.joint_net(input_tsr)
        return pred_para[0]

    def predict_batch(self, input_arr):
        input_tsr = torch.tensor(input_arr)
        input_tsr = input_tsr.to(self.device).float()
        pred_para = self.joint_net(input_tsr)
        return pred_para

#==============================================================================
# anchor move predictor
#============================================================================== 
class anchor_predictor():
    def __init__(self, test_model,
                 sil_ver = False,
                 gpu = True
                ):
        # device config
        self.device = torch.device("cuda:0" if gpu else "cpu")

        # make the network, reload the trained model
        if sil_ver is False:
            in_channels = 4
        else:
            in_channels = 2
        out_channels = 1
        
        # make the network, reload the trained model
        self.anchor_net = anchor_net_vgg16(init_weights = False,
                                          in_channels = in_channels,
                                          num_classes = out_channels,
                                         ).eval().to(self.device)
        if gpu is True:
            self.anchor_net.load_state_dict(torch.load(test_model))
        else:
            self.anchor_net.load_state_dict(torch.load(test_model, 
                                                      map_location='cpu'))
    # Important: predict_one is not an efficient way to predict, we recommend
    #            to use predict_list to predict large amount of data.
    def predict_one(self, input_arr):
        input_tsr = torch.tensor(input_arr).unsqueeze(0)
        input_tsr = input_tsr.to(self.device).float()
        pred_para = self.anchor_net(input_tsr)
        return pred_para[0]

    def predict_batch(self, input_arr):
        input_tsr = torch.tensor(input_arr)
        input_tsr = input_tsr.to(self.device).float()
        pred_para = self.anchor_net(input_tsr)
        return pred_para
