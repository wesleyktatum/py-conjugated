"""
This module contains neural networks for predicting device performance,
based on tabular data and m2py labels
"""

import torch
import torch.nn as nn


class OPV_df_NN(nn.Module):
    """
    expects tabular data predicting 4 OPV device metrics: PCE, Voc, Jsc, and FF
    """
    
    def __init__(self, in_dims, out_dims):
        super(OPV_df_NN, self).__init__()
        
        #emedding layer
        self.em_layer = nn.Sequential(
            nn.Linear(in_dims, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
        )
        
        #output layers
        self.PCE_branch = nn.Sequential(
            nn.Dropout(p = 0.1),
            nn.Linear(500, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU()
        )
        self.Voc_branch = nn.Sequential(
            nn.Dropout(p = 0.1),
            nn.Linear(500, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 1),
            nn.ReLU()
        )
        self.Jsc_branch = nn.Sequential(
            nn.Dropout(p = 0.1),
            nn.Linear(500, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU()
        )
        self.FF_branch = nn.Sequential(
            nn.Dropout(p = 0.1),
            nn.Linear(500, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU()
        )
        
    def forward(self, x):
        #data enters embedding layer
        out = self.em_layer(x)
        
        #embedded data is passed to output layer
        PCE_out = self.PCE_branch(out)
        Voc_out = self.Voc_branch(out)
        Jsc_out = self.Jsc_branch(out)
        FF_out = self.FF_branch(out)
        
        return PCE_out, Voc_out, Jsc_out, FF_out
        


class OPV_m2py_NN(nn.Module):
    """
    expects m2py labels or images of size 256x256xz as a .npy file
    """
    
    def __init__(self, im_z):
        super(OPV_m2py_NN, self).__init__()
                
        self.layer1 = nn.Sequential(
            nn.Conv2d(im_z, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Flatten(),
            nn.Dropout()               #helps avoid over-fitting
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(65536, 5000),    # w/ 3 conv layers, input = 131072, w/ 2 conv layers, input = 262144
            nn.ReLU()
        )
        
        self.pce_layer = nn.Sequential(
            nn.BatchNorm1d(5000),
            nn.Linear(5000, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        
        self.voc_layer = nn.Sequential(
            nn.BatchNorm1d(5000),
            nn.Linear(5000, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        
        self.jsc_layer = nn.Sequential(
            nn.BatchNorm1d(5000),
            nn.Linear(5000, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        
        self.ff_layer = nn.Sequential(
            nn.BatchNorm1d(5000),
            nn.Linear(5000, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
            
    def forward(self, im):
        #convolution series
        im_encoding = self.layer1(im)
        im_encoding = self.layer2(im_encoding)
        im_encoding = self.layer3(im_encoding)
        
        #linear encoding
        im_encoding = self.layer4(im_encoding)
        
        #output layers
        pce_out = self.pce_layer(im_encoding)
        voc_out = self.pce_layer(im_encoding)
        jsc_out = self.pce_layer(im_encoding)
        ff_out = self.pce_layer(im_encoding)
        
        return pce_out, voc_out, jsc_out, ff_out
    
    

class OPV_mixed_NN(nn.Module):
    """
    This class calls three classes that are the data encoding branches whose
    outputs are concatenated before being fed into the predictor.
    
    The image branch is for image-like data with the shape (in_channels x 256 x 256)
    Branch three is for the tabular data (batchsize x in_dims)
    """
    def __init__(self, in_channels, in_dims):
        super(OPV_mixed_NN, self).__init__()
        
        self.im_branch = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(65536, 5000),
            nn.ReLU()
        )
        
        self.tab_branch = nn.Sequential(
            nn.Linear(in_dims, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU()
        )
        
        self.pce_predictor = nn.Sequential(
            nn.Linear(154000, 50000),
            nn.ReLU(),
            nn.Linear(50000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 500),
            nn.ReLU(),
            nn.Linear(500, 1)
        )
        
        self.voc_predictor = nn.Sequential(
            nn.Linear(154000, 50000),
            nn.ReLU(),
            nn.Linear(50000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 500),
            nn.ReLU(),
            nn.Linear(500, 1)
        )
        
        self.jsc_predictor = nn.Sequential(
            nn.Linear(154000, 50000),
            nn.ReLU(),
            nn.Linear(50000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 500),
            nn.ReLU(),
            nn.Linear(500, 1)
        )
        
        self.ff_predictor = nn.Sequential(
            nn.Linear(154000, 50000),
            nn.ReLU(),
            nn.Linear(50000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 500),
            nn.ReLU(),
            nn.Linear(500, 1)
        )
        
    def forward(self, im, df):
        im_enc = self.im_branch(im)
        df_enc = self.tab_branch(df)
        
        im_enc = im_enc.view(-1)
        df_enc = df_enc.view(-1)
        
        print(im_enc.size())
        print(df_enc.size())
        
        total_encoding = torch.cat([im_enc, df_enc], -1)
        
        pce_out = self.pce_predictor(total_encoding)
        voc_out = self.voc_predictor(total_encoding)
        jsc_out = self.jsc_predictor(total_encoding)
        ff_out = self.ff_predictor(total_encoding)
        
        return pce_out, voc_out, jsc_out, ff_out
    
    
class OPV_total_NN(nn.Module):
    """
    This class calls three classes that are the data encoding branches whose
    outputs are concatenated before being fed into the predictor.
    
    Branch one is for the raw afm data (8x256x256)
    Branch two is for the m2py labels (2x256x256)
    Branch three is for the tabular data (batchsize x in_dims)
    """
    def __init__(self, in_dims):
        super(OPV_total_NN, self).__init__()
        
        self.afm_branch = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(65536, 5000),
            nn.ReLU()
        )
        
        self.m2py_branch = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(65536, 5000),
            nn.ReLU()
        )
        
        self.tab_branch = nn.Sequential(
            nn.Linear(in_dims, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU()
        )
        
        self.pce_predictor = nn.Sequential(
            nn.Linear(294000, 50000),
            nn.ReLU(),
            nn.Linear(50000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 500),
            nn.ReLU(),
            nn.Linear(500, 1)
        )
        
        self.voc_predictor = nn.Sequential(
            nn.Linear(294000, 50000),
            nn.ReLU(),
            nn.Linear(50000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 500),
            nn.ReLU(),
            nn.Linear(500, 1)
        )
        
        self.jsc_predictor = nn.Sequential(
            nn.Linear(294000, 50000),
            nn.ReLU(),
            nn.Linear(50000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 500),
            nn.ReLU(),
            nn.Linear(500, 1)
        )
        
        self.ff_predictor = nn.Sequential(
            nn.Linear(294000, 50000),
            nn.ReLU(),
            nn.Linear(50000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 500),
            nn.ReLU(),
            nn.Linear(500, 1)
        )
        
    def forward(self, afm, m2py, df):
        afm_enc = self.afm_branch(afm)
        m2py_enc = self.m2py_branch(m2py)
        df_enc = self.tab_branch(df)
        
        afm_enc = afm_enc.view(-1)
        m2py_enc = m2py_enc.view(-1)
        df_enc = df_enc.view(-1)
        
        total_encoding = torch.cat([afm_enc, m2py_enc, df_enc])
        
        pce_out = self.pce_predictor(total_encoding)
        voc_out = self.voc_predictor(total_encoding)
        jsc_out = self.jsc_predictor(total_encoding)
        ff_out = self.ff_predictor(total_encoding)
        
        return pce_out, voc_out, jsc_out, ff_out
    

class OFET_df_NN(nn.Module):
    
    def __init__(self, in_dims, out_dims):
        super(OFET_df_NN, self).__init__()
        
         #emedding layer
        self.em_layer = nn.Sequential(
            nn.Linear(in_dims, 16),
            nn.ReLU()
        )
        
        #hidden layers
        self.h_layers = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        
        #output layers
        self.mu_branch = nn.Sequential(
            nn.Dropout(p = 0.1),
            nn.Linear(8, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )
        self.r_branch = nn.Sequential(
            nn.Dropout(p = 0.1),
            nn.Linear(8, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )
        self.on_off_branch = nn.Sequential(
            nn.Dropout(p = 0.1),
            nn.Linear(8, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )
        self.vt_branch = nn.Sequential(
            nn.Dropout(p = 0.1),
            nn.Linear(8, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )
        
    def forward(self, x):
        #data enters embedding layer
        out = self.em_layer(x)
        
        #embedded data is passed to hidden layers
        out = self.h_layers(out)
        
        #embedded data is passed to output layer
        mu_out = self.mu_branch(out)
        r_out = self.r_branch(out)
        on_off_out = self.on_off_branch(out)
        vt_out = self.vt_branch(out)
        
        return mu_out, r_out, on_off_out, vt_out
    
    
    