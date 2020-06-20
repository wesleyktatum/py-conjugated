"""
This modules contains utility functions for data manipulation and plotting of
results and data
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch
import boto3
from s3fs.core import S3FileSystem
import io
from torch.utils.data import Dataset

#######################################################
#                  Data Utilities    
#######################################################


def load_s3_ims(bucket_name, filepath):
    """
    This function takes in an s3 bucket path pointing to .npy format images.
    It returns a dictionary of those images, as well as a pd.DataFrame with
    the image's identifier labels.
    """
    
    client = boto3.client('s3')
    resource = boto3.resource('s3')
    s3_bucket = resource.Bucket(bucket_name)
    
    files = list(s3_bucket.objects.filter(Prefix = filepath))
    
    im_dict = {}
    label_dict = {}
    for i, obj in enumerate(files):
    #     print(obj.bucket_name)
    #     print(obj.key)
        fl = obj.key[37:]
#         print(fl)
        anl_temp = 0
        anl_time = 0
        sub = 0
        dev = 0
        
        if fl[-1] == '/':
            pass
        
        else:
            byte_stream = io.BytesIO(obj.get()['Body'].read())
            
            im = np.load(byte_stream)
#             plt.imshow(im[:,:,0])
#             plt.show()

            im_index = len(im_dict)
            im_dict[im_index] = im

            if 'NOANNEAL' in fl:
                #time = temp = 0
                anl_temp = 0
                anl_time = 0

                #extract sub and dev
                s_idx = fl.index('S')+1
                d_idx = fl.index('D')+1

                sub = fl[s_idx]
                dev = fl[d_idx]

            elif 'postexam' in fl:
                #extract temp, time, sub, dev from filename
                temp_stop_indx = fl.index('C')
                temp_start_indx = fl.index('/') + 1
                anl_temp = int(fl[temp_start_indx:temp_stop_indx])

                time_start_indx = temp_stop_indx+2
                time_stop_indx = fl.index('m')
                time_stop_indx = time_stop_indx
                anl_time = fl[time_start_indx:time_stop_indx]
                anl_time = int(anl_time)

                sub = 4
                dev = 6

            elif fl[-1] != '/':
                #extract temp, time, sub, dev from filename
                temp_stop_indx = fl.index('C')
                temp_start_indx = fl.index('/') + 1
                anl_temp = int(fl[temp_start_indx:temp_stop_indx])

                time_start_indx = temp_stop_indx+2
                time_stop_indx = fl.index('m')
                time_stop_indx = time_stop_indx
                anl_time = fl[time_start_indx:time_stop_indx]
                anl_time = int(anl_time)

                s_idx = fl.index('b')+1
                d_idx = fl.index('v')+1

                sub = fl[s_idx]
                dev = fl[d_idx]

            #assign entry identifiers to label key
            label_dict[i] = {'Anneal_time' : int(anl_time), 'Anneal_temp' : int(anl_temp),
                                     'Substrate' : int(sub), 'Device': int(dev)}
        
    label_df = pd.DataFrame.from_dict(label_dict, orient = 'index')
            
    return im_dict, label_df


class OPV_ImDataset(torch.utils.data.Dataset):
    """
    This class takes in an s3 bucket name and filepath, and calls nuts.load_s3_ims
    to initialize a custom dataset class that inherets from PyTorch. 
    """
    def __init__(self,bucket_name, filepath):
        super(OPV_ImDataset).__init__()
        self.im_dict, self.im_labels = load_s3_ims(bucket_name, filepath)
        self.keys = self.im_labels.index
        

    def __len__(self):
        return len(self.im_dict)

    
    def __getitem__(self, key):
        
        self.im_tensor = self.convert_im_to_tensors(self.im_dict[key])
        self.label_tensor = self.convert_label_to_tensors(self.im_labels.iloc[key].tolist())
        
        return self.im_tensor, self.label_tensor
    
    
    def convert_im_to_tensors(self, im):
        
        im_tensor = torch.from_numpy(im).float()
        im_tensor = im_tensor.view(2, 256, 256)
        
        return im_tensor
        
        
    def convert_label_to_tensors(self, label_df):
        label_tensor =  torch.tensor(label_df[:2]).float()
        
        return label_tensor
    
    
class local_OPV_ImDataset(torch.utils.data.Dataset):
    """
    This class takes in a filepath pointing to a directory of images and labels,
    and loads them into a custom dataset class that inherets from PyTorch. 
    """
    def __init__(self, filepath):
        super(local_OPV_ImDataset).__init__()
        
        files = os.listdir(filepath)
        
        self.im_dict = {}
        label_dict = {}
        for i, fl in enumerate(files):
#             print(fl)
            anl_temp = 0
            anl_time = 0
            sub = 0
            dev = 0

            if fl[-1] == '/':
                pass
            elif fl[-1] == 'e':
                pass

            else:
                im = np.load(filepath+fl)

                im_index = len(self.im_dict)
                self.im_dict[im_index] = im

                if 'NOANNEAL' in fl:
                    #time = temp = 0
                    anl_temp = 0
                    anl_time = 0

                    #extract sub and dev
                    s_idx = fl.index('S')+1
                    d_idx = fl.index('D')+1

                    sub = fl[s_idx]
                    dev = fl[d_idx]

                elif 'postexam' in fl:
                    #extract temp, time, sub, dev from filename
                    temp_stop_indx = fl.index('C')
                    temp_start_indx = 0
                    anl_temp = int(fl[temp_start_indx:temp_stop_indx])

                    time_start_indx = temp_stop_indx+2
                    time_stop_indx = fl.index('m')
                    time_stop_indx = time_stop_indx
                    anl_time = fl[time_start_indx:time_stop_indx]
                    anl_time = int(anl_time)

                    sub = 4
                    dev = 6

                elif fl[-1] != '/':
                    #extract temp, time, sub, dev from filename
                    temp_stop_indx = fl.index('C')
                    temp_start_indx = 0
                    anl_temp = int(fl[temp_start_indx:temp_stop_indx])

                    time_start_indx = temp_stop_indx+2
                    time_stop_indx = fl.index('m')
                    time_stop_indx = time_stop_indx
                    anl_time = fl[time_start_indx:time_stop_indx]
                    anl_time = int(anl_time)

                    s_idx = fl.index('b')+1
                    d_idx = fl.index('v')+1

                    sub = fl[s_idx]
                    dev = fl[d_idx]

                #assign entry identifiers to label key
                label_dict[i] = {'Anneal_time' : int(anl_time), 'Anneal_temp' : int(anl_temp),
                                         'Substrate' : int(sub), 'Device': int(dev)}

        self.im_labels = pd.DataFrame.from_dict(label_dict, orient = 'index')
        
        
    def __len__(self):
        return len(self.im_dict)

    
    def __getitem__(self, key):
        
        self.im_tensor = self.convert_im_to_tensors(self.im_dict[key])
        self.label_tensor = self.convert_label_to_tensors(self.im_labels.iloc[key].tolist())
        
        return self.im_tensor, self.label_tensor
    
    
    def convert_im_to_tensors(self, im):
        
        im_tensor = torch.from_numpy(im).float()
        im_tensor = im_tensor.view(1, 2, 256, 256)
        
        return im_tensor
        
        
    def convert_label_to_tensors(self, label_df):
        label_tensor =  torch.tensor(label_df).float()
        
        return label_tensor
    

def load_trained_model(previous_model, model, optimizer):
    
    checkpoint = torch.load(previous_model)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()
        
    return model, optimizer


def save_trained_model(save_path, epoch, model, optimizer, train_loss, test_loss):
    save_dict = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
#         'train_losses': train_loss
#         'test_losses': [pce_test_loss, voc_test_loss,
#                        jsc_test_loss, ff_test_loss]
        'optimizer': optimizer.state_dict()
        }
        
    torch.save(save_dict, save_path)
    return


def df_MinMax_normalize(dataframe):
    
    df = dataframe
    
    normed_df = pd.DataFrame()

    df_norm_key = {}

    for colname, coldata in df.iteritems():
        max_val = coldata.max()
        min_val = coldata.min()

        df_norm_key[colname] = [min_val, max_val]

        normed_col = (coldata - min_val) / (max_val - min_val)
        normed_df[colname] = normed_col
        
    return normed_df, df_norm_key 


def df_MinMax_denormalize(normed_df, norm_key):
    
    denormed_df = pd.DataFrame()
    
    for colname, coldata in normed_df.iteritems():
        mn = norm_key[colname][0]
        mx = norm_key[colname][1]
        
        denormed_col = (coldata * (mx - mn)) + mn
        
        denormed_df[colname] = denormed_col
        
    return denormed_df


def df_Gaussian_normalize(dataframe):
    
    df = dataframe
    normed_df = pd.DataFrame()
    norm_key = {}
    
    for colname, coldata in df.iteritems():
        stdev = coldata.std()
        mean = coldata.mean()
        
        normed_col = (coldata - mean) / stdev
        normed_df[colname] = normed_col
        
        norm_key[colname] = [mean, stdev]
        
    return normed_df, norm_key


def df_Gaussian_denormalize(normed_df, norm_key):
    
    denormed_df = pd.DataFrame()
    
    for colname, coldata in normed_df.iteritems():
        mean = norm_key[colname][0]
        std = norm_key[colname][1]
        
        denormed_col = (coldata * mean) + std
        
        denormed_df[colname] = denormed_col
        
    return denormed_df


def normed_areas(dataframe):
    """
    Takes in morphology descriptors dataframe and calculates a normalized area covered
    by each of the present phases
    """    
    phases = dataframe['GMM_label'].unique()
    
    areas = []
    
    for phase in phases:
        
        phase_rows = dataframe[dataframe['GMM_label'] == phase]
        total_area = phase_rows['area'].sum()
        areas.append(total_area)
        
    normed_areas = areas/max(areas)
    
    return normed_areas


def relative_areas(dataframe):
    """
    Takes in morphology descriptors dataframe and calculates a normalized relative ratios
    of the areas covered by each of the present phases
    """    
    phases = dataframe['GMM_label'].unique()
    
    areas = []
    
    total_area = 0

    for phase in phases:
        
        phase_rows = dataframe[dataframe['GMM_label'] == phase]
        phase_area = phase_rows['area'].sum()
        areas.append(phase_area)
        total_area+=phase_area
        
    relative_areas = areas/total_area
    
    return relative_areas


#######################################################
#                  Network Model Utilities
#######################################################

def init_weights(model):
    if type(model) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(model.weight)
        model.bias.data.fill_(0.01)
        
#     if type(model) == nn.BatchNorm1d:
#         model.reset_parameters()

    

#######################################################
#                  Plotting Utilities
#######################################################

def plot_OPV_df_loss(epochs, train_epoch_losses, test_epoch_losses,
                     pce_train_epoch_losses, pce_test_epoch_losses,
                     voc_train_epoch_losses, voc_test_epoch_losses,
                     jsc_train_epoch_losses, jsc_test_epoch_losses,
                     ff_train_epoch_losses, ff_test_epoch_losses):
    
    
    fig, ax = plt.subplots(figsize = (8,6))
    
    plt.plot(epochs, train_epoch_losses, c = 'k', label = 'training error')
    plt.plot(epochs, test_epoch_losses, c = 'r', label = 'testing error')
    plt.legend(loc = 'upper right')
    plt.title("Total Training & Testing Error")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total MSE Loss')
    plt.show()
    
    fig, ax = plt.subplots(figsize = (8,6))

    plt.plot(epochs[::], pce_train_epoch_losses[::], c = 'k', label = 'pce training')
    plt.plot(epochs[::], pce_test_epoch_losses[::], '-.', c = 'k', label = 'pce testing')

    plt.plot(epochs[::], voc_train_epoch_losses[::], c = 'r', label = 'voc training')
    plt.plot(epochs[::], voc_test_epoch_losses[::], '-.', c = 'r', label = 'voc testing')

    plt.plot(epochs[::], jsc_train_epoch_losses[::], c = 'g', label = 'jsc training')
    plt.plot(epochs[::], jsc_test_epoch_losses[::], '-.', c = 'g', label = 'jsc testing') 
    
    plt.plot(epochs[::], ff_train_epoch_losses[::], c = 'b', label = 'ff training') 
    plt.plot(epochs[::], ff_test_epoch_losses[::], '-.', c = 'b', label = 'ff testing') 

    plt.legend(loc = 'upper right')
    plt.title("Branch Training & Testing Error")
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE')
    plt.show()
    
    return

def plot_OPV_df_accuracies(epochs, pce_test_epoch_accuracies, voc_test_epoch_accuracies, 
                           jsc_test_epoch_accuracies, ff_test_epoch_accuracies):
    
    fig, ax = plt.subplots(figsize = (8,6))
    # plt.plot(epochs, train_epoch_accuracy, c = 'k', label = 'training accuracy')
    plt.plot(epochs, pce_test_epoch_accuracies, c = 'k', label = 'pce MAPE')
    plt.plot(epochs, voc_test_epoch_accuracies, c = 'r', label = 'voc MAPE')
    plt.plot(epochs, jsc_test_epoch_accuracies, c = 'g', label = 'jsc MAPE')
    plt.plot(epochs, ff_test_epoch_accuracies, c = 'b', label = 'ff MAPE')
    plt.legend(loc = 'upper right')
    plt.title("Branch Testing Accuracy")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Absolute Percent Error')
    plt.show()
    
    return

def plot_OPV_parity(pce_labels, PCE_out, voc_labels, Voc_out,
                    jsc_labels, Jsc_out, ff_labels, FF_out):
    
    xlin = ylin = np.arange(-10, 10, 1)

    r2 = r2_score(pce_labels, PCE_out)
    fig, ax = plt.subplots(figsize = (8,6))
    plt.scatter(PCE_out, pce_labels)
    plt.plot(xlin, ylin, c = 'k')
    ax.annotate(f"$R^{2}$ = {r2:.3f}", xy = (0.2, 0.4), xycoords = 'figure fraction')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Predictions")
    ax.set_ylabel("Ground Truth")
    plt.title('PCE Parity')
    plt.show()

    r2 = r2_score(voc_labels, Voc_out)
    fig, ax = plt.subplots(figsize = (8,6))
    ax.annotate(f"$R^{2}$ = {r2:.3f}", xy = (0.2, 0.4), xycoords = 'figure fraction')
    plt.scatter(Voc_out, voc_labels)
    plt.plot(xlin, ylin, c = 'k')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Predictions")
    ax.set_ylabel("Ground Truth")
    plt.title('Voc Parity')
    plt.show()

    r2 = r2_score(jsc_labels, Jsc_out)
    fig, ax = plt.subplots(figsize = (8,6))
    ax.annotate(f"$R^{2}$ = {r2:.3f}", xy = (0.2, 0.4), xycoords = 'figure fraction')
    plt.scatter(Jsc_out, jsc_labels)
    plt.plot(xlin, ylin, c = 'k')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Predictions")
    ax.set_ylabel("Ground Truth")
    plt.title('Jsc Parity')
    plt.show()

    r2 = r2_score(ff_labels, FF_out)
    fig, ax = plt.subplots(figsize = (8,6))
    ax.annotate(f"$R^{2}$ = {r2:.3f}", xy = (0.2, 0.4), xycoords = 'figure fraction')
    plt.scatter(FF_out, ff_labels)
    plt.plot(xlin, ylin, c = 'k')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Predictions")
    ax.set_ylabel("Ground Truth")
    plt.title('FF Parity')
    plt.show()
    
    
def plot_OFET_df_loss(epochs, train_epoch_losses, test_epoch_losses,
                      mu_train_epoch_losses, mu_test_epoch_losses,
                      r_train_epoch_losses, r_test_epoch_losses,
                      on_off_train_epoch_losses, on_off_test_epoch_losses,
                      vt_train_epoch_losses, vt_test_epoch_losses):
    
    
    fig, ax = plt.subplots(figsize = (8,6))
    
    plt.plot(epochs[::], train_epoch_losses[::], c = 'k', label = 'training error')
    plt.plot(epochs[::], test_epoch_losses[::], c = 'r', label = 'testing error')
    plt.legend(loc = 'upper right')
    plt.title("Total Training & Testing Error")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total MSE Loss')
    plt.show()
    
    fig, ax = plt.subplots(figsize = (8,6))

    plt.plot(epochs[::], mu_train_epoch_losses[::], c = 'k', label = 'mu training')
    plt.plot(epochs[::], mu_test_epoch_losses[::], '-.', c = 'k', label = 'mu testing')

    plt.plot(epochs[::], r_train_epoch_losses[::], c = 'r', label = 'r training')
    plt.plot(epochs[::], r_test_epoch_losses[::], '-.', c = 'r', label = 'r testing')

    plt.plot(epochs[::], on_off_train_epoch_losses[::], c = 'g', label = 'on_off training')
    plt.plot(epochs[::], on_off_test_epoch_losses[::], '-.', c = 'g', label = 'on_off testing') 
    
    plt.plot(epochs[::], vt_train_epoch_losses[::], c = 'b', label = 'vt training') 
    plt.plot(epochs[::], vt_test_epoch_losses[::], '-.', c = 'b', label = 'vt testing') 

    plt.legend(loc = 'upper right')
    plt.title("Branch Training & Testing Error")
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE')
    plt.show()
    
    return


def plot_OFET_df_accuracies(epochs, mu_test_epoch_accuracies, r_test_epoch_accuracies, 
                            on_off_test_epoch_accuracies, vt_test_epoch_accuracies):
    
    fig, ax = plt.subplots(figsize = (8,6))
    # plt.plot(epochs, train_epoch_accuracy, c = 'k', label = 'training accuracy')
    plt.plot(epochs, mu_test_epoch_accuracies, c = 'k', label = 'mu MAPE')
    plt.plot(epochs, r_test_epoch_accuracies, c = 'r', label = 'r MAPE')
    plt.plot(epochs, on_off_test_epoch_accuracies, c = 'g', label = 'on_off MAPE')
    plt.plot(epochs, vt_test_epoch_accuracies, c = 'b', label = 'vt MAPE')
    plt.legend(loc = 'upper right')
    plt.title("Branch Testing Accuracy")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Absolute Percent Error')
    plt.show()
    
    return


def plot_OFET_parity(mu_labels, mu_out, r_labels, r_out,
                     on_off_labels, on_off_out, vt_labels, vt_out):
    
    xlin = ylin = np.arange(-20, 20, 1)

    r2 = r2_score(mu_labels, mu_out)
    fig, ax = plt.subplots(figsize = (8,6))
    plt.scatter(mu_labels, mu_out)
    plt.plot(xlin, ylin, c = 'k')
    ax.annotate(f"$R^{2}$ = {r2:.3f}", xy = (0.2, 0.4), xycoords = 'figure fraction')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("Predictions")
    ax.set_xlabel("Ground Truth")
    plt.title('mu Parity')
    plt.show()

    r2 = r2_score(r_labels, r_out)
    fig, ax = plt.subplots(figsize = (8,6))
    ax.annotate(f"$R^{2}$ = {r2:.3f}", xy = (0.2, 0.4), xycoords = 'figure fraction')
    plt.scatter(r_labels, r_out)
    plt.plot(xlin, ylin, c = 'k')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("Predictions")
    ax.set_xlabel("Ground Truth")
    plt.title('r Parity')
    plt.show()

    r2 = r2_score(on_off_labels, on_off_out)
    fig, ax = plt.subplots(figsize = (8,6))
    ax.annotate(f"$R^{2}$ = {r2:.3f}", xy = (0.2, 0.4), xycoords = 'figure fraction')
    plt.scatter(on_off_labels, on_off_out)
    plt.plot(xlin, ylin, c = 'k')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("Predictions")
    ax.set_xlabel("Ground Truth")
    plt.title('on_off Parity')
    plt.show()

    r2 = r2_score(vt_labels, vt_out)
    fig, ax = plt.subplots(figsize = (8,6))
    ax.annotate(f"$R^{2}$ = {r2:.3f}", xy = (0.2, 0.4), xycoords = 'figure fraction')
    plt.scatter(vt_labels, vt_out)
    plt.plot(xlin, ylin, c = 'k')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("Predictions")
    ax.set_xlabel("Ground Truth")
    plt.title('Vt Parity')
    plt.show()