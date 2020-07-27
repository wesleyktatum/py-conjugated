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
    #get s3 directory info
    client = boto3.client('s3')
    resource = boto3.resource('s3')
    s3_bucket = resource.Bucket(bucket_name)
    
    #gather s3 file objects from directory
    files = list(s3_bucket.objects.filter(Prefix = filepath))
    
    im_dict = {}
    label_dict = {}
    
    for i, obj in enumerate(files):
        #get filename
        fl = obj.key
        
        if fl[-1] == 'x':
            obj1 = client.get_object(Bucket = bucket_name, Key = fl)
            sample_labels = pd.read_excel(obj1['Body'].read())
        else:
            pass
    
    #loop through each file, which is an m2py labelset
    for i, obj in enumerate(files):
        #get filename
        fl = obj.key
        
        #skip directories
        if fl[-1] != '/':
            #skip excel files
            if fl[-1] != 'x':
                #only .npy files
                if fl[-1] == 'y':
                    anl_temp = 0
                    anl_time = 0
                    sub = 0
                    dev = 0
                    
                    byte_stream = io.BytesIO(obj.get()['Body'].read())

                    im = np.load(byte_stream)

                    im_index = len(im_dict)
                    im_dict[im_index] = im

                    if 'postexam' in fl:
                        #extract temp, time, sub, dev from filename
                        temp_start_indx = fl.index('set/') + 4
                        temp_stop_indx = fl.index('C')
                        anl_temp = int(fl[temp_start_indx:temp_stop_indx])

                        time_start_indx = temp_stop_indx+2
                        time_stop_indx = fl.index('min_')
                        time_stop_indx = time_stop_indx
                        anl_time = fl[time_start_indx:time_stop_indx]
                        anl_time = int(anl_time)

                        s_idx = fl.index('ub')+2

                        sub = fl[s_idx]
                        dev = 3

                    elif 'NOANNEAL' in fl:
                        anl_temp = 0
                        anl_time = 0

                        #extract sub and dev
                        s_idx = fl.index('S')+1
                        d_idx = fl.index('D')+1

                        sub = fl[s_idx]
                        dev = fl[d_idx]
                    else:
                        #extract temp, time, sub, dev from filename
                        temp_start_indx = fl.index('set/') + 4
                        temp_stop_indx = fl.index('C')
                        anl_temp = int(fl[temp_start_indx:temp_stop_indx])

                        time_start_indx = temp_stop_indx+2
                        time_stop_indx = fl.index('min_')
                        anl_time = fl[time_start_indx:time_stop_indx]
                        anl_time = int(anl_time)

                        s_idx = fl.index('ub')+2
                        d_idx = fl.index('ev')+2

                        sub = fl[s_idx]
                        dev = fl[d_idx]

            #assign entry identifiers to label key
            label_dict[i] = {'Anneal_time' : int(anl_time),
                             'Anneal_temp' : int(anl_temp),
                             'Substrate' : int(sub),
                             'Device': int(dev),
                            }
            
    label_df = pd.DataFrame.from_dict(label_dict, orient = 'index')    

    sample_indexs = []
    pce = []
    voc = []
    jsc = []
    ff = []
    indxs = []
    for i, row in label_df.iterrows():
 
        #query for sample labels that = test set identifiers
        time_matches = sample_labels[sample_labels['Anneal_time'] == row[0]]
        temp_matches = time_matches.query('Anneal_temp == @row[1]')
        sub_matches = temp_matches.query('Substrate == @row[2]')
        matches = sub_matches.query('Device == @row[3]')
        
        if len(matches) <= 0:
            print('no matches')
            print(row)
            
            pass

        else:
            #append index of match to test_sample_idxs
            match_idxs = matches.index[:].tolist()
            sample_indexs.append(match_idxs[0])

    for indx in sample_indexs:
        row = sample_labels[sample_labels.index == indx]

        pce.append(row['PCE'].item())
        voc.append(row['VocL'].item())
        jsc.append(row['Jsc'].item())
        ff.append(row['FF'].item())
        indxs.append(row['Unnamed: 0'].item())
        

    label_df['PCE'] = pce
    label_df['Vocl'] = voc
    label_df['Jsc'] = jsc
    label_df['FF'] = ff
    label_df['Index'] = indxs
            
    return im_dict, label_df


class OPV_ImDataset(torch.utils.data.Dataset):
    """
    This class takes in an s3 bucket name and filepath, and calls nuts.load_s3_ims
    to initialize a custom dataset class that inherets from PyTorch. 
    """
    def __init__(self, bucket_name, filepath):
        super(OPV_ImDataset).__init__()
        self.im_dict, self.im_labels = load_s3_ims(bucket_name, filepath)
        self.keys = self.im_labels.index        

    def __len__(self):
        return len(self.im_dict)

    
    def __getitem__(self, key):
        
        self.im_tensor = self.convert_im_to_tensors(self.im_dict[key])
        label_df = self.im_labels.iloc[key]
        label_df = label_df.drop(['Anneal_time', 'Anneal_temp', 'Substrate', 'Device'])
        self.label_tensor = self.convert_label_to_tensors(label_df)
        
        return self.im_tensor, self.label_tensor
    
    
    def convert_im_to_tensors(self, im):
        
        x, y, z = im.shape
        
        im_tensor = torch.from_numpy(im).float()
        im_tensor = im_tensor.view(z, x, y)
        
        return im_tensor
        
        
    def convert_label_to_tensors(self, label_df):
        label_tensor =  torch.tensor(label_df).float()
        label_tensor.view(-1,1)
        
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


def save_trained_model(save_path, epoch, model, optimizer):
    save_dict = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
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
    
    classname = model.__class__.__name__
    
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.zeros_(model.bias)
        
    elif classname.find('Conv2d') != -1:
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.zeros_(model.bias)
        
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.zeros_(model.bias)

    

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
    plt.scatter(pce_labels, PCE_out)
    plt.plot(xlin, ylin, c = 'k')
    ax.annotate(f"$R^{2}$ = {r2:.3f}", xy = (0.2, 0.4), xycoords = 'figure fraction')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("Predictions")
    ax.set_xlabel("Ground Truth")
    plt.title('PCE Parity')
    plt.show()

    r2 = r2_score(voc_labels, Voc_out)
    fig, ax = plt.subplots(figsize = (8,6))
    ax.annotate(f"$R^{2}$ = {r2:.3f}", xy = (0.2, 0.4), xycoords = 'figure fraction')
    plt.scatter(voc_labels, Voc_out)
    plt.plot(xlin, ylin, c = 'k')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("Predictions")
    ax.set_xlabel("Ground Truth")
    plt.title('Voc Parity')
    plt.show()

    r2 = r2_score(jsc_labels, Jsc_out)
    fig, ax = plt.subplots(figsize = (8,6))
    ax.annotate(f"$R^{2}$ = {r2:.3f}", xy = (0.2, 0.4), xycoords = 'figure fraction')
    plt.scatter(jsc_labels, Jsc_out)
    plt.plot(xlin, ylin, c = 'k')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("Predictions")
    ax.set_xlabel("Ground Truth")
    plt.title('Jsc Parity')
    plt.show()

    r2 = r2_score(ff_labels, FF_out)
    fig, ax = plt.subplots(figsize = (8,6))
    ax.annotate(f"$R^{2}$ = {r2:.3f}", xy = (0.2, 0.4), xycoords = 'figure fraction')
    plt.scatter(ff_labels,FF_out)
    plt.plot(xlin, ylin, c = 'k')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("Predictions")
    ax.set_xlabel("Ground Truth")
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
    
    
def plot_fit_results(fit_dict):
    lr = float(fit_dict['lr'])
    best_loss_epoch = int(fit_dict['best_loss_epoch'])
    best_acc_epoch = int(fit_dict['best_acc_epoch'])
    best_r2_epoch = int(fit_dict['best_r2_epoch'])
    
    test_loss = [float(i) for i in fit_dict['test_losses']]
    pce_loss = [float(i) for i in fit_dict['pce_loss']]
    voc_loss = [float(i) for i in fit_dict['voc_loss']]
    jsc_loss = [float(i) for i in fit_dict['jsc_loss']]
    ff_loss = [float(i) for i in fit_dict['ff_loss']]
    
    test_acc = [float(i) for i in fit_dict['test_accs']]
    pce_acc = [float(i) for i in fit_dict['pce_acc']]
    voc_acc = [float(i) for i in fit_dict['voc_acc']]
    jsc_acc = [float(i) for i in fit_dict['jsc_acc']]
    ff_acc = [float(i) for i in fit_dict['ff_acc']]
    
    test_r2 = [float(i) for i in fit_dict['test_r2s']]
    pce_r2 = [float(i) for i in fit_dict['pce_r2']]
    voc_r2 = [float(i) for i in fit_dict['voc_r2']]
    jsc_r2 = [float(i) for i in fit_dict['jsc_r2']]
    ff_r2 = [float(i) for i in fit_dict['ff_r2']]
    
    train_pce_loss = [float(i) for i in fit_dict['train_pce_loss']]
    train_voc_loss = [float(i) for i in fit_dict['train_voc_loss']]
    train_jsc_loss = [float(i) for i in fit_dict['train_jsc_loss']]
    train_ff_loss = [float(i) for i in fit_dict['train_ff_loss']]

    epochs = np.arange(0, (len(test_loss)), 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12, 6))
    ax1.plot(epochs, pce_loss, c = 'r', label = 'pce loss')
    ax1.plot(epochs, voc_loss, c = 'g', label = 'voc loss')
    ax1.plot(epochs, jsc_loss, c = 'b', label = 'jsc loss')
    ax1.plot(epochs, ff_loss, c = 'c', label = 'ff loss')
    ax1.plot(epochs, test_loss, c = 'k', label = 'total loss')
    ax1.plot(epochs, train_pce_loss, c = 'r', linestyle = '-.', label = 'pce train loss')
    ax1.plot(epochs, train_voc_loss, c = 'g', linestyle = '-.', label = 'voc train loss')
    ax1.plot(epochs, train_jsc_loss, c = 'b', linestyle = '-.', label = 'jsc train loss')
    ax1.plot(epochs, train_ff_loss, c = 'c', linestyle = '-.', label = 'ff train loss')
    ax1.scatter(best_loss_epoch, min(test_loss), s = 64, c = 'c')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Mean Squared Error Loss')
    ax1.legend(loc = 'best')
    ax1.set_title(f'MSE Loss with lr = {lr}')

    ax2.plot(epochs, pce_acc, c = 'r', label = 'pce acc')
    ax2.plot(epochs, voc_acc, c = 'g', label = 'voc acc')
    ax2.plot(epochs, jsc_acc, c = 'b', label = 'jsc acc')
    ax2.plot(epochs, ff_acc, c = 'c', label = 'ff acc')
    ax2.plot(epochs, test_acc, c = 'k', label = 'total acc')
    ax2.scatter(best_acc_epoch, min(test_acc), s = 64, c = 'c')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Mean Absolute Percent Error')
    ax2.legend(loc = 'best')
    ax2.set_title(f'MAPE with lr = {lr}')

    ax3.plot(epochs, pce_r2, c = 'r', label = 'pce R$^2$')
    ax3.plot(epochs, voc_r2, c = 'g', label = 'voc R$^2$')
    ax3.plot(epochs, jsc_r2, c = 'b', label = 'jsc R$^2$')
    ax3.plot(epochs, ff_r2, c = 'c', label = 'ff R$^2$')
    ax3.plot(epochs, test_r2, c = 'k', label = 'total R$^2$')
    ax3.scatter(best_r2_epoch, max(test_r2), s = 64, c = 'c')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('R$^2$')
    ax3.legend(loc = 'best')
    ax3.set_title(f'R$^2$ with lr = {lr}')
    
    plt.tight_layout()
    plt.show()    
    
    
def plot_best_fit_lrs(fit_dict):
    """
    A function that plots the best loss, accuracy, and r2 of a lr fit series. Note
    that the epoch shown is loss's best epoch, which many not correspond to the best
    epoch for accuracy or r2.
    """
    
    #for each lr fit, collect best values
    lrs = []
    best_losses = []
    best_accs = []
    best_r2s = []
    
    for key, fit in fit_dict.items():
        lrs.append(fit['lr'])
        
        loss_ep = fit['best_loss_epoch']
        acc_ep = fit['best_acc_epoch']
        r2_ep = fit['best_r2_epoch']
        
        #these will all need to come from the same epoch during parameter selection
        best_losses.append(fit['test_losses'][loss_ep])
        best_accs.append(fit['test_accs'][acc_ep])
        best_r2s.append(fit['test_r2s'][r2_ep])
        
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12, 6))
    ax1.plot(lrs, best_losses, c = 'r')
    ax1.scatter(lrs[best_losses.index(min(best_losses))], min(best_losses), s = 64, alpha = 0.8, c = 'turquoise')
    ax1.set_xlabel('Learning Rates')
    ax1.set_ylabel('Mean Squared Error Loss')
    ax1.set_title(f'MSE Loss with lr')
    ax1.set_yscale('log')
    
    ax2.plot(lrs, best_accs, c = 'r')
    ax2.scatter(lrs[best_accs.index(min(best_accs))], min(best_accs), s = 64, alpha = 0.8, c = 'turquoise')
    ax2.set_xlabel('Learning Rates')
    ax2.set_ylabel('Mean Absolute Percent Error')
    ax2.set_title(f'MAPE with lr')
    ax2.set_yscale('log')
    
    ax3.plot(lrs, best_r2s, c = 'r')
    ax3.scatter(lrs[best_r2s.index(max(best_r2s))], max(best_r2s), s = 64, alpha = 0.8, c = 'turquoise')
    ax3.set_xlabel('Learning Rates')
    ax3.set_ylabel('R$^2$')
    ax3.set_title(f'R$^2$ with lr')
    ax3.set_ylim(-500, 1)
    
    plt.tight_layout()
    plt.show()