"""
This module contains functions that train neural networks for predicting device performance,
based on tabular data and m2py labels
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

module_path = os.path.abspath(os.path.join('./'))
if module_path not in sys.path:
    sys.path.append(module_path)
import physically_informed_loss_functions as pilf


def train_OPV_df_model(model, training_data_set, optimizer):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    train_epoch_loss = []
    pce_train_epoch_loss = []
    voc_train_epoch_loss = []
    jsc_train_epoch_loss = []
    ff_train_epoch_loss = []
    
    train_losses = []
    pce_train_losses = []
    voc_train_losses = []
    jsc_train_losses = []
    ff_train_losses = []
    
    train_total = 0
    
    #switch model to training mode
    model.train()
    
    #Define boundaries for physical solutions
#     pce_criterion = pilf.ThresholdedMSELoss(lower = 0, upper = 1.5)
#     voc_criterion = pilf.ThresholdedMSELoss(lower = 0, upper = 1.5)
#     jsc_criterion = pilf.ThresholdedMSELoss(lower = 0, upper = 1.5)
#     ff_criterion = pilf.ThresholdedMSELoss(lower = 0, upper = 1.5)

    pce_criterion = nn.MSELoss()
    voc_criterion = nn.MSELoss()
    jsc_criterion = nn.MSELoss()
    ff_criterion = nn.MSELoss()
    
    for train_data, pce_labels, voc_labels, jsc_labels, ff_labels in training_data_set:
        
        train_data = train_data.to(device)
        pce_labels = pce_labels.to(device)
        voc_labels = voc_labels.to(device)
        jsc_labels = jsc_labels.to(device)
        ff_labels = ff_labels.to(device)
                
        model.zero_grad() #zero out any gradients from prior loops
        
        #gather model predictions for this loop
        PCE_out, Voc_out, Jsc_out, FF_out = model(train_data) 
        
        #calculate error in the predictions
        pce_loss = pce_criterion(PCE_out, pce_labels)
        voc_loss = voc_criterion(Voc_out, voc_labels)
        jsc_loss = jsc_criterion(Jsc_out, jsc_labels)
        ff_loss = ff_criterion(FF_out, ff_labels)
                
        total_loss = pce_loss + voc_loss + jsc_loss + ff_loss
        
        #BACKPROPOGATE LIKE A MF
        torch.autograd.backward([pce_loss, voc_loss, jsc_loss, ff_loss])
        optimizer.step()
        
        #save loss for this batch
        train_losses.append(total_loss.item())
        train_total+=1
        
        pce_train_losses.append(pce_loss.item())
        voc_train_losses.append(voc_loss.item())
        jsc_train_losses.append(jsc_loss.item())
        ff_train_losses.append(ff_loss.item())
        
    #calculate and save total error for this epoch of training
    epoch_loss = sum(train_losses)/train_total
    train_epoch_loss.append(epoch_loss)
    
    pce_train_epoch_loss = sum(pce_train_losses)/train_total
    voc_train_epoch_loss = sum(voc_train_losses)/train_total
    jsc_train_epoch_loss = sum(jsc_train_losses)/train_total
    ff_train_epoch_loss = sum(ff_train_losses)/train_total
    
    return model, train_epoch_loss, pce_train_epoch_loss, voc_train_epoch_loss, jsc_train_epoch_loss, ff_train_epoch_loss


def train_OPV_m2py_model(model, training_data_set, criterion, optimizer):
    
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    total_step = len(training_data_set)
    pce_loss_list = []
    voc_loss_list = []
    jsc_loss_list = []
    voc_loss_list = []
    ff_loss_list = []
    total_loss_list = []
    
    model.train()
    
    batch_iterator = 0
    for images, labels in training_data_set:
        batch_iterator+=1
#         print(f'training batch # {batch_iterator}')
#         images = images.to(device)
#         labels = labels.to(device)
        
        # Run the forward pass   
        model.zero_grad()
        pce_pred, voc_pred, jsc_pred, ff_pred = model(images)
        
        pce_labels = labels[:,0].squeeze()
        voc_labels = labels[:,1].squeeze()
        jsc_labels = labels[:,2].squeeze()
        ff_labels = labels[:,3].squeeze()
        
        #Gather the loss
        pce_loss = criterion(pce_pred, pce_labels)
        voc_loss = criterion(voc_pred, voc_labels)
        jsc_loss = criterion(jsc_pred, jsc_labels)
        ff_loss = criterion(ff_pred, ff_labels)
        
        total_loss = pce_loss + voc_loss + jsc_loss + ff_loss
        
        #BACKPROPOGATE LIKE A MF
        torch.autograd.backward([pce_loss, voc_loss, jsc_loss, ff_loss])
        optimizer.step()
        
        #gather the loss
        pce_loss_list.append(pce_loss.item())
        voc_loss_list.append(voc_loss.item())
        jsc_loss_list.append(jsc_loss.item())
        voc_loss_list.append(ff_loss.item())
        total_loss_list.append(total_loss.item())
    
    total_count = len(total_loss_list)
    pce_epoch_loss = sum(pce_loss_list)/total_count
    voc_epoch_loss = sum(voc_loss_list)/total_count
    jsc_epoch_loss = sum(jsc_loss_list)/total_count
    ff_epoch_loss = sum(ff_loss_list)/total_count
    
    return [pce_epoch_loss, voc_epoch_loss, jsc_epoch_loss, ff_epoch_loss]


def train_OFET_df_model(model, training_data_set, optimizer):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    train_epoch_loss = []
    mu_train_epoch_loss = []
    r_train_epoch_loss = []
    on_off_train_epoch_loss = []
    vt_train_epoch_loss = []
    
    train_losses = []
    mu_train_losses = []
    r_train_losses = []
    on_off_train_losses = []
    vt_train_losses = []
    
    train_total = 0
    
    #switch model to training mode
    model.train()
    
#     mu_criterion = pilf.ThresholdedMSELoss(lower = -2, upper = 4)
#     r_criterion = pilf.ThresholdedMSELoss(lower = -3, upper = 3)
#     on_off_criterion = pilf.ThresholdedMSELoss(lower = -0.75, upper = 6.5)
#     vt_criterion = pilf.ThresholdedMSELoss(lower = -2, upper = 6)

    mu_criterion = nn.MSELoss()
    r_criterion = nn.MSELoss()
    on_off_criterion = nn.MSELoss()
    vt_criterion = nn.MSELoss()
    
    for train_data, mu_labels, r_labels, on_off_labels, vt_labels in training_data_set:
        
        train_data = train_data.to(device)
        mu_labels = mu_labels.to(device)
        r_labels = r_labels.to(device)
        on_off_labels = on_off_labels.to(device)
        vt_labels = vt_labels.to(device)
        
        model.zero_grad() #zero out any gradients from prior loops 
        mu_out, r_out, on_off_out, vt_out = model(train_data) #gather model predictions for this loop
        
        #calculate error in the predictions
        mu_loss = mu_criterion(mu_out, mu_labels)
        r_loss = r_criterion(r_out, r_labels)
        on_off_loss = on_off_criterion(on_off_out, on_off_labels)
        vt_loss = vt_criterion(vt_out, vt_labels)
        
        total_loss = mu_loss + r_loss + on_off_loss + vt_loss
        
        #BACKPROPOGATE LIKE A MF
        torch.autograd.backward([mu_loss, r_loss, on_off_loss, vt_loss])
        optimizer.step()
        
        #save loss for this batch
        train_losses.append(total_loss.item())
        train_total+=1
        
        mu_train_losses.append(mu_loss.item())
        r_train_losses.append(r_loss.item())
        on_off_train_losses.append(on_off_loss.item())
        vt_train_losses.append(vt_loss.item())
        
    #calculate and save total error for this epoch of training
    epoch_loss = sum(train_losses)/train_total
    train_epoch_loss.append(epoch_loss)
    
    mu_train_epoch_loss.append(sum(mu_train_losses)/train_total)
    r_train_epoch_loss.append(sum(r_train_losses)/train_total)
    on_off_train_epoch_loss.append(sum(on_off_train_losses)/train_total)
    vt_train_epoch_loss.append(sum(vt_train_losses)/train_total)
    
    return model, train_epoch_loss, mu_train_epoch_loss, r_train_epoch_loss, on_off_train_epoch_loss, vt_train_epoch_loss


