"""
This module contains functions that train neural networks for predicting device performance,
based on tabular data and m2py labels
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

module_path = os.path.abspath(os.path.join('./'))
if module_path not in sys.path:
    sys.path.append(module_path)
import physically_informed_loss_functions as PhysLoss


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
    pce_criterion = PhysLoss.ThresholdedMSELoss(lower = -5, upper = 1.5)
    voc_criterion = PhysLoss.ThresholdedMSELoss(lower = -5, upper = 1.5)
    jsc_criterion = PhysLoss.ThresholdedMSELoss(lower = -5, upper = 1.5)
    ff_criterion = PhysLoss.ThresholdedMSELoss(lower = -5, upper = 1.5)

#     pce_criterion = nn.MSELoss()
#     voc_criterion = nn.MSELoss()
#     jsc_criterion = nn.MSELoss()
#     ff_criterion = nn.MSELoss()
    
    for train_data, pce_labels, voc_labels, jsc_labels, ff_labels in training_data_set:
        
        train_data = train_data.to(device)
        pce_labels = pce_labels.to(device)
        voc_labels = voc_labels.to(device)
        jsc_labels = jsc_labels.to(device)
        ff_labels = ff_labels.to(device)
        
        model.zero_grad() #zero out any gradients from prior loops 
        PCE_out, Voc_out, Jsc_out, FF_out = model(train_data) #gather model predictions for this loop
        
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
    
    pce_train_epoch_loss.append(sum(pce_train_losses)/train_total)
    voc_train_epoch_loss.append(sum(voc_train_losses)/train_total)
    jsc_train_epoch_loss.append(sum(jsc_train_losses)/train_total)
    ff_train_epoch_loss.append(sum(ff_train_losses)/train_total)
    
    return model, train_epoch_loss, pce_train_epoch_loss, voc_train_epoch_loss, jsc_train_epoch_loss, ff_train_epoch_loss


def train_OPV_m2py_model(model, training_data_set, criterion, optimizer):
    
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    total_step = len(training_data_set)
    loss_list = []
    
    model.train()

    for images, labels in training_data_set:
#         images = images.to(device)
#         labels = labels.to(device)
        
        # Run the forward pass
        im_out, im_train_out = model(images)
        optimizer.zero_grad()
        
        # Gather the loss
        loss = criterion(im_train_out, labels)
        loss_list.append(loss.item())

        # backprop and perform Adam optimization
        torch.autograd.backward(loss)
        optimizer.step()
    
    total = len(loss_list)
    epoch_loss = sum(loss_list)/total
    
    return model, epoch_loss


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
    
#     mu_criterion = PhysLoss.ThresholdedMSELoss(lower = -2, upper = 4)
#     r_criterion = PhysLoss.ThresholdedMSELoss(lower = -3, upper = 3)
#     on_off_criterion = PhysLoss.ThresholdedMSELoss(lower = -0.75, upper = 6.5)
#     vt_criterion = PhysLoss.ThresholdedMSELoss(lower = -2, upper = 6)

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


