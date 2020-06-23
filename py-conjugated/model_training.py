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
    time_loss_list = []
    temp_loss_list = []
    
    model.train()
    
    batch_iterator = 0
    for images, labels in training_data_set:
        batch_iterator+=1
        print(f'image # {batch_iterator}')
#         images = images.to(device)
#         labels = labels.to(device)
        
        # Run the forward pass
                
        optimizer.zero_grad()
        im_pred, im_enc = model(images)
        print(im_pred.size())
        print(im_pred)
        print(labels.size())
        
        
        time_pred = im_pred[:,0]
        temp_pred = im_pred[:,1]
        time_label = labels[:,0]
        temp_label = labels[:,1]
        
        print("pre")
        print(time_pred.size())
        print(time_label.size())
        print(temp_pred.size())
        print(temp_label.size())
        
        #drop superfluous dimensions (e.g. batch)
        time_pred = time_pred.view(-1)
        temp_pred = temp_pred.view(-1)
        time_label = time_label.view(-1)
        temp_label = temp_label.view(-1)
        
        print("post")
        print(time_pred.size())
        print(time_label.size())
        print(temp_pred.size())
        print(temp_label.size())
        
        #Gather the loss
        time_loss_func = nn.MSELoss()
        temp_loss_func = nn.MSELoss()
        time_loss = time_loss_func(time_pred, time_label)
        temp_loss = temp_loss_func(temp_pred, temp_label)
        print("loss calculated")
        
        time_loss_list.append(time_loss.item())
        temp_loss_list.append(temp_loss.item())
        
        
#         loss = F.nll_loss(im_pred, labels)
#         loss_list.append(loss.item())
        
        # backprop and perform Adam optimization
        torch.autograd.backward(time_loss, temp_loss)
        print("backprop calculated")
#         torch.autograd.backward(loss)
        optimizer.step()
        print("end of loop {}".format(epoch))
    
    total_count = len(loss_list)
    time_epoch_loss = sum(time_loss_list)/total_count
    temp_epoch_loss = sum(test_loss_list)/total_count
    
    return time_epoch_loss, temp_epoch_loss


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


