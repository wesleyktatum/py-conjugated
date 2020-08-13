"""
This module contains functions that validate neural networks for predicting device performance,
based on tabular data and m2py labels
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

module_path = os.path.abspath(os.path.join('./'))
if module_path not in sys.path:
    sys.path.append(module_path)
import physically_informed_loss_functions as pilf


def eval_OPV_df_model(model, testing_data_set):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    #evaluate the model
    model.eval()
    
#     pce_criterion = pilf.ThresholdedMSELoss(lower = 0, upper = 1.2)
#     voc_criterion = pilf.ThresholdedMSELoss(lower = 0, upper = 1.2)
#     jsc_criterion = pilf.ThresholdedMSELoss(lower = 0, upper = 1.2)
#     ff_criterion = pilf.ThresholdedMSELoss(lower = 0, upper = 1.2)
    
    pce_criterion = nn.MSELoss()
    voc_criterion = nn.MSELoss()
    jsc_criterion = nn.MSELoss()
    ff_criterion = nn.MSELoss()
    
    accuracy = pilf.MAPE()
#     accuracy = nn.L1Loss()

    #don't update nodes during evaluation b/c not training
    with torch.no_grad():
        test_losses = []
        pce_test_losses = []
        voc_test_losses = []
        jsc_test_losses = []
        ff_test_losses = []
        
        pce_test_acc_list = []
        voc_test_acc_list = []
        jsc_test_acc_list = []
        ff_test_acc_list = []
        test_accs = []
        
        pce_test_r2_list = []
        voc_test_r2_list = []
        jsc_test_r2_list = []
        ff_test_r2_list = []
        test_r2s = []
    
        test_total = 0

        for inputs, pce_labels, voc_labels, jsc_labels, ff_labels in testing_data_set:
            inputs = inputs.to(device)
            pce_labels = pce_labels.to(device)
            voc_labels = voc_labels.to(device)
            jsc_labels = jsc_labels.to(device)
            ff_labels = ff_labels.to(device)

            PCE_out, Voc_out, Jsc_out, FF_out = model(inputs)
            
            PCE_out.squeeze_(-1)
            Voc_out.squeeze_(-1)
            Jsc_out.squeeze_(-1)
            FF_out.squeeze_(-1)
    
            # calculate loss per batch of testing data
            pce_test_loss = pce_criterion(PCE_out, pce_labels)
            voc_test_loss = voc_criterion(Voc_out, voc_labels)
            jsc_test_loss = jsc_criterion(Jsc_out, jsc_labels)
            ff_test_loss = ff_criterion(FF_out, ff_labels)

            #gather the losses
            pce_test_losses.append(pce_test_loss.item())
            voc_test_losses.append(voc_test_loss.item())
            jsc_test_losses.append(jsc_test_loss.item())
            ff_test_losses.append(ff_test_loss.item())
            test_total += 1 
                        
            #calculate the accs
            pce_acc = accuracy(PCE_out, pce_labels)
            voc_acc = accuracy(Voc_out, voc_labels)
            jsc_acc = accuracy(Jsc_out, jsc_labels)
            ff_acc = accuracy(FF_out, ff_labels)
            
            #gather the accs
            pce_test_acc_list.append(pce_acc.item())
            voc_test_acc_list.append(voc_acc.item())
            jsc_test_acc_list.append(jsc_acc.item())
            ff_test_acc_list.append(ff_acc.item())
            
            #calculate the r2s
            pce_r2 = r2_score(PCE_out.data.numpy(), pce_labels.data.numpy())
            voc_r2 = r2_score(Voc_out.data.numpy(), voc_labels.data.numpy())
            jsc_r2 = r2_score(Jsc_out.data.numpy(), jsc_labels.data.numpy())
            ff_r2 = r2_score(FF_out.data.numpy(), ff_labels.data.numpy())
            
            #gather the r2s
            pce_test_r2_list.append(pce_r2)
            voc_test_r2_list.append(voc_r2)
            jsc_test_r2_list.append(jsc_r2)
            ff_test_r2_list.append(ff_r2)
            
        pce_test_epoch_loss = sum(pce_test_losses)/test_total
        voc_test_epoch_loss = sum(voc_test_losses)/test_total
        jsc_test_epoch_loss = sum(jsc_test_losses)/test_total
        ff_test_epoch_loss = sum(ff_test_losses)/test_total
        
        losses = [pce_test_epoch_loss, voc_test_epoch_loss, jsc_test_epoch_loss, ff_test_epoch_loss]
        
        pce_epoch_acc = sum(pce_test_acc_list)/test_total
        voc_epoch_acc = sum(voc_test_acc_list)/test_total
        jsc_epoch_acc = sum(jsc_test_acc_list)/test_total
        ff_epoch_acc = sum(ff_test_acc_list)/test_total 
        
        accs = [pce_epoch_acc, voc_epoch_acc, jsc_epoch_acc, ff_epoch_acc]
        
        pce_epoch_r2 = sum(pce_test_r2_list)/test_total
        voc_epoch_r2 = sum(voc_test_r2_list)/test_total
        jsc_epoch_r2 = sum(jsc_test_r2_list)/test_total
        ff_epoch_r2 = sum(ff_test_r2_list)/test_total
        
        r2s = [pce_epoch_r2, voc_epoch_r2, jsc_epoch_r2, ff_epoch_r2]
        
        print(f"Total Epoch Testing MAPE: PCE = {pce_epoch_acc}")
        print(f"                              Voc = {voc_epoch_acc}")
        print(f"                              Jsc = {jsc_epoch_acc}")
        print(f"                              FF = {ff_epoch_acc}")
    return losses, accs, r2s


def eval_OPV_m2py_model(model, test_data_set, criterion):
    
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    #evaluate the model
    model.eval()

    #don't update nodes during evaluation b/c not training
    with torch.no_grad():
        total_step = len(test_data_set)
        
        pce_loss_list = []
        voc_loss_list = []
        jsc_loss_list = []
        voc_loss_list = []
        ff_loss_list = []
        total_loss_list = []
        
        pce_acc_list = []
        voc_acc_list = []
        jsc_acc_list = []
        ff_acc_list = []
        total_acc_list = []
        
        pce_r2_list = []
        voc_r2_list = []
        jsc_r2_list = []
        ff_r2_list = []
        total_r2_list = []
        
        batch_iterator = 0
        for images, labels in test_data_set:
            batch_iterator+=1
#             print(f'testing batch # {batch_iterator}')
    #         images = images.to(device)
    #         labels = labels.to(device)

            # Run the forward pass
            pce_pred, voc_pred, jsc_pred, ff_pred = model(images)

            #calculate the loss
            pce_loss = criterion(pce_pred, labels[:,0])
            voc_loss = criterion(voc_pred, labels[:,1])
            jsc_loss = criterion(jsc_pred, labels[:,2])
            ff_loss = criterion(ff_pred, labels[:,3])

            #gather the loss
            pce_loss_list.append(pce_loss.item())
            voc_loss_list.append(voc_loss.item())
            jsc_loss_list.append(jsc_loss.item())
            voc_loss_list.append(ff_loss.item())
            
            #calculate the accs
            acc = pilf.MAPE()

            pce_acc = acc(pce_pred, labels[:,0])
            voc_acc = acc(voc_pred, labels[:,1])
            jsc_acc = acc(jsc_pred, labels[:,2])
            ff_acc = acc(ff_pred, labels[:,3])
            
            #gather the accs
            pce_acc_list.append(pce_acc.item())
            voc_acc_list.append(voc_acc.item())
            jsc_acc_list.append(jsc_acc.item())
            ff_acc_list.append(ff_acc.item())
            
            #calculate the r2s
            pce_r2 = r2_score(labels[:,0].data.numpy(), pce_pred.data.numpy())
            voc_r2 = r2_score(labels[:,1].data.numpy(), voc_pred.data.numpy())
            jsc_r2 = r2_score(labels[:,2].data.numpy(), jsc_pred.data.numpy())
            ff_r2 = r2_score(labels[:,3].data.numpy(), ff_pred.data.numpy())
            
            #gather the r2s
            pce_r2_list.append(pce_r2)
            voc_r2_list.append(voc_r2)
            jsc_r2_list.append(jsc_r2)
            ff_r2_list.append(ff_r2)

        total_count = len(pce_loss_list)
        
        pce_epoch_loss = sum(pce_loss_list)/total_count
        voc_epoch_loss = sum(voc_loss_list)/total_count
        jsc_epoch_loss = sum(jsc_loss_list)/total_count
        ff_epoch_loss = sum(ff_loss_list)/total_count
        
        losses = [pce_epoch_loss, voc_epoch_loss, jsc_epoch_loss, ff_epoch_loss]
        
        pce_epoch_acc = sum(pce_acc_list)/total_count
        voc_epoch_acc = sum(voc_acc_list)/total_count
        jsc_epoch_acc = sum(jsc_acc_list)/total_count
        ff_epoch_acc = sum(ff_acc_list)/total_count
        
        accs = [pce_epoch_acc, voc_epoch_acc, jsc_epoch_acc, ff_epoch_acc]
        
        pce_epoch_r2 = sum(pce_r2_list)/total_count
        voc_epoch_r2 = sum(voc_r2_list)/total_count
        jsc_epoch_r2 = sum(jsc_r2_list)/total_count
        ff_epoch_r2 = sum(ff_r2_list)/total_count
        
        r2s = [pce_epoch_r2, voc_epoch_r2, jsc_epoch_r2, ff_epoch_r2]


    return losses, accs, r2s


def eval_OFET_df_model(model, testing_data_set):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    #evaluate the model
    model.eval()
    
    mu_criterion = pilf.ThresholdedMSELoss(lower = 0, upper = 1.5)
    r_criterion = pilf.ThresholdedMSELoss(lower = 0, upper = 1.5)
    on_off_criterion = pilf.ThresholdedMSELoss(lower = 0, upper = 1.5)
    vt_criterion = pilf.ThresholdedMSELoss(lower = 0, upper = 1.5)
    
    accuracy = pilf.MAPE()

    #don't update nodes during evaluation b/c not training
    with torch.no_grad():
        test_losses = []
        mu_test_losses = []
        r_test_losses = []
        on_off_test_losses = []
        vt_test_losses = []
        
        mu_test_acc_list = []
        r_test_acc_list = []
        on_off_test_acc_list = []
        vt_test_acc_list = []
    
        test_total = 0

        for inputs, mu_labels, r_labels, on_off_labels, vt_labels in testing_data_set:
            inputs = inputs.to(device)
            mu_labels = mu_labels.to(device)
            r_labels = r_labels.to(device)
            on_off_labels = on_off_labels.to(device)
            vt_labels = vt_labels.to(device)

            mu_out, r_out, on_off_out, vt_out = model(inputs)

    
            # calculate loss per batch of testing data
            mu_loss = mu_criterion(mu_out, mu_labels)
            r_loss = r_criterion(r_out, r_labels)
            on_off_loss = on_off_criterion(on_off_out, on_off_labels)
            vt_loss = vt_criterion(vt_out, vt_labels)
            
            test_loss = mu_loss + r_loss + on_off_loss + vt_loss
            
            test_losses.append(test_loss.item())
            mu_test_losses.append(mu_loss.item())
            r_test_losses.append(r_loss.item())
            on_off_test_losses.append(on_off_loss.item())
            vt_test_losses.append(vt_loss.item())
            test_total += 1 
            
            mu_acc = accuracy(mu_out, mu_labels)
            r_acc = accuracy(r_out, r_labels)
            on_off_acc = accuracy(on_off_out, on_off_labels)
            vt_acc = accuracy(vt_out, vt_labels)
            
            mu_test_acc_list.append(mu_acc)
            r_test_acc_list.append(r_acc)
            on_off_test_acc_list.append(on_off_acc)
            vt_test_acc_list.append(vt_acc)

        test_epoch_loss = sum(test_losses)/test_total
        mu_test_epoch_loss = sum(mu_test_losses)/test_total
        r_test_epoch_loss = sum(r_test_losses)/test_total
        on_off_test_epoch_loss = sum(on_off_test_losses)/test_total
        vt_test_epoch_loss = sum(vt_test_losses)/test_total
        
        mu_epoch_acc = sum(mu_test_acc_list)/test_total
        r_epoch_acc = sum(r_test_acc_list)/test_total
        on_off_epoch_acc = sum(on_off_test_acc_list)/test_total
        vt_epoch_acc = sum(vt_test_acc_list)/test_total 

        print(f"Total Epoch Testing Loss = {test_epoch_loss}")
        print(f"Total Epoch Testing MAPE: mu = {mu_epoch_acc}")
        print(f"                              r = {r_epoch_acc}")
        print(f"                              on_off = {on_off_epoch_acc}")
        print(f"                              Vt = {vt_epoch_acc}")
    return test_epoch_loss, mu_test_epoch_loss, r_test_epoch_loss, on_off_test_epoch_loss, vt_test_epoch_loss, mu_epoch_acc, r_epoch_acc, on_off_epoch_acc, vt_epoch_acc


