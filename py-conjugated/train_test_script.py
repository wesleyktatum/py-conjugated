import os
import sys
import logging
import argparse
import boto3
import torch
import torch.nn as nn

module_path = os.path.abspath(os.path.join('./'))
if module_path not in sys.path:
    sys.path.append(module_path)
import morphology_networks as net
import model_training as train
import model_testing as test
import physically_informed_loss_functions as pilf
import network_utils as nuts


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def train_test_cycle(args, tracker = None):
    bucket_name = 'sagemaker-us-east-2-362637960691'
    train_data_location = 'py-conjugated/m2py_labels/OPV_labels/train_set/'
    test_data_location = 'py-conjugated/m2py_labels/OPV_labels/test_set/'
    model_states_location = 'py-conjugated/model_states/OPV/'

    client = boto3.client('s3')
    resource = boto3.resource('s3')
    s3_bucket = resource.Bucket(bucket_name)
    
    ############
    # Get the Data
    ############

    train_data = nuts.OPV_ImDataset(bucket_name, train_data_location)
    test_data = nuts.OPV_ImDataset(bucket_name, test_data_location)

    train_data_set = torch.utils.data.DataLoader(dataset = train_data,
                                                 batch_size = len(train_data),
                                                 shuffle = True)
    test_data_set = torch.utils.data.DataLoader(dataset = test_data,
                                                batch_size = len(test_data),
                                                shuffle = True)

    ##############
    #Define the model
    ##############

    im_branch_model = net.OPV_m2py_NN(2)

    #define the loss function and the optimizer
    im_criterion = nn.CrossEntropyLoss()
    im_optimizer = torch.optim.Adam(im_branch_model.parameters(), lr = args.lr)

    im_train_epoch_losses = []
    im_test_epoch_losses = []

    for epoch in range(args.epochs):
        ###############
        #Train the model
        ###############

        im_train_epoch_loss = train.train_OPV_m2py_model(model = im_branch_model,
                                       training_data_set = im_training_data_set,
                                       criterion = im_criterion,
                                       optimizer = im_optimizer)

        im_train_epoch_losses.append(im_train_epoch_loss)

        im_test_epoch_loss = test.eval_OPV_m2py_model(model = im_branch_model,
                                     testing_data_set = im_testing_data_set,
                                     criterion = im_criterion)
        
        ###############
        #Test the model
        ###############

        im_test_epoch_losses.append(im_test_epoch_loss)
        logger.debug(f'Test MSE = {im_test_epoch_loss}')
        
    
        
    epochs = np.arange(1, (args.epochs+1), 1)

    nuts.plot_OPV_df_loss(epochs, train_epoch_losses, test_epoch_losses,
                         pce_train_epoch_losses, pce_test_epoch_losses,
                         voc_train_epoch_losses, voc_test_epoch_losses,
                         jsc_train_epoch_losses, jsc_test_epoch_losses,
                         ff_train_epoch_losses, ff_test_epoch_losses)
    
    ###############
    #Evaluate the results
    ###############

    nuts.plot_OPV_df_accuracies(epochs, pce_test_epoch_accuracies, voc_test_epoch_accuracies, 
                               jsc_test_epoch_accuracies, ff_test_epoch_accuracies)
    
    im_branch_model.eval()

    with torch.no_grad():
        for inputs, pce_labels, voc_labels, jsc_labels, ff_labels in testing_data_set:
            inputs = inputs.to(device)
            pce_labels = pce_labels.to(device)
            voc_labels = voc_labels.to(device)
            jsc_labels = jsc_labels.to(device)
            ff_labels = ff_labels.to(device)

            PCE_out, Voc_out, Jsc_out, FF_out = im_branch_model(inputs)


    mape = pilf.reg_MAPE()

    pce_mse = mean_squared_error(PCE_out, pce_labels)
    pce_r2 = r2_score(PCE_out, pce_labels)
    pce_mape = mape.forward(PCE_out, pce_labels)

    print(f'mse = {pce_mse}, mape = {pce_mape}, r2 = {pce_r2}')

    voc_mse = mean_squared_error(Voc_out, voc_labels)
    voc_r2 = r2_score(Voc_out, voc_labels)
    voc_mape = mape.forward(Voc_out, voc_labels)

    print(f'mse = {voc_mse}, mape = {voc_mape}, r2 = {voc_r2}')

    jsc_mse = mean_squared_error(Jsc_out, jsc_labels)
    jsc_r2 = r2_score(Jsc_out, jsc_labels)
    jsc_mape = mape.forward(Jsc_out, jsc_labels)

    print(f'mse = {jsc_mse}, mape = {jsc_mape}, r2 = {jsc_r2}')

    ff_mse = mean_squared_error(FF_out, ff_labels)
    ff_r2 = r2_score(FF_out, ff_labels)
    ff_mape = mape.forward(FF_out, ff_labels)

    print(f'mse = {ff_mse}, mape = {ff_mape}, r2 = {ff_r2}')


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    
    return parser.parse_args


if __name__ == '__main__':
    args = parse_args()

    train(args)

    