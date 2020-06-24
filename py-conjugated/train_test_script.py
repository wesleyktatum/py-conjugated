import os
import sys
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
    
    lr = args.lr
    epochs = args.epochs

    #define the loss function and the optimizer
    im_criterion = nn.MSELoss()
    im_optimizer = torch.optim.Adam(im_branch_model.parameters(), lr = lr)

    time_epoch_losses_train = []
    temp_epoch_losses_train = []
    time_epoch_losses_test = []
    temp_epoch_losses_test = []
    total_epoch_losses = []

    for epoch in range(epochs):
        ###############
        #Train the model
        ###############

        time_epoch_loss_train, temp_epoch_loss_train = train.train_OPV_m2py_model(model = im_branch_model,
                                                                                  training_data_set = train_data_set,
                                                                                  criterion = im_criterion,
                                                                                  optimizer = im_optimizer)

        time_epoch_losses_train.append(time_epoch_loss_train)
        temp_epoch_losses_train.append(temp_epoch_loss_train)
        print('finished testing epoch {}'.format(epoch))
        
        ###############
        #Test the model
        ###############
        
        time_epoch_loss_test, temp_epoch_loss_test = test.eval_OPV_m2py_model(model = im_branch_model,
                                                                              testing_data_set = test_data_set,
                                                                              criterion = im_criterion)

        time_epoch_losses_test.append(time_epoch_loss_test)
        temp_epoch_losses_test.append(temp_epoch_loss_test)
        
        total_epoch_loss = time_epoch_loss_test + temp_epoch_loss_test
        total_epoch_losses.append(total_epoch_loss)
        
        print('Test - Time MSE = {};'.format(time_epoch_loss_test))
        print('Test - Temp MSE = {};'.format(temp_epoch_loss_test))
        print('Test - Total MSE = {};'.format(total_epoch_loss))
        
    
        
    epochs = np.arange(1, (epochs+1), 1)

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

    print('PCE: mse = {}, mape = {}, r2 = {}'.format(pce_mse, pce_mape, pce_r2))

    voc_mse = mean_squared_error(Voc_out, voc_labels)
    voc_r2 = r2_score(Voc_out, voc_labels)
    voc_mape = mape.forward(Voc_out, voc_labels)

    print('Voc: mse = {}, mape = {}, r2 = {}'.format(voc_mse, voc_mape, voc_r2))

    jsc_mse = mean_squared_error(Jsc_out, jsc_labels)
    jsc_r2 = r2_score(Jsc_out, jsc_labels)
    jsc_mape = mape.forward(Jsc_out, jsc_labels)

    print('Jsc: mse = {}, mape = {}, r2 = {}'.format(jsc_mse, jsc_mape, jsc_r2))

    ff_mse = mean_squared_error(FF_out, ff_labels)
    ff_r2 = r2_score(FF_out, ff_labels)
    ff_mape = mape.forward(FF_out, ff_labels)

    print('FF: mse = {}, mape = {}, r2 = {}'.format(ff_mse, ff_mape, ff_r2))
    
    total_MAPE = pce_mape + voc_mape + jsc_mape + ff_mape
    total_r2 = pce_r2 + voc_r2 + jsc_r2 + ff_r2
    
    print('Test - Total MAPE = {};'.format(total_MAPE))
    print('Test - Total r2 = {};'.format(total_r2))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()

    train_test_cycle(args)

    