import boto3
import torch
import torch.nn as nn

module_path = os.path.abspath(os.path.join('./'))
if module_path not in sys.path:
    sys.path.append(module_path)
import morphology_networks as net
import model_training as train
import model_testing as test
import physically_informed_loss_functions as PhysLoss
import network_utils as nuts


############
# Get the Data
############


if __name__ == '__main__':
    
    bucket_name = 'sagemaker-us-east-2-362637960691'
    train_data_location = 'py-conjugated/m2py_labels/OPV_labels/train_set/'
    test_data_location = 'py-conjugated/m2py_labels/OPV_labels/test_set/'
    model_states_location = 'py-conjugated/model_states/OPV/'

    client = boto3.client('s3')
    resource = boto3.resource('s3')
    s3_bucket = resource.Bucket(bucket_name)

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
    im_optimizer = torch.optim.Adam(im_branch_model.parameters(), lr = lr)

    ###############
    #Train the model
    ###############

    im_train_epoch_losses = []
    im_test_epoch_losses = []

    for epoch in range(epochs):

        im_train_epoch_loss = train.train_OPV_m2py_model(model = im_branch_model,
                                       training_data_set = im_training_data_set,
                                       criterion = im_criterion,
                                       optimizer = im_optimizer)

        im_train_epoch_losses.append(im_train_epoch_loss)

        im_test_epoch_loss = test.eval_OPV_m2py_model(model = im_branch_model,
                                     testing_data_set = im_testing_data_set,
                                     criterion = im_criterion)

        im_test_epoch_losses.append(im_test_epoch_loss)