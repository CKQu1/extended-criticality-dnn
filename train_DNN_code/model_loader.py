import os
import torch, torchvision
import train_DNN_code.models.vgg as vgg
import train_DNN_code.models.resnet as resnet
import train_DNN_code.models.densenet as densenet
import train_DNN_code.models.Alexnet as Alexnet

# map between model name and function
models = {
    'fc3'                   : Alexnet.fc3,
    'fc3_sq_mnist'          : Alexnet.fc3_sq_mnist,
    'fc4_sq_mnist'          : Alexnet.fc4_sq_mnist,
    # FCN with square WMs (except for the last one), bias:no, activation:tanh, dataset:MNIST
    'fc3_mnist_tanh'        : Alexnet.fc3_mnist_tanh,
    'fc4_mnist_tanh'        : Alexnet.fc4_mnist_tanh,
    'fc5_mnist_tanh'        : Alexnet.fc5_mnist_tanh,
    'fc6_mnist_tanh'        : Alexnet.fc6_mnist_tanh,
    'fc7_mnist_tanh'        : Alexnet.fc7_mnist_tanh,
    'fc8_mnist_tanh'        : Alexnet.fc8_mnist_tanh,
    'fc9_mnist_tanh'        : Alexnet.fc9_mnist_tanh,
    'fc10_mnist_tanh'       : Alexnet.fc10_mnist_tanh,
    'fc15_mnist_tanh'       : Alexnet.fc15_mnist_tanh,
    'fc20_mnist_tanh'       : Alexnet.fc20_mnist_tanh,
    ##################################################
    'fc5_mnist_tanh_bias'   : Alexnet.fc5_mnist_tanh_bias,
    'fc5_nq_mnist_tanh_bias': Alexnet.fc5_mnist_tanh_bias,
    #'fc6_sq_mnist'          : Alexnet.fc6_sq_mnist,
    'fc10_ns_mnist_tanh'    : Alexnet.fc10_ns_mnist_tanh,
    'fc15_tanh'             : Alexnet.fc15_tanh,
    'fc15_mnist_tanh_bias'  : Alexnet.fc15_mnist_tanh_bias,
    'fc20'                  : Alexnet.fc20,
    'fc56'                  : Alexnet.fc56,
    'fc110'                 : Alexnet.fc110,
    
}

def load(model_name, model_file=None, data_parallel=False):
    net = models[model_name]()
    if data_parallel: # the model is saved in data paralle mode
        net = torch.nn.DataParallel(net)

    if model_file:
        assert os.path.exists(model_file), model_file + " does not exist."
        stored = torch.load(model_file, map_location=lambda storage, loc: storage)
        if 'state_dict' in stored.keys():
            net.load_state_dict(stored['state_dict'])
        else:
            net.load_state_dict(stored)

    if data_parallel: # convert the model back to the single GPU version
        net = net.module

    net.eval()
    return net
