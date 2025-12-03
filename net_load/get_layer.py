import torch
import scipy.io as sio

# get network type from folder name, only works for fcN_tanh_mnist
def get_nettype(folder_name):
    return folder_name.split("_")[0] + "_mnist_tanh"

# extract weight parameters from epoch
def get_epoch_weights(path, folder_name, epoch):

    epoch_file = "model_" + str(epoch) + "_sub_loss_w.mat"
    epoch_weights = sio.loadmat(path + '/' + folder_name + '/' + epoch_file)

    return epoch_weights['sub_weights'][0]

def layer_names(net):
    return list(net.state_dict().keys())

# load weights into network
def load_weights(net, weights):

    weight_index_ls = layer_names(net)    
    for i in range(len(weight_index_ls)):
        layer_id = weight_index_ls[i]
        net.state_dict()[layer_id].data.copy_(torch.from_numpy(weights[i]))

    return net

# returns layer structures of network
def layer_struct(net):
    return list(net.modules())[1]

# returns all the preactivation input of the layers including the input and output (better method in Alexnet.py under class FullyConnected_tanh)
def get_hidden_layers(net, weights, x):

    wm_num = len(weights)
    net_weights = list(net.state_dict().values())
    correct = 0
    for i in range(wm_num):
        if weights[i].shape == net_weights[i].shape:
            correct += 1

    if correct != wm_num:
        return f"Correct number of weights is {wm_num}, but you have {correct}."
    else:
        net_updated = load_weights(net, weights)

        layer_outputs = []
        # include the input
        layer_outputs.append(x)
        
        layer_ls = layer_struct(net_updated)
        x = layer_ls[0](x)
        layer_outputs.append(x)

        for j in range(1,len(layer_ls)):
            layer_func = layer_ls[j]
            # might need to replace the below

            x = layer_func(x)

            if j % 2 == 0:
                layer_outputs.append(x)

    return layer_outputs