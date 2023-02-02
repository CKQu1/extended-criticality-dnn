# Identical copies of two AlexNet models
import torch
import torch.nn as nn
import copy 
import torch.nn.functional as F
import torch.optim as optim

class FullyConnected(nn.Module):

    def __init__(self, input_dim=28*28 , width=50, depth=3, num_classes=10):
        super(FullyConnected, self).__init__()
        self.input_dim = input_dim 
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        
        layers = self.get_layers()

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.width, bias=False),
            nn.ReLU(inplace=True),
            *layers,
            nn.Linear(self.width, self.num_classes, bias=False),
        )

    def get_layers(self):
        layers = []
        for i in range(self.depth - 2):
            layers.append(nn.Linear(self.width, self.width, bias=False))
            layers.append(nn.ReLU())
        return layers
    """
    # new function that spits out the outputs for each layer including the first and final layers
    def layer_output(self, inputs):
        hidden_layers = []
        hidden_layers.append(inputs)

        for i in range(self.depth - 2):
            layers.append(nn.Linear(self.width, self.width, bias=False))
            layers.append(nn.ReLU())

        return hidden_layers
    """

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        x = self.fc(x)
        return x

# derivative for tanh, only for scalar output
def dtanh(x):
    m = nn.Tanh()
    x = torch.autograd.Variable(x, requires_grad=True)
    y = m(x)
    y.backward( torch.ones_like(x) )

    return x.grad

class FullyConnected_tanh(nn.Module):

    def __init__(self, input_dim=28*28 , width=50, depth=3, num_classes=10):
        super(FullyConnected_tanh, self).__init__()
        self.input_dim = input_dim 
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        
        layers = self.get_layers()

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.width, bias=False),
            nn.Tanh(),
            *layers,
            nn.Linear(self.width, self.num_classes, bias=False),
        )

    def get_layers(self):
        layers = []
        for i in range(self.depth - 2):
            layers.append(nn.Linear(self.width, self.width, bias=False))
            layers.append(nn.Tanh())
        return layers

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        x = self.fc(x)
        return x

    # preactivation outputs
    def preact_layer(self, x):
        # number of hidden layers
        hidden = [None] * (self.depth - 1)
        x = x.view(x.size(0), -1)
        ell = 2

        for idx in range(self.depth - 1):
            if idx == 0:
                hidden[idx] = self.fc[idx](x)
            else:
                hidden[idx] = self.fc[ell * idx - 1: ell * idx + 1](hidden[idx - 1])

        return hidden + [self.fc[-2:](hidden[-1])]
    
    # the problem for this is that the last weight matrix in all these settings are not symmetrical, i.e. for mnist W_L is 784 by 10 (might need to adjust this in the future)
    # final argument decides whether it is post-activation or not (pre-activation)
    def layerwise_jacob_ls(self, x, post):
        # check "The Emergence of Spectral Universality in Deep Networks"
        
        m = nn.Tanh()
        preact_h = self.preact_layer(x)     # do not include the input layer check eq (1) and (2) of "Emergence of Spectral Universality in Deep Networks"     
        # get weights
        weights = [p for p in self.parameters()]
        


        if post:
            dphi_h = dtanh(preact_h[0][0])
            DW_l = torch.matmul(torch.diag( dphi_h ), weights[0])
            #DW_l = torch.matmul(torch.diag( m(preact_h[0][0]) ), weights[0])
            DW_ls = [DW_l]

            for i in range(1, len(preact_h)):   # include the last one even if it is not symmetrical
            #for i in range(1, len(preact_h) - 1):
                dphi_h = dtanh(preact_h[i][0])
                if i != len(preact_h) - 1:
                    DW_l = torch.matmul(torch.diag( dphi_h ), weights[i])
                else:
                    DW_l = torch.matmul(torch.diag( torch.ones_like(dphi_h) ), weights[i])

                #DW_l = torch.matmul(torch.diag( m(preact_h[i][0]) ), weights[i])
                DW_ls.append(DW_l)

        else:
            # essentially the first activation function is just the identity map
            DW_ls = [weights[0]]    # it's actually W^{l + 1} D^l

            for i in range(0, len(preact_h) - 1):
                dphi_h = dtanh(preact_h[i][0])
                DW_l = torch.matmul(weights[i + 1], torch.diag( dphi_h ))
                #print(DW_l.shape)
                DW_ls.append(DW_l)
            
        return DW_ls

    # postactivation outputs
    def postact_layer(self, x):
        # number of hidden layers
        hidden = [None] * (self.depth - 1)
        x = x.view(x.size(0), -1)
        ell = 2

        for idx in range(self.depth - 1):
            if idx == 0:
                hidden[idx] = self.fc[ell * idx: ell * idx + ell](x)
            else:
                hidden[idx] = self.fc[ell * idx: ell * idx + ell](hidden[idx - 1])

        return hidden, self.sequential[-1](hidden[-1])

# fully connected with activation to the last layer
class FullyConnected_tanh_2(nn.Module):

    def __init__(self, input_dim=28*28 , width=50, depth=3, num_classes=10):
        super(FullyConnected_tanh_2, self).__init__()
        self.input_dim = input_dim 
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        
        layers = self.get_layers()

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.width, bias=False),
            nn.Tanh(),
            *layers,
            nn.Linear(self.width, self.num_classes, bias=False),
            nn.Tanh(),
        )

    def get_layers(self):
        layers = []
        for i in range(self.depth - 2):
            layers.append(nn.Linear(self.width, self.width, bias=False))
            layers.append(nn.Tanh())
        return layers

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        x = self.fc(x)
        return x

    # preactivation outputs
    def preact_layer(self, x):
        # number of hidden layers
        hidden = [None] * (self.depth - 1)
        x = x.view(x.size(0), -1)
        ell = 2

        for idx in range(self.depth - 1):
            if idx == 0:
                hidden[idx] = self.fc[idx](x)
            else:
                hidden[idx] = self.fc[ell * idx - 1: ell * idx + 1](hidden[idx - 1])

        return hidden + [self.fc[-2:](hidden[-1])]
    
    # the problem for this is that the last weight matrix in all these settings are not symmetrical, i.e. for mnist W_L is 784 by 10 (might need to adjust this in the future)
    def jacob_ls(self, x):
        # check "The Emergence of Spectral Universality in Deep Networks"
        
        m = nn.Tanh()
        preact_h = self.preact_layer(x)     
        # get weights
        weights = [p for p in self.parameters()]
        
        dphi_h = dtanh(preact_h[0][0])
        DW_l = torch.matmul(torch.diag( dphi_h ), weights[0])
        #DW_l = torch.matmul(torch.diag( m(preact_h[0][0]) ), weights[0])
        DW_ls = [DW_l]

        # due to the last matrix being non-square, the case l = L is not included
        for i in range(1, len(preact_h) - 1):
            dphi_h = dtanh(preact_h[i][0])
            DW_l = torch.matmul(torch.diag( dphi_h ), weights[i])
            #DW_l = torch.matmul(torch.diag( m(preact_h[i][0]) ), weights[i])
            DW_ls.append(DW_l)

        return DW_ls

    # postactivation outputs
    """
    def postact_layer(self, x):
        # number of hidden layers
        hidden = [None] * (self.depth - 1)
        x = x.view(x.size(0), -1)
        ell = 2

        for idx in range(self.depth - 1):
            if idx == 0:
                hidden[idx] = self.fc[ell * idx: ell * idx + ell](x)
            else:
                hidden[idx] = self.fc[ell * idx: ell * idx + ell](hidden[idx - 1])

        return hidden, self.sequential[-1](hidden[-1])
    """

class FullyConnected_bias(nn.Module):

    def __init__(self, input_dim=28*28 , width=50, depth=3, num_classes=10):
        super(FullyConnected_bias, self).__init__()
        self.input_dim = input_dim 
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        
        layers = self.get_layers()

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.width, bias=True),
            #nn.ReLU(inplace=True),
            nn.Sigmoid(),
            *layers,
            nn.Linear(self.width, self.num_classes, bias=True),
        )

    def get_layers(self):
        layers = []
        for i in range(self.depth - 2):
            layers.append(nn.Linear(self.width, self.width, bias=True))
            layers.append(nn.Sigmoid())
        return layers

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        x = self.fc(x)
        return x

class FullyConnected_sigmoid(nn.Module):

    def __init__(self, input_dim=28*28 , width=50, depth=3, num_classes=10):
        super(FullyConnected_sigmoid, self).__init__()
        self.input_dim = input_dim 
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        
        layers = self.get_layers()

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.width, bias=False),
            #nn.ReLU(inplace=True),
            nn.Sigmoid(),
            *layers,
            nn.Linear(self.width, self.num_classes, bias=False),
        )

    def get_layers(self):
        layers = []
        for i in range(self.depth - 2):
            layers.append(nn.Linear(self.width, self.width, bias=False))
            layers.append(nn.Sigmoid())
        return layers

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        x = self.fc(x)
        return x


# This is a copy from online repositories 
class AlexNet(nn.Module):

    def __init__(self, input_height=32, input_width=32, input_channels=3, ch=64, num_classes=1000):
        # ch is the scale factor for number of channels
        super(AlexNet, self).__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels

        self.features = nn.Sequential(
            nn.Conv2d(3, out_channels=ch, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.size = self.get_size()
        print(self.size)
        a = torch.tensor(self.size).float()
        b = torch.tensor(2).float()
        self.width = int(a) * int(1 + torch.log(a) / torch.log(b))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.size, self.width),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(self.width, self.width),
            nn.ReLU(inplace=True),
            nn.Linear(self.width, num_classes),
        )

    def get_size(self):
        # hack to get the size for the FC layer...
        x = torch.randn(1, self.input_channels, self.input_height, self.input_width)
        y = self.features(x)
        print(y.size())
        return y.view(-1).size(0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def alexnet():
    return AlexNet(ch=64, num_classes=10)

# FCN with square WMs (except for the last one), bias:no, activation:tanh, dataset:MNIST   
def fc2_mnist_tanh(**kwargs):
    return FullyConnected_tanh(input_dim=28*28, width=100, depth=2, num_classes=10)
 
def fc3_mnist_tanh(**kwargs):
    return FullyConnected_tanh(input_dim=28*28, width=28*28, depth=3, num_classes=10)

def fc4_mnist_tanh(**kwargs):
    return FullyConnected_tanh(input_dim=28*28, width=28*28, depth=4, num_classes=10)

def fc5_mnist_tanh(**kwargs):
    return FullyConnected_tanh(input_dim=28*28, width=28*28, depth=5, num_classes=10)
    
def fc6_mnist_tanh(**kwargs):
    return FullyConnected_tanh(input_dim=28*28, width=28*28, depth=6, num_classes=10)
    
def fc7_mnist_tanh(**kwargs):
    return FullyConnected_tanh(input_dim=28*28, width=28*28, depth=7, num_classes=10)
    
def fc8_mnist_tanh(**kwargs):
    return FullyConnected_tanh(input_dim=28*28, width=28*28, depth=8, num_classes=10)
    
def fc9_mnist_tanh(**kwargs):
    return FullyConnected_tanh(input_dim=28*28, width=28*28, depth=9, num_classes=10)
    
def fc10_mnist_tanh(**kwargs):
    return FullyConnected_tanh(input_dim=28*28, width=28*28, depth=10, num_classes=10)
    
def fc15_mnist_tanh(**kwargs):
    return FullyConnected_tanh(input_dim=28*28, width=28*28, depth=15, num_classes=10)

def fc20_mnist_tanh(**kwargs):
    return FullyConnected_tanh(input_dim=28*28, width=28*28, depth=20, num_classes=10)

def fc25_mnist_tanh(**kwargs):
    return FullyConnected_tanh(input_dim=28*28, width=28*28, depth=25, num_classes=10)

def fc30_mnist_tanh(**kwargs):
    return FullyConnected_tanh(input_dim=28*28, width=28*28, depth=30, num_classes=10)

# FCN with square WMs (except for the last one), bias:no, activation:tanh, dataset:MNIST, additional tanh added to last layer    
def fc3_mnist_tanh_2(**kwargs):
    return FullyConnected_tanh_2(input_dim=28*28, width=28*28, depth=3, num_classes=10)

def fc4_mnist_tanh_2(**kwargs):
    return FullyConnected_tanh_2(input_dim=28*28, width=28*28, depth=4, num_classes=10)

def fc5_mnist_tanh_2(**kwargs):
    return FullyConnected_tanh_2(input_dim=28*28, width=28*28, depth=5, num_classes=10)
    
def fc6_mnist_tanh_2(**kwargs):
    return FullyConnected_tanh_2(input_dim=28*28, width=28*28, depth=6, num_classes=10)
    
def fc7_mnist_tanh_2(**kwargs):
    return FullyConnected_tanh_2(input_dim=28*28, width=28*28, depth=7, num_classes=10)
    
def fc8_mnist_tanh_2(**kwargs):
    return FullyConnected_tanh_2(input_dim=28*28, width=28*28, depth=8, num_classes=10)
    
def fc9_mnist_tanh_2(**kwargs):
    return FullyConnected_tanh_2(input_dim=28*28, width=28*28, depth=9, num_classes=10)
    
def fc10_mnist_tanh_2(**kwargs):
    return FullyConnected_tanh_2(input_dim=28*28, width=28*28, depth=10, num_classes=10)
    
def fc15_mnist_tanh_2(**kwargs):
    return FullyConnected_tanh_2(input_dim=28*28, width=28*28, depth=15, num_classes=10)

def fc20_mnist_tanh_2(**kwargs):
    return FullyConnected_tanh_2(input_dim=28*28, width=28*28, depth=20, num_classes=10)

def fc25_mnist_tanh_2(**kwargs):
    return FullyConnected_tanh_2(input_dim=28*28, width=28*28, depth=25, num_classes=10)

def fc30_mnist_tanh_2(**kwargs):
    return FullyConnected_tanh_2(input_dim=28*28, width=28*28, depth=30, num_classes=10)
    
# FCN with square WMs (except for the last one), bias:yes, activation:tanh, dataset:MNIST  
def fc5_mnist_tanh_bias(**kwargs):
    return FullyConnected_tanh_bias(input_dim=28*28, width=28*28, depth=5, num_classes=10)

def fc10_ns_mnist_tanh(**kwargs):
    return FullyConnected_tanh(input_dim=28*28, width=100, depth=10, num_classes=10)
    
def fc15_tanh(**kwargs):
    return FullyConnected_tanh(input_dim=32*32*3, width=100, depth=15, num_classes=10)

def fc15_mnist_tanh_bias(**kwargs):
    return FullyConnected_tanh_bias(input_dim=28*28, width=28*28, depth=15, num_classes=10)

def fc15_ns_mnist_tanh_bias(**kwargs):
    return FullyConnected_tanh(input_dim=28*28, width=100, depth=15, num_classes=10)


def fc2(**kwargs):
    return FullyConnected(input_dim=32*32*3, width=100, depth=2, num_classes=10)

def fc3(**kwargs):
    return FullyConnected(input_dim=32*32*3, width=100, depth=3, num_classes=10)

def fc3_sq_bias(**kwargs):
    return FullyConnected_bias(input_dim=32*32*3, width=100, depth=3, num_classes=10)

def fc3_sigmoid(**kwargs):
    return FullyConnected_sigmoid(input_dim=32*32*3, width=100, depth=3, num_classes=10)

def fc4(**kwargs):
    return FullyConnected(input_dim=32*32*3, width=100, depth=4, num_classes=10)

def fc5(**kwargs):
    return FullyConnected(input_dim=32*32*3, width=100, depth=5, num_classes=10)

def fc6(**kwargs):
    return FullyConnected(input_dim=32*32*3, width=100, depth=6, num_classes=10)

def fc7(**kwargs):
    return FullyConnected(input_dim=32*32*3, width=100, depth=7, num_classes=10)

def fc20(**kwargs):
    return FullyConnected(input_dim=32*32*3, width=100, depth=20, num_classes=10)

def fc56(**kwargs):
    return FullyConnected(input_dim=32*32*3, width=100, depth=56, num_classes=10)

def fc110(**kwargs):
    return FullyConnected(input_dim=32*32*3, width=100, depth=110, num_classes=10)

def simplenet(**kwargs):
    return SimpleNet(**kwargs)
