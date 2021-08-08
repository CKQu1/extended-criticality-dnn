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

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        x = self.fc(x)
        return x

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

class FullyConnected_tanh_bias(nn.Module):

    def __init__(self, input_dim=28*28 , width=50, depth=3, num_classes=10):
        super(FullyConnected_tanh_bias, self).__init__()
        self.input_dim = input_dim 
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        
        layers = self.get_layers()

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.width, bias=True),
            nn.Tanh(),
            *layers,
            nn.Linear(self.width, self.num_classes, bias=True),
        )

    def get_layers(self):
        layers = []
        for i in range(self.depth - 2):
            layers.append(nn.Linear(self.width, self.width, bias=True))
            layers.append(nn.Tanh())
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

class LeNet5_bias(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120, bias=False)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.fc3 = nn.Linear(84, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def alexnet():
    return AlexNet(ch=64, num_classes=10)


def fc3(**kwargs):
    return FullyConnected(input_dim=32*32*3, width=100, depth=3, num_classes=10)

def fc3_sq_mnist(**kwargs):
    return FullyConnected(input_dim=28*28, width=28*28, depth=3, num_classes=10)

def fc4_sq_mnist(**kwargs):
    return FullyConnected(input_dim=28*28, width=28*28, depth=4, num_classes=10)

# FCN with square WMs (except for the last one), bias:no, activation:tanh, dataset:MNIST    
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
    
############################################################################################

def fc20(**kwargs):
    return FullyConnected(input_dim=32*32*3, width=100, depth=20, num_classes=10)

def fc56(**kwargs):
    return FullyConnected(input_dim=32*32*3, width=100, depth=56, num_classes=10)

def fc110(**kwargs):
    return FullyConnected(input_dim=32*32*3, width=100, depth=110, num_classes=10)

def simplenet(**kwargs):
    return SimpleNet(**kwargs)
