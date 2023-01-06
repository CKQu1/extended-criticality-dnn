"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import levy_stable

# HT-initialization for nn.Conv2d structures
def conv2d_ht(module: nn.Conv2d, alpha, g):
    with torch.no_grad():
        size = [module.out_channels, module.in_channels]
        for dim in module.kernel_size:
            size.append(dim)
        size = tuple(size)
        # old
        #N_eff = np.prod(size)**0.5
        # new
        N_eff = int(np.prod(module.kernel_size)) * module.in_channels     # most correct
        module.weight = nn.Parameter(torch.Tensor(levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=g*(0.5/N_eff)**(1./alpha),
                                    size=size)))   

    return module  

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, alpha, g, activation, in_channels, out_channels, stride=1):
        super().__init__()

        self.alpha = alpha
        self.g = g
        self.activation = activation

        if self.activation=='tanh':
            #a_func=F.tanh
            a_func = nn.Tanh()
        elif self.activation=='relu':
            #a_func=F.relu
            a_func = nn.ReLU(inplace=True)
        else:
            raise NameError("activation does not exist in NetPortal.architectures")     

        #residual function
        self.residual_function = nn.Sequential(
            conv2d_ht(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False), alpha, g),
            nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True),
            a_func,
            conv2d_ht(nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False), alpha, g),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        """
        for idx in len(self.residual_function):
            if isinstance(self.residual_function[idx], nn.Conv2d):
                self.residual_function[idx] = conv2d_ht(self.residual_function[idx], alpha, g)
        """

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                conv2d_ht(nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False), alpha, g),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
            """
            for idx in len(self.shortcut):
                if isinstance(self.shortcut[idx], nn.Conv2d):
                    self.shortcut[idx] = conv2d_ht(self.shortcut[idx], alpha, g)
            """


    def forward(self, x):
        if self.activation=='tanh':
            #a_func=F.tanh
            a_func = nn.Tanh()
        elif self.activation=='relu':
            #a_func=F.relu
            a_func = nn.ReLU(inplace=True)
        else:
            raise NameError("activation does not exist in NetPortal.architectures")    

        #return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
        return a_func(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, alpha, g, activation, in_channels, out_channels, stride=1):
        super().__init__()

        self.alpha = alpha
        self.g = g
        self.activation = activation

        if self.activation=='tanh':
            #a_func=F.tanh
            a_func = nn.Tanh()
        elif self.activation=='relu':
            #a_func=F.relu
            a_func = nn.ReLU(inplace=True)
        else:
            raise NameError("activation does not exist in NetPortal.architectures")     

        self.residual_function = nn.Sequential(
            conv2d_ht(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), alpha, g),
            nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True),
            a_func,
            conv2d_ht(nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False), alpha, g),
            nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True),
            a_func,
            conv2d_ht(nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False), alpha, g),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet_HT(nn.Module):

    def __init__(self, alpha, g, activation, block, num_block, num_classes=100):
        super().__init__()

        self.alpha = alpha
        self.g = g
        self.activation = activation

        if self.activation=='tanh':
            #a_func=F.tanh
            a_func = nn.Tanh()
        elif self.activation=='relu':
            #a_func=F.relu
            a_func = nn.ReLU(inplace=True)
        else:
            raise NameError("activation does not exist in NetPortal.architectures")     
        
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            #nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            conv2d_ht(nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False), alpha, g),
            nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True)
            a_func)
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.alpha, self.g, self.activation, self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def resnet18_HT(alpha, g, activation):
    """ return a ResNet 18 object
    """
    return ResNet_HT(alpha, g, activation, BasicBlock, [2, 2, 2, 2])

def resnet34_HT(alpha, g, activation):
    """ return a ResNet 34 object
    """
    return ResNet_HT(alpha, g, activation, BasicBlock, [3, 4, 6, 3])

def resnet50_HT(alpha, g, activation):
    """ return a ResNet 50 object
    """
    return ResNet_HT(alpha, g, activation, BottleNeck, [3, 4, 6, 3])

def resnet101_HT(alpha, g, activation):
    """ return a ResNet 101 object
    """
    return ResNet_HT(alpha, g, activation, BottleNeck, [3, 4, 23, 3])

def resnet152_HT(alpha, g, activation):
    """ return a ResNet 152 object
    """
    return ResNet_HT(alpha, g, activation, BottleNeck, [3, 8, 36, 3])
