import numpy as np
import torch
import torch.nn as nn
from scipy.stats import levy_stable

# This is a copy from online repositories 
class AlexNet(nn.Module):

    def __init__(self, input_height=32, input_width=32, input_channels=3, ch=64, num_classes=1000, alpha=None, g=None, activation="tanh", fc_init=None, with_bias=False):
        # ch is the scale factor for number of channels
        super(AlexNet, self).__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        # added
        self.alpha = alpha
        self.g = g
        self.activation = activation
        self.fc_init = fc_init
        self.with_bias = with_bias

        # activation function
        if self.activation.lower()=='tanh':
            #a_func=F.tanh
            a_func = nn.Tanh()
        elif self.activation.lower()=='relu':
            #a_func=F.relu
            a_func = nn.ReLU(inplace=True)
        else:
            raise NameError("activation does not exist in NetPortal.architectures")       

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, out_channels=ch, kernel_size=4, stride=2, padding=2, bias=with_bias),
            a_func,
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=5, padding=2, bias=with_bias),
            a_func,
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=with_bias),
            a_func,
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=with_bias),
            a_func,
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=with_bias),
            a_func,
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # for conv2d
        if alpha != None and g != None:
            with torch.no_grad():
                for m in self.features:
                    if isinstance(m, nn.Conv2d):
                        size = m.weight.shape
                        N_eff = int(np.prod(m.kernel_size)) * m.in_channels     # most correct
                        m.weight = nn.Parameter(torch.Tensor(levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=g*(0.5/N_eff)**(1./alpha),
                                                    size=size)))             
        
        self.size = self.get_size()
        a = torch.tensor(self.size).float()
        b = torch.tensor(2).float()
        self.width = int(a) * int(1 + torch.log(a) / torch.log(b))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.size, self.width, bias=with_bias),
            a_func,
            nn.Dropout(),
            nn.Linear(self.width, self.width, bias=with_bias),
            a_func,
            nn.Linear(self.width, num_classes, bias=with_bias),
        )

        # for linear
        if fc_init == "fc_ht":
            if alpha != None and g != None:
                with torch.no_grad():
                    for l in self.classifier:
                        if isinstance(l, nn.Linear):
                            size = l.weight.shape
                            N_eff = (size[0]*size[1])**0.5
                            l.weight = nn.Parameter(torch.Tensor(levy_stable.rvs(alpha, 0, size=size,
                                                        scale=g*(0.5/N_eff)**(1./alpha))))        

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


def alexnet(**kwargs):
    return AlexNet(**kwargs)