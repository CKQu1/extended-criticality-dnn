import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F

from scipy.stats import levy_stable
from torch import nn

class FullyConnected(nn.Module):
    def __init__(self, dims, alpha, g, init_path, init_epoch, with_bias, 
                 is_weight_share=False, init_type='ht', activation='tanh', **pretrained):
        super(FullyConnected, self).__init__()
        self.activation = activation
        self.input_dim = dims[0]
        self.dims = dims
        self.depth = len(dims) - 1
        self.num_classes = dims[-1]
        self.alpha, self.g = alpha, g
        self.init_type = init_type
        self.with_bias = with_bias

        #self.init_supported = ['mfrac_orthogonal', 'mfrac', 'mfrac_sym', 'ht']
        self.init_supported = ['ht']
        assert init_type in self.init_supported, "Initialization type does not exist!"
        
        modules = []
        for idx in range(len(dims) - 2):
            modules.append(nn.Linear(dims[idx], dims[idx + 1], bias=with_bias))
            if activation == 'relu':
                modules.append(nn.ReLU())
            elif activation == 'tanh':
                modules.append(nn.Tanh())
            else:
                raise NameError("activation does not exist in NetPortal.architectures")
        modules.append(nn.Linear(dims[-2], dims[-1], bias=with_bias))
    
        # train from pretrained networks
        if init_path != None and init_epoch != None:
            print(f"Loading pretrained weights from {init_path} at epoch {init_epoch}!")
            with torch.no_grad():
                widx = 0
                if 'fcn_grid' in init_path:     # (by main_last_epoch_2.py)
                    print('fcn_grid')
                    w_all = sio.loadmat(f"{init_path}/model_{init_epoch}_sub_loss_w.mat")                
                    for l in modules:
                        if not isinstance(l, nn.Linear): continue
                        l.weight = nn.Parameter(torch.Tensor( w_all['sub_weights'][0][widx].reshape(l.weight.shape) ))

                        widx += 1
                else:                   # (by train_supervised.py)
                    w_all = np.load(f"{init_path}/epoch_{init_epoch}/weights.npy")
                    start = 0
                    for l in modules:
                        if not isinstance(l, nn.Linear): continue
                        w_delta = dims[widx] * dims[widx+1]
                        l.weight = nn.Parameter(torch.Tensor( w_all[start:start+w_delta].reshape(l.weight.shape) ))                                    
            
                        start += dims[widx] * dims[widx+1]
                        widx += 1   

        else:
            if alpha != None and g != None:
                print("Heavy-tailed initialization applied!")                
                with torch.no_grad():
                    if is_weight_share:
                        if init_type == 'ht':
                            is_first = True
                            for l in modules:
                                if not isinstance(l, nn.Linear): continue
                                 
                                if is_first:
                                    size = l.weight.shape
                                    N_eff = (size[0]*size[1])**0.5
                                    shared_weights = torch.Tensor(levy_stable.rvs(alpha, 0, size=size,
                                                       scale=g*(0.5/N_eff)**(1./alpha)))
                                    l.weight = nn.Parameter(shared_weights.clone().detach())
                                    is_first = False
                                else:
                                    size_cur = l.weight.shape
                                    N_eff = (size_cur[0]*size_cur[1])**0.5                                    
                                    if size == size_cur:
                                        l.weight = nn.Parameter(shared_weights.clone().detach())
                                    else:
                                        shared_weights = torch.Tensor(levy_stable.rvs(alpha, 0, size=size_cur,
                                                           scale=g*(0.5/N_eff)**(1./alpha)))
                                        l.weight = nn.Parameter(shared_weights.clone().detach())                                        

                                    size = size_cur                                    

                    else:
                        # set seed for weight matrices
                        l_seeds = torch.randint(0,1000000,(self.depth,))
                        lidx = 0
                        for l in modules:
                            if not isinstance(l, nn.Linear): continue
                            size = l.weight.shape
                            if init_type == 'ht':                            

                                # set seed for layer l
                                np.random.seed(seed=l_seeds[lidx].item())

                                N_eff = (size[0]*size[1])**0.5
                                l.weight = nn.Parameter(torch.Tensor(levy_stable.rvs(alpha, 0, size=size,
                                                            scale=g*(0.5/N_eff)**(1./alpha))))
                                lidx += 1

                                # init biases (should be at fixed point?)
                                if with_bias == True:
                                    pass

        self.sequential = nn.Sequential(*modules)             

    """
    def forward(self, x):
        #x = x.view(x.size(0), -1)
        #x = x.view(x.size(0), self.input_dim)
        #print(x.shape)         
        if x.shape[-1] == self.input_dim:
            return self.sequential(x)
        else:
            og_shape = x.shape
            x = x.view(-1, self.input_dim)
            return self.sequential(x).view( list(og_shape)[:-2] + [self.input_dim] )
    """

    def forward(self, x):
        #x = x.view(x.size(0), -1)
        #x = x.view(x.size(0), self.input_dim)
        #print(x.shape)         
        x = x.view(-1, self.input_dim)
        return self.sequential(x)

    def dm(self, x):
        m = nn.ReLU() if self.activation == 'relu' else (nn.Tanh() if self.activation == 'tanh' else None)
        x = torch.autograd.Variable(x, requires_grad=True)
        y = m(x)
        y.backward( torch.ones_like(x) )

        return x.grad

    # preactivation outputs
    def preact_layer(self, x):
        # number of hidden layers
        hidden = [None] * (self.depth - 1)
        x = x.view(x.size(0), -1)
        ell = 2

        for idx in range(self.depth - 1):
            if idx == 0:
                hidden[idx] = self.sequential[idx](x)
            else:
                hidden[idx] = self.sequential[ell * idx - 1: ell * idx + 1](hidden[idx - 1])

        # final vector is the network output
        return hidden + [self.sequential[-2:](hidden[-1])]

    # postactivation outputs
    def postact_layer(self, x):
        # number of hidden layers
        hidden = [None] * (self.depth - 1)
        x = x.view(x.size(0), -1)
        ell = 2

        for idx in range(self.depth - 1):
            if idx == 0:
                hidden[idx] = self.sequential[ell * idx: ell * idx + ell](x)
            else:
                hidden[idx] = self.sequential[ell * idx: ell * idx + ell](hidden[idx - 1])

        return hidden, self.sequential[-1](hidden[-1])
    """
    The problem for this is that the last weight matrix in all these settings are not symmetrical, i.e. for mnist W_L is 784 by 10 (might need to adjust this in the future)
    final argument decides whether it is post-activation or not (pre-activation)
    """
    def layerwise_jacob_ls(self, x, post: bool):    # option to get Jacobian for pre-/post-activation layerwise Jacobian
        # check "The Emergence of Spectral Universality in Deep Networks"
        
        m = nn.ReLU() if self.activation == 'relu' else (nn.Tanh() if self.activation == 'tanh' else None)
        preact_h = self.preact_layer(x)     # do not include the input layer check eq (1) and (2) of "Emergence of Spectral Universality in Deep Networks"     
        # get weights
        weights = [p for p in self.parameters()]        

        if post:
            dphi_h = self.dm(preact_h[0][0])
            DW_l = torch.matmul(torch.diag( dphi_h ), weights[0])
            #DW_l = torch.matmul(torch.diag( m(preact_h[0][0]) ), weights[0])
            DW_ls = [DW_l]

            for i in range(1, len(preact_h)):   # include the last one even if it is not symmetrical
            #for i in range(1, len(preact_h) - 1):
                dphi_h = self.dm(preact_h[i][0])
                if i != len(preact_h) - 1:
                    DW_l = torch.matmul(torch.diag( dphi_h ), weights[i])
                else:
                    # After final linear/affine transformer, there is no non-linear activation.
                    DW_l = torch.matmul(torch.diag( torch.ones_like(dphi_h) ), weights[i])                

                #DW_l = torch.matmul(torch.diag( m(preact_h[i][0]) ), weights[i])
                DW_ls.append(DW_l)

                #print(weights[i].shape)  # delete
        else:
            # essentially the first activation function is just the identity map
            DW_ls = [weights[0]]    # it's actually W^{l + 1} D^l

            for i in range(0, len(preact_h) - 1):
                dphi_h = self.dm(preact_h[i][0])
                DW_l = torch.matmul(weights[i + 1], torch.diag( dphi_h ))
                #print(DW_l.shape)
                DW_ls.append(DW_l)
            
                #print(weights[i].shape)  # delete
        return DW_ls

####################################################################### ALL CNNS #######################################################################


"""
class AlexNet(nn.Module):

    def __init__(self, alpha, g, input_height=32, input_width=32, input_channels=3, ch=64, num_classes=1000):
        # ch is the scale factor for number of channels
        super(AlexNet, self).__init__()
        
        # HT initialization
        self.alpha = alpha
        self.g = g

        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels

        self.features = nn.Sequential(
            nn.Conv2d(3, out_channels=ch, kernel_size=4, stride=2, padding=2, bias=with_bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=5, padding=2, bias=with_bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=with_bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=with_bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=with_bias),
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

"""

class AlexNet(nn.Module):

    def __init__(self, alpha, g, dataset, activation, fc_init, dropout: float=0.5, num_classes=1000, with_bias=False):
        # ch is the scale factor for number of channels
        super(AlexNet, self).__init__()
        
        # HT initialization
        self.alpha = alpha
        self.g = g
        self.dataset = dataset
        self.activation = activation
        self.fc_init = fc_init

        #if dataset == "mnist":
        #    N_final = 1024
        #elif dataset == "cifar10":
        #    N_final = 4096
        N_final = 1024

        if self.activation=='tanh':
            #a_func=F.tanh
            a_func = nn.Tanh()
        elif self.activation=='relu':
            #a_func=F.relu
            a_func = nn.ReLU(inplace=True)
        else:
            raise NameError("activation does not exist in NetPortal.architectures")     

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2), bias=with_bias),
            a_func,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=with_bias),
            a_func,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=with_bias),
            a_func,
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=with_bias),
            a_func,
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=with_bias),
            a_func,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            #nn.MaxPool2d(kernel_size=3, stride=2),
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

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096,bias=with_bias),
            a_func,
            nn.Dropout(p=dropout),
            nn.Linear(4096, N_final, bias=with_bias),
            a_func,
            nn.Linear(N_final, num_classes, bias=with_bias),
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
        
        # orthogonal matrix
        elif fc_init == "fc_orthogonal":
            with torch.no_grad():
                for l in self.classifier:
                    if isinstance(l, nn.Linear):
                        nn.init.orthogonal_(l.weight, gain=1)

        # otherwise it is the default (uniform) option
        
    def get_size(self):
        # hack to get the size for the FC layer...
        x = torch.randn(1, self.input_channels, self.input_height, self.input_width)
        y = self.features(x)
        print(y.size())
        return y.view(-1).size(0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNetold(nn.Module):

    def __init__(self, alpha, g, activation, fc_init, input_height=32, input_width=32, input_channels=3, ch=64, num_classes=1000, with_bias=False):
        # ch is the scale factor for number of channels
        super(AlexNetold, self).__init__()
        
        # HT initialization
        self.alpha = alpha
        self.g = g
        self.activation = activation
        self.fc_init = fc_init

        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels

        if self.activation=='tanh':
            #a_func=F.tanh
            a_func = nn.Tanh()
        elif self.activation=='relu':
            #a_func=F.relu
            a_func = nn.ReLU(inplace=True)
        else:
            raise NameError("activation does not exist in NetPortal.architectures")     

        self.features = nn.Sequential(
            nn.Conv2d(3, out_channels=ch, kernel_size=4, stride=2, padding=2, bias=with_bias),
            #nn.ReLU(inplace=True),
            a_func,
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=5, padding=2, bias=with_bias),
            #nn.ReLU(inplace=True),
            a_func,
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=with_bias),
            #nn.ReLU(inplace=True),
            a_func,
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=with_bias),
            #nn.ReLU(inplace=True),
            a_func,
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=with_bias),
            #nn.ReLU(inplace=True),
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
        print(self.size)
        a = torch.tensor(self.size).float()
        b = torch.tensor(2).float()
        self.width = int(a) * int(1 + torch.log(a) / torch.log(b))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.size, self.width, bias=with_bias),
            #nn.ReLU(inplace=True),
            a_func,
            nn.Dropout(),
            nn.Linear(self.width, self.width, bias=with_bias),
            #nn.ReLU(inplace=True),
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
        
            
        elif fc_init == "fc_orthogonal":
            with torch.no_grad():
                for l in self.classifier:
                    if isinstance(l, nn.Linear):
                        nn.init.orthogonal_(l.weight, gain=1)
        
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


############### ResNet-HT ###############

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


class BasicBlock_ht(nn.Module):
    expansion = 1

    def __init__(self, alpha, g, activation, in_planes, planes, stride=1):
        super(BasicBlock_ht, self).__init__()
        self.alpha = alpha
        self.g = g
        self.activation = activation
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # heavy-tailed initialization  
        self.conv1 = conv2d_ht(self.conv1, alpha, g)
        self.bn1   = nn.BatchNorm2d(planes)
        # should something be done for the batchn normalization layer?
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # heavy-tailed initialization  
        self.conv2 = conv2d_ht(self.conv2, alpha, g)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            conv_shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            conv_shortcut = conv2d_ht(conv_shortcut, alpha, g)
            self.shortcut = nn.Sequential(
                conv_shortcut,
                nn.BatchNorm2d(self.expansion*planes)
            )         

    def forward(self, x):
        if self.activation=='tanh':
            a_func=F.tanh
        elif self.activation=='relu':
            a_func=F.relu
        else:
            raise NameError("activation does not exist in NetPortal.architectures")     

        #out = F.relu(self.bn1(self.conv1(x)))
        out = a_func(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        #out = F.relu(out)
        out = a_func(out)
        return out

    def get_hidden_outputs(self, x):
        if self.activation=='tanh':
            a_func=F.tanh
        elif self.activation=='relu':
            a_func=F.relu
        else:
            raise NameError("activation does not exist in NetPortal.architectures")     

        hidden_ls = []
        out = self.conv1(x)
        hidden_ls.append(out)
        out = self.bn1(out)
        hidden_ls.append(out)
        out = a_func(out)
        hidden_ls.append(out)
        out = self.bn2(self.conv2(out))
        hidden_ls.append(out)
        out += self.shortcut(x)
        hidden_ls.append(out)
        out = a_func(out)
        hidden_ls.append(out)
        return hidden_ls

class ResNet_ht(nn.Module):
    def __init__(self, alpha, g, activation, block, num_blocks, num_classes=10):
        super(ResNet_ht, self).__init__()
        self.in_planes = 16

        self.alpha = alpha
        self.g = g
        self.activation = activation
        self.conv1  = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = conv2d_ht(self.conv1, alpha, g)
        self.bn1    = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.alpha, self.g, self.activation, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.activation=='tanh':
            a_func=F.tanh
        elif self.activation=='relu':
            a_func=F.relu
        else:
            raise NameError("activation does not exist in NetPortal.architectures")  

        #out = F.relu(self.bn1(self.conv1(x)))
        out = a_func(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# CIFAR-10 models
def ResNet14_ht(alpha, g, activation):
    depth = 14
    n = (depth - 2) // 6
    return ResNet_ht(alpha, g, activation, BasicBlock_ht, [n,n,n])