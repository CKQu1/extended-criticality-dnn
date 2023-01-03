import torch

class AlexnetBackbone(torch.nn.Module):
    def __init__(self, model):
        super(AlexnetBackbone, self).__init__()
        self.N = list(model.children())[-1][-1].weight.shape[-1]
        self.pre_features = torch.nn.Sequential(*list(model.children())[:-1])
        self.features = torch.nn.Sequential(*list(model.children())[-1][:-1])
        self.flatten = torch.nn.Flatten()
        
    def forward(self, x):
        x = self.pre_features(x)
        x = self.flatten(x)
        x = self.features(x)
        return x


class SqueezeBackbone(torch.nn.Module):
    def __init__(self, model):
        super(SqueezeBackbone, self).__init__()
        self.N = list(model.children())[-1][-3].weight.shape[-1]
        self.pre_features = torch.nn.Sequential(*list(model.children())[:-1])
        self.features = torch.nn.Sequential(*list(model.children())[-1][:-1])
        self.flatten = torch.nn.Flatten()
        
    def forward(self, x):
        x = self.pre_features(x)
        x = self.features(x)
        x = self.flatten(x)
        return x


class WideResNetBackbone(torch.nn.Module):
    def __init__(self, model, module_idx):
        super(WideResNetBackbone, self).__init__()
        self.N = list(model.children())[-1].weight.shape[-1]
        self.features = torch.nn.Sequential(*list(model.children())[:-module_idx])
        self.flatten = torch.nn.Flatten()
        
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return x

    
class VGGBackbone(torch.nn.Module):
    def __init__(self, model):
        super(VGGBackbone, self).__init__()
        self.N = list(model.children())[-1][3].weight.shape[-1]
        self.pre_features = torch.nn.Sequential(*list(model.children())[:-1])
        self.features = torch.nn.Sequential(*list(model.children())[-1][:-1])
        self.flatten = torch.nn.Flatten()
        
    def forward(self, x):
        x = self.pre_features(x)
        x = self.flatten(x)
        x = self.features(x)
        x = self.flatten(x)
        return x
    
    
class EfficientNetBackbone(torch.nn.Module):
    def __init__(self, model):
        super(EfficientNetBackbone, self).__init__()
        self.N = list(model.children())[-1].weight.shape[-1]
        self.pre_features = torch.nn.Sequential(*list(model.children())[:-1])
        self.flatten = torch.nn.Flatten()
        
    def forward(self, x):
        x = self.pre_features(x)
        x = self.flatten(x)
        return x
