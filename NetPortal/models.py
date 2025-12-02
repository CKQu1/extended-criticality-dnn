import inspect
from NetPortal.architectures import Vanilla
from NetPortal.architectures import FullyConnected, AlexNet, AlexNetold, ResNet14, ResNet14_ht, ResNet_ht, CONVNET_1D, CONVNET_simple, van2nobias, van10nobias, van5, van5nobias, van20nobias, van50nobias, van100nobias, van150nobias, van300nobias, van5_og, van32_og, van32, van50, van100, van200, van300
from NetPortal.extra_architectures import resnext50, resnext101, resnext152
from NetPortal.resnet_architectures import resnet18_HT, resnet34_HT, resnet50_HT, resnet101_HT, resnet152_HT

#from NetPortal.architectures import MlpMixer, CONFIGS

def filter_n_eval(func, **kwargs):
    """
    Takes kwargs and passes ONLY the named parameters that are specified in the callable func
    :param func: Callable for which we'll filter the kwargs and then pass them
    :param kwargs:
    :return:
    """
    args = inspect.signature(func)
    right_ones = kwargs.keys() & args.parameters.keys()
    newargs = {key: kwargs[key] for key in right_ones}
    return func(**newargs)


def ModelFactory(**kwargs):
    classes = {'fc':           FullyConnected,
               'van':          Vanilla,
               'van5_og':      van5_og,
               'van32_og':     van32_og,
               'convnet_1d':   CONVNET_1D,
               'convnet_old':  CONVNET_simple,
               'alexnetold':   AlexNetold,    # can add more architectures in the future
               'alexnet':      AlexNet,            
               'resnet14':     ResNet14,
               'resnet14_ht':  ResNet14_ht,
               'resnext50':    resnext50,
               'resnext101':    resnext101,
               'resnext152':    resnext152,
               'resnet18_HT':  resnet18_HT,    # cifar100 versions
               'resnet34_HT':  resnet34_HT,
               'resnet50_HT':  resnet50_HT,
               'resnet101_HT': resnet101_HT,
               'resnet152_HT':  resnet152_HT 
               }
    #model = filter_n_eval(classes[kwargs["architecture"].lower()], **kwargs)
    model = filter_n_eval(classes[kwargs["architecture"]], **kwargs)
    return model
