import inspect
from NetPortal.architectures import FullyConnected, AlexNet, AlexNetold

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
               'alexnetold':   AlexNetold,  
               'alexnet':      AlexNet           
               }
    #model = filter_n_eval(classes[kwargs["architecture"].lower()], **kwargs)
    model = filter_n_eval(classes[kwargs["architecture"]], **kwargs)
    return model
