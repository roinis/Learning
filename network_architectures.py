from collections import OrderedDict
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init as nninit
import torchvision.models as models, torchvision.models.resnet as resnet


_ARCH_REGISTRY = {}


def architecture(name, sample_shape):
    """
    Decorator to register an architecture;

    Use like so:

    >>> @architecture('my_architecture', (3, 32, 32))
    ... class MyNetwork(nn.Module):
    ...     def __init__(self, n_classes):
    ...         # Build network
    ...         pass
    """
    def decorate(fn):
        _ARCH_REGISTRY[name] = (fn, sample_shape)
        return fn
    return decorate


def get_net_and_shape_for_architecture(arch_name):
    """
    Get network building function and expected sample shape:

    For example:
    >>> net_class, shape = get_net_and_shape_for_architecture('my_architecture')
    

    >>> if shape != expected_shape:
    ...     raise Exception('Incorrect shape')
    """
    return _ARCH_REGISTRY[arch_name]


@architecture('general', (3,32,32))
class General(nn.Module):
    def __init__(self, n_classes):
        ''' initialize the network '''

        super(General, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, (5,5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, (5,5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)


    def forward(self, x):
        ''' the forward propagation algorithm '''

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@architecture('general_improved', (3,32,32))
class GeneralImproved(nn.Module):
    def __init__(self, n_classes):
        super(GeneralImproved, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 32, (5, 5))
        self.conv1_1_bn = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(32, 64, (3, 3))
        self.conv2_1_bn = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, (3, 3))
        self.conv2_2_bn = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.drop1 = nn.Dropout()
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1_1_bn(self.conv1_1(x))))

        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = self.pool2(F.relu(self.conv2_2_bn(self.conv2_2(x))))
        x = x.view(-1, 1024)
        x = self.drop1(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x



def robust_binary_crossentropy(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = -pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))

def bugged_cls_bal_bce(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))

def log_cls_bal(pred, tgt):
    return -torch.log(pred + 1.0e-6)

def get_cls_bal_function(name):
    if name == 'bce':
        return robust_binary_crossentropy
    elif name == 'log':
        return log_cls_bal
    elif name == 'bug':
        return bugged_cls_bal_bce