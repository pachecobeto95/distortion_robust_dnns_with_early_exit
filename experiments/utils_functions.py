import torch.nn as nn
import torch

class ConvBasic(nn.Module):
    def __init__(self, nIn, nOut, kernel=3, stride=1,
                 padding=1):
        super(ConvBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(nOut),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class ExitBlock(nn.Module):
  """
  This class defines the Early Exit, which allows to finish the inference at the middle layers when
  the classification confidence achieves a predefined threshold
  """
  def __init__(self, n_classes, input_shape, exit_type, device):
    super(ExitBlock, self).__init__()
    _, channel, width, height = input_shape
    """
    This creates a random input sample whose goal is to find out the input shape after each layer.
    In fact, this finds out the input shape that arrives in the early exits, to build a suitable branch.

    Arguments are

    nIn:          (int)    input channel of the data that arrives into the given branch.
    n_classes:    (int)    number of the classes
    input_shape:  (tuple)  input shape that arrives into the given branch
    exit_type:    (str)    this argument define the exit type: exit with conv layer or not, just fc layer
    dataset_name: (str)   defines tha dataset used to train and evaluate the branchyNet
     """

    self.expansion = 1
    self.device = device
    self.layers = nn.ModuleList()

    # creates a random input sample to find out input shape in order to define the model architecture.
    x = torch.rand(1, channel, width, height).to(device)
    
    self.conv = nn.Sequential(
        ConvBasic(channel, channel, kernel=3, stride=2, padding=1),
        nn.AvgPool2d(2),)
    
    #gives the opportunity to add conv layers in the branch, or only fully-connected layers
    if (exit_type == "conv"):
      self.layers.append(self.conv)
    else:
      self.layers.append(nn.AdaptiveAvgPool2d(2))
      
    feature_shape = nn.Sequential(*self.layers).to(device)(x).shape
    
    total_neurons = feature_shape[1]*feature_shape[2]*feature_shape[3] # computes the input neurons of the fc layer 
    self.layers = self.layers.to(device)
    self.classifier = nn.Linear(total_neurons , n_classes).to(device) # finally creates 
    
  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    x = x.view(x.size(0), -1)

    return self.classifier(x)
