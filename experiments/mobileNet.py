import torch
import torch.nn as nn
from utils_functions import ExitBlock
from pthflops import count_ops
import torchvision.models as models
import numpy as np


class B_MobileNet(nn.Module):
  def __init__(self, n_classes: int, 
               pretrained: bool, n_branches: int, img_dim:int, 
               exit_type: str, device, branches_positions=None, distribution="linear"):
    super(B_MobileNet, self).__init__()

    self.n_classes = n_classes
    self.pretrained = pretrained
    self.n_branches = n_branches
    self.img_dim = img_dim
    self.exit_type = exit_type
    self.branches_positions = branches_positions
    self.distribution = distribution
    self.softmax = nn.Softmax(dim=1)
    self.device = device

    self.model = self.initialize_model()
    self.n_blocks = len(list(self.model.features))
    self.insertBranches()
  
  def initialize_model(self):
    model = models.mobilenet_v2(pretrained=self.pretrained)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.n_classes)
    return model.to(self.device)
  
  def countFlops(self):
    x = torch.rand(1, 3, self.img_dim, self.img_dim).to(self.device)
    flops_count_dict = {}
    flops_acc_dict = {}
    flops_list = []
    total_flops = 0
    for i, layer in enumerate(self.model.features, 1):
      ops, all_data = count_ops(layer, x, print_readable=False, verbose=False)
      x = layer(x)
      flops_count_dict[i] = ops
      total_flops += ops
      flops_acc_dict[i] = total_flops
    
    #for key, value in flops_acc_dict.items():
    #  flops_acc_dict[key] = value/total_flops

    return flops_count_dict, flops_acc_dict, total_flops

  def set_thresholds(self, total_flops):
    """
    """
    gold_rate = 1.61803398875
    flop_margin = 1.0 / (self.n_branches+1)
    self.threshold = []
    self.percentage_threshold = []
        
    for i in range(self.n_branches):
      if (self.distribution == 'pareto'):
        self.threshold.append(total_flops * (1 - (0.8**(i+1))))
        self.percentage_threshold.append(1 - (0.8**(i+1)))
      elif (self.distribution == 'fine'):
        self.threshold.append(total_flops * (1 - (0.95**(i+1))))
        self.percentage_threshold.append(1 - (0.95**(i+1)))
      elif (self.distribution == 'linear'):
        self.threshold.append(total_flops * flop_margin * (i+1))
        self.percentage_threshold.append(flop_margin * (i+1))

      else:
        self.threshold.append(total_flops * (gold_rate**(i - self.num_ee)))
        self.percentage_threshold.append(gold_rate**(i - self.n_branches))
  
  
  def is_suitable_for_exit(self, i, flop_count):
    if (self.branches_positions is None):
      return self.stage_id < self.n_branches and flop_count >= self.threshold[self.stage_id]
    
    else:
      return i in self.branches_positions
  
  def add_early_exit(self, layer):
    #print("Adding")
    self.stages.append(nn.Sequential(*self.layers))
    x = torch.rand(1, 3, self.img_dim, self.img_dim).to(self.device)
    feature_shape = nn.Sequential(*self.stages)(x).shape
    self.exits.append(ExitBlock(self.n_classes, feature_shape, self.exit_type, self.device))
    self.stage_id += 1
    self.layers = nn.ModuleList()

  def insertBranches(self):
    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.stage_id = 0

    flops_count_dict, flops_acc_dict, total_flops = self.countFlops()
    self.set_thresholds(total_flops)

    for i, layer in enumerate(self.model.features, 1):
      if (self.is_suitable_for_exit(i, flops_acc_dict[i])):
        self.add_early_exit(layer)
      else:
        self.layers.append(layer)

    self.stages.append(nn.Sequential(*self.layers))
    self.fully_connected = self.model.classifier


  def forwardTrain(self, x):
    output_list, conf_list, class_list  = [], [], []
    for i, exitBlock in enumerate(self.exits):
      x = self.stages[i](x)
      output_branch = exitBlock(x)
      output_list.append(output_branch)
      conf, infered_class = torch.max(self.softmax(output_branch), 1)
      conf_list.append(conf)
      class_list.append(infered_class)

    x = self.stages[-1](x)
    x = x.mean(3).mean(2)

    output = self.fully_connected(x)
    infered_conf, infered_class = torch.max(self.softmax(output), 1)
    output_list.append(output)
    conf_list.append(infered_conf)
    class_list.append(infered_class)
    return output_list, conf_list, class_list

  def forwardEval(self, x, p_tar):
    output_list, conf_list, class_list  = [], [], []
    for i, exitBlock in enumerate(self.exits):
      x = self.stages[i](x)
      output_branch = exitBlock(x)
      conf, infered_class = torch.max(self.softmax(output_branch), 1)

      if (conf.item() > p_tar):
        return output_branch, conf, infered_class, i

      else:
        output_list.append(output_branch)
        conf_list.append(conf)
        class_list.append(infered_class)

    x = self.stages[-1](x)
    x = x.mean(3).mean(2)

    output = self.fully_connected(x)
    conf, infered_class = torch.max(self.softmax(output), 1)
    conf_list.append(conf)
    class_list.append(infered_class)
    output_list.append(output)
    
    if (conf.item() >= p_tar):
      return output, conf, infered_class, self.n_branches 
    else:
      max_conf = np.argmax(conf_list)
      return output_list[max_conf], conf_list[max_conf], class_list[max_conf], self.n_branches

  def forwardEvalRepo(self, x, p_tar):
    exit_branch_list = np.zeros(self.n_branches+1)
    for i, exitBlock in enumerate(self.exits):
      x = self.stages[i](x)
      output_branch = exitBlock(x)
      conf, infered_class = torch.max(self.softmax(output_branch), 1)


      if (conf.item() >= p_tar):
        exit_branch_list[i] = 1

    x = self.stages[-1](x)
    x = x.mean(3).mean(2)

    output = self.fully_connected(x)
    exit_branch_list[-1] = 1
    return exit_branch_list

  def forward(self, x, p_tar=0.5, train=True, repo=False):
    if (train):
      return self.forwardTrain(x)
    else:
      if (repo):
        return self.forwardEvalRepo(x, p_tar)
      else:
        return self.forwardEval(x, p_tar)
