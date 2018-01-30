import sys
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as helper
from torch import nn
import numpy as np

#This implementation will be basic CapsNet, and class size will be modular

def softmax(input):
    size = input.size()
    transposeVal = input.transpose(1, size - 1)
    softmaxVal = helper.softmax(transposeVal.contiguous().view(-1, transposeVal.size(-1)))
    return softmaxVal.view(*transposeVal.size()).transpose(dim, len(input.size()) - 1)

#this class is responsible for constructing all capsule layers, either primary caps or digit caps
class CapsLayer(nn.Module):

    def __init__(self, capsule_num, route_nodes_num, input_channels, output_channels, kernel_size=None,
                 stride=None, iterations=3):
        
        super(CapsLayer, self).__init__()

        self.iterations = iterations
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.route_nodes_num = route_nodes_num
        self.capsule_num = capsule_num

        if self.route_nodes_num != -1:
            #this is digit caps, assign weights for that layer

            self.weights = nn.parameter(torch.randn(self.capsule_num, self.route_nodes_num, self.input_channels, self.output_channels))

        else:
            #this is primary caps
            convLayers = []

            for i in range(self.capsule_num):
                convLayers.append(nn.Conv2d(self.input_channels, self.output_channels, kernel_size=kernel_size, stride=stride, padding=0))
            self.capsules = convLayers

    def squash(self, input, dim=-1):
        squared_input = (input ** 2).sum(dim=dim, keepdim=True)

        normalized = squared_input / (squared_input + 1)

        return normalized
