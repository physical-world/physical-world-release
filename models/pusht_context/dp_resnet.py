"""Torchvision ResNet backbone with custom GroupNorm and spatial pooling."""
import torch
from torch import nn
from torchvision import models
from torchvision.models.resnet import BasicBlock


def _replace_bn(module, num_groups):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            groups = min(num_groups, child.num_features)
            while child.num_features % groups:
                groups -= 1
            setattr(module, name, nn.GroupNorm(groups, child.num_features))
        else:
            _replace_bn(child, num_groups)


class DiffusionPolicyResNet(nn.Module):
    def __init__(self, name='resnet18', input_shape=(3,96,96), num_kp=64,
                 gn_num_groups=32, temperature=1.0, learnable_temperature=False, weights=None):
        super().__init__()
        backbone = getattr(models, name)(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        for part in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            _replace_bn(part, gn_num_groups)
        with torch.no_grad():
            feature_shape = self._forward_backbone(torch.zeros(1, *input_shape)).shape[1:]
        self.spatial_softmax = SpatialSoftmax(feature_shape, num_kp, temperature, learnable_temperature)
        self._output_dim = self.spatial_softmax.output_dim

    def _forward_backbone(self, x):
        for layer in (self.stem,self.layer1,self.layer2,self.layer3,self.layer4):
            x = layer(x)
        return x

    def forward(self, x):
        return self.spatial_softmax(self._forward_backbone(x))

    @property
    def output_dim(self):
        return self._output_dim


class SpatialSoftmax(nn.Module):
    def __init__(self,input_shape,num_kp=None,temperature=1.0,learnable_temperature=False):
        super().__init__()
        channels,height,width = input_shape
        self.num_kp = channels if num_kp is None else num_kp
        self.channel_proj = nn.Identity() if num_kp is None else nn.Conv2d(channels,self.num_kp,1)
        value = torch.tensor(temperature)
        if learnable_temperature:
            self.temperature = nn.Parameter(value.reshape(1))
        else:
            self.register_buffer('temperature',value)
        y,x = torch.meshgrid(torch.linspace(-1,1,height),torch.linspace(-1,1,width),indexing='ij')
        self.register_buffer('grid_x',x.reshape(1,1,-1))
        self.register_buffer('grid_y',y.reshape(1,1,-1))

    def forward(self,x):
        scores = self.channel_proj(x).flatten(2)
        probability = torch.softmax(scores/self.temperature.clone(),dim=-1)
        x_coord = (probability*self.grid_x.clone()).sum(-1)
        y_coord = (probability*self.grid_y.clone()).sum(-1)
        return torch.stack((x_coord,y_coord),dim=-1).flatten(1)

    @property
    def output_dim(self):
        return 2*self.num_kp
