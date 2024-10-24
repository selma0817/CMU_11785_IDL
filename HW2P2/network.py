# network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
import torch.nn.init as init

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale=1e-6):
        super(ConvNeXtBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),  # depthwise conv
            nn.GroupNorm(32, dim),  # GroupNorm
            nn.Conv2d(dim, 4 * dim, kernel_size=1),  # pointwise expansion
            nn.GELU(),
            nn.Conv2d(4 * dim, dim, kernel_size=1),  # pointwise reduction
        )
        self.layer_scale = nn.Parameter(layer_scale * torch.ones(dim, 1, 1), requires_grad=True) if layer_scale > 0 else None
        self.stochastic_depth = StochasticDepth(drop_path, mode='row') if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.block(x)
        if self.layer_scale is not None:
            x = self.layer_scale * x
        return input + self.stochastic_depth(x)


class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0.0, layer_scale=1e-6):
        super(ConvNeXt, self).__init__()

        # Stem layer
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(32, dims[0]),  # GroupNorm
        )

        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        for i in range(4):
            # Create stages with ConvNeXt blocks
            stage_blocks = [ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale=layer_scale) for j in range(depths[i])]
            self.stages.append(nn.Sequential(*stage_blocks))
            cur += depths[i]

            # Downsampling layer between stages
            if i < 3:  # No downsample after the last stage
                downsample_layer = nn.Sequential(
                    nn.GroupNorm(32, dims[i]),  # GroupNorm
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),  # Downsampling
                )
                self.downsample_layers.append(downsample_layer)

        self.norm = nn.GroupNorm(32, dims[-1])  # Final GroupNorm

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Linear(dims[-1], num_classes)

    def forward_features(self, x):
        x = self.stem(x)

        for i in range(4):
            x = self.stages[i](x)
            if i < 3:
                x = self.downsample_layers[i](x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        feats = self.forward_features(x)
        out = self.cls_head(feats)
        
        return {'out': out, 'feats': feats}




class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)  # ReLU after skip connection
    def forward(self, x):
        identity = x
        x = self.block(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity  # Skip connection
        x = self.relu(x)
        return x


class ResNet34(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet34, self).__init__()
        self.in_channels = 64

        # convolutional layer
        self.convs = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ResNet layers
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward_features(self, x):
        x = self.convs(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)  # Return features
    
    
    def forward(self, x):
        feats = self.forward_features(x) 
        out = self.fc(feats)

        return {'feats': feats,
                'out': out}



# Implement Resnet-50
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        width = out_channels // 4  

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.Conv2d(width, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        x = self.block(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)
        return x
    
class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        self.in_channels = 64

        self.convs = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.layer1 = self._make_layer(256, 3)
        self.layer2 = self._make_layer(512, 4, stride=2)
        self.layer3 = self._make_layer(1024, 6, stride=2)
        self.layer4 = self._make_layer(2048, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(BottleneckBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BottleneckBlock(out_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward_features(self, x):
        x = self.convs(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        feats = self.forward_features(x) 
        out = self.fc(feats)  

        return {'feats': feats, 'out': out}
