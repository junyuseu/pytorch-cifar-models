"""
resneXt for cifar with pytorch

Reference:
[1] S. Xie, G. Ross, P. Dollar, Z. Tu and K. He Aggregated residual transformations for deep neural networks. In CVPR, 2017
"""

import torch
import torch.nn as nn
import math


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cardinality, baseWidth, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        D = int(planes * (baseWidth / 64.))
        C = cardinality
        self.conv1 = nn.Conv2d(inplanes, D*C, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D*C)
        self.conv2 = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D*C)
        self.conv3 = nn.Conv2d(D*C, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        if residual.size() != out.size():
            print(out.size(), residual.size())
        out += residual
        out = self.relu(out)

        return out


class ResNeXt_Cifar(nn.Module):

    def __init__(self, block, layers, cardinality, baseWidth, num_classes=10):
        super(ResNeXt_Cifar, self).__init__()
        self.inplanes = 64
        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, self.baseWidth, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality, self.baseWidth))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resneXt_cifar(depth, cardinality, baseWidth, **kwargs):
    assert (depth - 2) % 9 == 0
    n = (depth - 2) / 9
    model = ResNeXt_Cifar(Bottleneck, [n, n, n], cardinality, baseWidth, **kwargs)
    return model


if __name__ == '__main__':
    net = resneXt_cifar(29, 16, 64)
    y = net(torch.randn(1, 3, 32, 32))
    print(net)
    print(y.size())
