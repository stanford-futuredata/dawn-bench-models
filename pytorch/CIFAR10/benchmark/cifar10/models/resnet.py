import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or inplanes != (planes * self.expansion):
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        H = self.conv1(inputs)
        H = self.bn1(H)
        H = F.relu(H)

        H = self.conv2(H)
        H = self.bn2(H)

        H += self.shortcut(inputs)
        outputs = F.relu(H)

        return outputs


class StochasticBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, survival_rate=1):
        super().__init__()
        self.survival_rate = survival_rate
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.increasing = inplanes != (planes * self.expansion)
        if self.increasing:
            assert ((1. * planes * self.expansion) / inplanes) == 2
        if stride != 1:
            self.shortcut = nn.Sequential(nn.AvgPool2d(stride))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        shortcut = self.shortcut(inputs)
        if self.increasing:
            shortcut = torch.cat([shortcut] + [shortcut.mul(0)], 1)

        if not self.training or torch.rand(1)[0] <= self.survival_rate:
            H = self.conv1(inputs)
            H = self.bn1(H)
            H = F.relu(H)

            H = self.conv2(H)
            H = self.bn2(H)

            if self.training:
                H /= self.survival_rate
            H += shortcut
        else:
            H = shortcut
        outputs = F.relu(H)

        return outputs


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1,
                               bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)

        self.increasing = stride != 1 or inplanes != (planes * self.expansion)
        if self.increasing:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride=stride,
                          bias=False)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        H = self.bn1(inputs)
        H = F.relu(H)
        if self.increasing:
            inputs = H
        H = self.conv1(H)

        H = self.bn2(H)
        H = F.relu(H)
        H = self.conv2(H)

        H += self.shortcut(inputs)
        return H


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        if stride != 1 or inplanes != (planes * self.expansion):
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        H = self.conv1(inputs)
        H = self.bn1(H)
        H = F.relu(H)

        H = self.conv2(H)
        H = self.bn2(H)
        H = F.relu(H)

        H = self.conv3(H)
        H = self.bn3(H)

        H += self.shortcut(inputs)
        outputs = F.relu(H)

        return outputs


class ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, cardinality=32,
                 base_width=4):
        super().__init__()

        width = math.floor(planes * (base_width / 64.0))

        self.conv1 = nn.Conv2d(inplanes, width * cardinality, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * cardinality)

        self.conv2 = nn.Conv2d(width * cardinality, width * cardinality, 3,
                               groups=cardinality, padding=1, stride=stride,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(width * cardinality)

        self.conv3 = nn.Conv2d(width * cardinality, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        if stride != 1 or inplanes != (planes * self.expansion):
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        H = self.conv1(inputs)
        H = self.bn1(H)
        H = F.relu(H)

        H = self.conv2(H)
        H = self.bn2(H)
        H = F.relu(H)

        H = self.conv3(H)
        H = self.bn3(H)

        H += self.shortcut(inputs)
        outputs = F.relu(H)

        return outputs


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, stride=stride,
                               bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)

        self.increasing = stride != 1 or inplanes != (planes * self.expansion)
        if self.increasing:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride=stride,
                          bias=False)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        H = self.bn1(inputs)
        H = F.relu(H)
        if self.increasing:
            inputs = H
        H = self.conv1(H)

        H = self.bn2(H)
        H = F.relu(H)
        H = self.conv2(H)

        H = self.bn3(H)
        H = F.relu(H)
        H = self.conv3(H)

        H += self.shortcut(inputs)
        return H


class ResNet(nn.Module):

    def __init__(self, Block, layers, filters, num_classes=10, inplanes=None):
        self.inplanes = inplanes or filters[0]
        super().__init__()

        self.pre_act = 'Pre' in Block.__name__

        self.conv1 = nn.Conv2d(3, self.inplanes, 3, padding=1, bias=False)
        if not self.pre_act:
            self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.num_sections = len(layers)
        for section_index, (size, planes) in enumerate(zip(layers, filters)):
            section = []
            for layer_index in range(size):
                if section_index != 0 and layer_index == 0:
                    stride = 2
                else:
                    stride = 1
                section.append(Block(self.inplanes, planes, stride=stride))
                self.inplanes = planes * Block.expansion
            section = nn.Sequential(*section)
            setattr(self, f'section_{section_index}', section)

        if self.pre_act:
            self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.fc = nn.Linear(filters[-1] * Block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        H = self.conv1(inputs)

        if not self.pre_act:
            H = self.bn1(H)
            H = F.relu(H)

        for section_index in range(self.num_sections):
            H = getattr(self, f'section_{section_index}')(H)

        if self.pre_act:
            H = self.bn1(H)
            H = F.relu(H)

        H = F.avg_pool2d(H, H.size()[2:])
        H = H.view(H.size(0), -1)
        outputs = self.fc(H)

        return outputs


class StochasticResNet(ResNet):

    def __init__(self, Block, layers, filters, num_classes=10, inplanes=None,
                 min_survival_rate=1.0, decay='linear'):
        super().__init__(Block, layers, filters,
                         num_classes=num_classes,
                         inplanes=inplanes)
        L = sum(layers)
        l = 1
        for section_index in range(self.num_sections):
            section = getattr(self, f'section_{section_index}')
            for name, module in section.named_children():
                if decay == 'linear':
                    survival_rate = 1 - ((l / L) * (1 - min_survival_rate))
                elif decay == 'uniform':
                    survival_rate = min_survival_rate
                else:
                    raise NotImplementedError(
                        f"{decay} decay has not been implemented.")
                module.survival_rate = survival_rate
                l += 1
        assert (l - 1) == L


# From "Deep Residual Learning for Image Recognition"
def ResNet20():
    return ResNet(BasicBlock, layers=[3] * 3, filters=[16, 32, 64])


def ResNet32():
    return ResNet(BasicBlock, layers=[5] * 3, filters=[16, 32, 64])


def ResNet44():
    return ResNet(BasicBlock, layers=[7] * 3, filters=[16, 32, 64])


def ResNet56():
    return ResNet(BasicBlock, layers=[9] * 3, filters=[16, 32, 64])


def ResNet110():
    return ResNet(BasicBlock, layers=[18] * 3, filters=[16, 32, 64])


def ResNet1202():
    return ResNet(BasicBlock, layers=[200] * 3, filters=[16, 32, 64])


# From "Identity Mappings in Deep Residual Networks"
def PreActResNet110():
    return ResNet(PreActBlock, layers=[18] * 3, filters=[16, 32, 64])


def PreActResNet164():
    return ResNet(PreActBottleneck, layers=[18] * 3, filters=[16, 32, 64])


def PreActResNet1001():
    return ResNet(PreActBottleneck, layers=[111] * 3, filters=[16, 32, 64])


# Based on but not in "Identity Mappings in Deep Residual Networks"
def PreActResNet8():
    return ResNet(PreActBlock, layers=[1] * 3, filters=[16, 32, 64])


def PreActResNet14():
    return ResNet(PreActBlock, layers=[2] * 3, filters=[16, 32, 64])


def PreActResNet20():
    return ResNet(PreActBlock, layers=[3] * 3, filters=[16, 32, 64])


def PreActResNet56():
    return ResNet(PreActBlock, layers=[9] * 3, filters=[16, 32, 64])


def PreActResNet164Basic():
    return ResNet(PreActBlock, layers=[27] * 3, filters=[16, 32, 64])


# From "Deep Networks with Stochastic Depth"
def StochasticResNet110():
    return StochasticResNet(StochasticBlock, layers=[18] * 3,
                            filters=[16, 32, 64], min_survival_rate=0.5,
                            decay='linear')


def StochasticResNet1202():
    return StochasticResNet(StochasticBlock, layers=[200] * 3,
                            filters=[16, 32, 64], min_survival_rate=0.5,
                            decay='linear')


# Based on but not in "Deep Networks for Stochastic Depth"
def StochasticResNet56():
    return StochasticResNet(StochasticBlock, layers=[9] * 3,
                            filters=[16, 32, 64], min_survival_rate=0.5,
                            decay='linear')


def StochasticResNet56_08():
    return StochasticResNet(StochasticBlock, layers=[9] * 3,
                            filters=[16, 32, 64], min_survival_rate=0.8,
                            decay='linear')


# From "Wide Residual Networks"
def WRN(n, k):
    assert (n - 4) % 6 == 0
    base_filters = [16, 32, 64]
    filters = [num_filters * k for num_filters in base_filters]
    d = (n - 4) / 2  # l = 2
    return ResNet(PreActBlock, layers=[int(d / 3)] * 3, filters=filters,
                  inplanes=16)


def WRN_40_4():
    return WRN(40, 4)


def WRN_16_8():
    return WRN(16, 8)


def WRN_28_10():
    return WRN(28, 10)


# From "Aggregated Residual Transformations for Deep Neural Networks"
def ResNeXt29(cardinality, base_width):
    Block = partial(ResNeXtBottleneck, cardinality=cardinality,
                    base_width=base_width)
    Block.__name__ = ResNeXtBottleneck.__name__
    Block.expansion = ResNeXtBottleneck.expansion
    return ResNet(Block, layers=[3, 3, 3], filters=[64, 128, 256])


# From kunagliu/pytorch
def ResNet18():
    return ResNet(BasicBlock, layers=[2, 2, 2, 2], filters=[64, 128, 256, 512])


def ResNet34():
    return ResNet(BasicBlock, layers=[3, 4, 6, 3], filters=[64, 128, 256, 512])


def ResNet50():
    return ResNet(Bottleneck, layers=[3, 4, 6, 3], filters=[64, 128, 256, 512])


def ResNet101():
    return ResNet(Bottleneck,
                  layers=[3, 4, 23, 3], filters=[64, 128, 256, 512])


def ResNet152():
    return ResNet(Bottleneck,
                  layers=[3, 8, 36, 3], filters=[64, 128, 256, 512])
