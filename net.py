import math

import torch
from torch import nn


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class Net(nn.Module):
    def __init__(self, in_channels=1, out_channels=46):
        super(Net, self).__init__()

        self.conv_layer1 = nn.Conv2d(in_channels, 256, kernel_size=4, stride=2, padding=1, bias=False)

        conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        bn2_2 = nn.BatchNorm2d(256)
        conv_layer2 = [conv2, bn2_2]
        self.conv_layer2 = nn.Sequential(*conv_layer2)
        self.relu2 = nn.ReLU(inplace=True)

        conv3 = nn.Conv2d(256, 512, kernel_size=4, stride=4, padding=0, bias=False)
        bn2_3 = nn.BatchNorm2d(512)
        relu3 = nn.ReLU(inplace=True)
        conv_layer3 = [conv3, bn2_3, relu3]
        self.conv_layer3 = nn.Sequential(*conv_layer3)

        conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        bn2_4 = nn.BatchNorm2d(512)
        conv_layer4 = [conv4, bn2_4]
        self.conv_layer4 = nn.Sequential(*conv_layer4)
        self.relu4 = nn.ReLU(inplace=True)

        conv5 = nn.Conv2d(512, 1024, kernel_size=4, stride=4, padding=0, bias=False)
        bn2_5 = nn.BatchNorm2d(1024)
        relu5 = nn.ReLU(inplace=True)
        conv_layer5 = [conv5, bn2_5, relu5]
        self.conv_layer5 = nn.Sequential(*conv_layer5)

        conv6 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        bn2_6 = nn.BatchNorm2d(1024)
        conv_layer6 = [conv6, bn2_6]
        self.conv_layer6 = nn.Sequential(*conv_layer6)
        self.relu6 = nn.ReLU(inplace=True)

        conv11 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        relu11 = nn.ReLU(inplace=True)
        trans1 = [conv11, relu11]
        self.trans1 = nn.Sequential(*trans1)
        deconv11 = nn.ConvTranspose3d(512, 512, kernel_size=1, stride=1, padding=0, output_padding=0, bias=False)
        relu12 = nn.ReLU(inplace=True)
        trans2 = [deconv11, relu12]
        self.trans2 = nn.Sequential(*trans2)

        deconv7 = nn.ConvTranspose3d(512, 256, kernel_size=4, stride=4, padding=0, output_padding=0, bias=False)
        bn3_7 = nn.BatchNorm3d(256)
        relu16 = nn.ReLU(inplace=True)
        deconv_layer7 = [deconv7, bn3_7, relu16]
        self.deconv_layer7 = nn.Sequential(*deconv_layer7)

        deconv6 = nn.ConvTranspose3d(256, 256, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False)
        bn3_6 = nn.BatchNorm3d(256)
        relu17 = nn.ReLU(inplace=True)
        deconv_layer6 = [deconv6, bn3_6, relu17]
        self.deconv_layer6 = nn.Sequential(*deconv_layer6)

        deconv5 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=4, padding=0, output_padding=0, bias=False)
        bn3_5 = nn.BatchNorm3d(128)
        relu17 = nn.ReLU(inplace=True)
        deconv_layer5 = [deconv5, bn3_5, relu17]
        self.deconv_layer5 = nn.Sequential(*deconv_layer5)

        deconv4 = nn.ConvTranspose3d(128, 128, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False)
        bn3_4 = nn.BatchNorm3d(128)
        relu18 = nn.ReLU(inplace=True)
        deconv_layer4 = [deconv4, bn3_4, relu18]
        self.deconv_layer4 = nn.Sequential(*deconv_layer4)

        deconv3 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        bn3_3 = nn.BatchNorm3d(64)
        relu19 = nn.ReLU(inplace=True)
        deconv_layer3 = [deconv3, bn3_3, relu19]
        self.deconv_layer3 = nn.Sequential(*deconv_layer3)

        deconv2 = nn.ConvTranspose3d(64, 64, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False)
        bn3_2 = nn.BatchNorm3d(64)
        relu20 = nn.ReLU(inplace=True)
        deconv_layer2 = [deconv2, bn3_2, relu20]
        self.deconv_layer2 = nn.Sequential(*deconv_layer2)

        deconv1 = nn.Conv3d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)
        relu21 = nn.ReLU(inplace=True)
        deconv_layer1 = [deconv1, relu21]
        self.deconv_layer1 = nn.Sequential(*deconv_layer1)

        self.output_layer = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        initialize_weights(self)

    def forward(self, x):
        conv1 = self.conv_layer1(x)
        conv2 = self.conv_layer2(conv1)
        relu2 = self.relu2(conv1 + conv2)

        conv3 = self.conv_layer3(relu2)
        conv4 = self.conv_layer4(conv3)
        relu4 = self.relu4(conv3 + conv4)

        conv5 = self.conv_layer5(relu4)
        conv6 = self.conv_layer6(conv5)
        relu6 = self.relu6(conv5 + conv6)

        features = self.trans1(relu6)
        trans_features = features.view(-1, 512, 2, 4, 4)
        trans_features = self.trans2(trans_features)

        deconv7 = self.deconv_layer7(trans_features)
        deconv6 = self.deconv_layer6(deconv7)
        deconv5 = self.deconv_layer5(deconv6)
        deconv4 = self.deconv_layer4(deconv5)
        deconv3 = self.deconv_layer3(deconv4)
        deconv2 = self.deconv_layer2(deconv3)
        deconv1 = self.deconv_layer1(deconv2)

        out = torch.squeeze(deconv1, 1)
        out = self.output_layer(out)
        return out


# 2D Conv
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=1, stride=stride, padding=0,
                     bias=False)


def conv2x2(in_planes, out_planes, stride=2):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=2, stride=stride, padding=0,
                     bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=3, stride=stride, padding=1,
                     bias=False)


def conv4x4(in_planes, out_planes, stride=2):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=4, stride=stride, padding=1,
                     bias=False)


# 3D Conv
def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=1, stride=stride, padding=0,
                     bias=False)


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=3, stride=stride, padding=1,
                     bias=False)


def conv4x4x4(in_planes, out_planes, stride=2):
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=4, stride=stride, padding=1,
                     bias=False)


# 2D Deconv
def deconv1x1(in_planes, out_planes, stride):
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=1, stride=stride, padding=0, output_padding=0,
                              bias=False)


def deconv2x2(in_planes, out_planes, stride):
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=2, stride=stride, padding=0, output_padding=0,
                              bias=False)


def deconv3x3(in_planes, out_planes, stride):
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=3, stride=stride, padding=1, output_padding=0,
                              bias=False)


def deconv4x4(in_planes, out_planes, stride):
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=4, stride=stride, padding=1, output_padding=0,
                              bias=False)


# 3D Deconv
def deconv1x1x1(in_planes, out_planes, stride):
    return nn.ConvTranspose3d(in_planes, out_planes,
                              kernel_size=1, stride=stride, padding=0, output_padding=0,
                              bias=False)


def deconv3x3x3(in_planes, out_planes, stride):
    return nn.ConvTranspose3d(in_planes, out_planes,
                              kernel_size=3, stride=stride, padding=1, output_padding=0,
                              bias=False)


def deconv4x4x4(in_planes, out_planes, stride):
    return nn.ConvTranspose3d(in_planes, out_planes,
                              kernel_size=4, stride=stride, padding=1, output_padding=0,
                              bias=False)


def _make_layers(in_channels, output_channels, type, batch_norm=None, activation=None):
    layers = []

    if type == 'conv1_s1':
        layers.append(conv1x1(in_channels, output_channels, stride=1))
    elif type == 'conv2_s2':
        layers.append(conv2x2(in_channels, output_channels, stride=2))
    elif type == 'conv3_s1':
        layers.append(conv3x3(in_channels, output_channels, stride=1))
    elif type == 'conv4_s2':
        layers.append(conv4x4(in_channels, output_channels, stride=2))
    elif type == 'deconv1_s1':
        layers.append(deconv1x1(in_channels, output_channels, stride=1))
    elif type == 'deconv2_s2':
        layers.append(deconv2x2(in_channels, output_channels, stride=2))
    elif type == 'deconv3_s1':
        layers.append(deconv3x3(in_channels, output_channels, stride=1))
    elif type == 'deconv4_s2':
        layers.append(deconv4x4(in_channels, output_channels, stride=2))
    elif type == 'conv1x1_s1':
        layers.append(conv1x1x1(in_channels, output_channels, stride=1))
    elif type == 'deconv1x1_s1':
        layers.append(deconv1x1x1(in_channels, output_channels, stride=1))
    elif type == 'deconv3x3_s1':
        layers.append(deconv3x3x3(in_channels, output_channels, stride=1))
    elif type == 'deconv4x4_s2':
        layers.append(deconv4x4x4(in_channels, output_channels, stride=2))
    else:
        raise NotImplementedError('layer type [{}] is not implemented'.format(type))

    if batch_norm == '2d':
        layers.append(nn.BatchNorm2d(output_channels))
    elif batch_norm == '3d':
        layers.append(nn.BatchNorm3d(output_channels))

    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'sigm':
        layers.append(nn.Sigmoid())
    elif activation == 'leakyrelu':
        layers.append(nn.LeakyReLU(0.2, True))
    else:
        if activation is not None:
            raise NotImplementedError('activation function [{}] is not implemented'.format(activation))

    return nn.Sequential(*layers)


class ReconNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=46):
        super(ReconNet, self).__init__()

        ######### representation network - convolution layers
        self.conv_layer1 = _make_layers(in_channels, 256, 'conv4_s2', False)
        self.conv_layer2 = _make_layers(256, 256, 'conv3_s1', '2d')
        self.relu2 = nn.ReLU(inplace=True)
        self.conv_layer3 = _make_layers(256, 512, 'conv4_s2', '2d', 'relu')
        self.conv_layer4 = _make_layers(512, 512, 'conv3_s1', '2d')
        self.relu4 = nn.ReLU(inplace=True)
        self.conv_layer5 = _make_layers(512, 1024, 'conv4_s2', '2d', 'relu')
        self.conv_layer6 = _make_layers(1024, 1024, 'conv3_s1', '2d')
        self.relu6 = nn.ReLU(inplace=True)
        self.conv_layer7 = _make_layers(1024, 2048, 'conv4_s2', '2d', 'relu')
        self.conv_layer8 = _make_layers(2048, 2048, 'conv3_s1', '2d')
        self.relu8 = nn.ReLU(inplace=True)
        self.conv_layer9 = _make_layers(2048, 4096, 'conv4_s2', '2d', 'relu')
        self.conv_layer10 = _make_layers(4096, 4096, 'conv3_s1', '2d')
        self.relu10 = nn.ReLU(inplace=True)

        ######### transform module
        self.trans_layer1 = _make_layers(4096, 4096, 'conv1_s1', False, 'relu')
        self.trans_layer2 = _make_layers(2048, 2048, 'deconv1x1_s1', False, 'relu')

        ######### generation network - deconvolution layers
        self.deconv_layer10 = _make_layers(2048, 1024, 'deconv4x4_s2', '3d', 'relu')
        self.deconv_layer8 = _make_layers(1024, 512, 'deconv4x4_s2', '3d', 'relu')
        self.deconv_layer7 = _make_layers(512, 512, 'deconv3x3_s1', '3d', 'relu')
        self.deconv_layer6 = _make_layers(512, 256, 'deconv4x4_s2', '3d', 'relu')
        self.deconv_layer5 = _make_layers(256, 256, 'deconv3x3_s1', '3d', 'relu')
        self.deconv_layer4 = _make_layers(256, 128, 'deconv4x4_s2', '3d', 'relu')
        self.deconv_layer3 = _make_layers(128, 128, 'deconv3x3_s1', '3d', 'relu')
        self.deconv_layer2 = _make_layers(128, 64, 'deconv4x4_s2', '3d', 'relu')
        self.deconv_layer1 = _make_layers(64, 64, 'deconv3x3_s1', '3d', 'relu')
        self.deconv_layer0 = _make_layers(64, 1, 'conv1x1_s1', False, 'relu')
        self.output_layer = _make_layers(64, out_channels, 'conv1_s1', False)

        # network initialization
        initialize_weights(self)

    def forward(self, x, out_feature=False):
        ### representation network
        conv1 = self.conv_layer1(x)
        conv2 = self.conv_layer2(conv1)
        relu2 = self.relu2(conv1 + conv2)
        conv3 = self.conv_layer3(relu2)
        conv4 = self.conv_layer4(conv3)
        relu4 = self.relu4(conv3 + conv4)
        conv5 = self.conv_layer5(relu4)
        conv6 = self.conv_layer6(conv5)
        relu6 = self.relu6(conv5 + conv6)
        conv7 = self.conv_layer7(relu6)
        conv8 = self.conv_layer8(conv7)
        relu8 = self.relu8(conv7 + conv8)
        conv9 = self.conv_layer9(relu8)
        conv10 = self.conv_layer10(conv9)
        relu10 = self.relu10(conv9 + conv10)

        ### transform module
        features = self.trans_layer1(relu10)
        trans_features = features.view(-1, 2048, 2, 4, 4)
        trans_features = self.trans_layer2(trans_features)

        ### generation network
        deconv10 = self.deconv_layer10(trans_features)
        deconv8 = self.deconv_layer8(deconv10)
        deconv7 = self.deconv_layer7(deconv8)
        deconv6 = self.deconv_layer6(deconv7)
        deconv5 = self.deconv_layer5(deconv6)
        deconv4 = self.deconv_layer4(deconv5)
        deconv3 = self.deconv_layer3(deconv4)
        deconv2 = self.deconv_layer2(deconv3)
        deconv1 = self.deconv_layer1(deconv2)

        ### output
        out = self.deconv_layer0(deconv1)
        out = torch.squeeze(out, 1)
        out = self.output_layer(out)

        if out_feature:
            return out, features, trans_features
        else:
            return out


class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv3d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ConstantPad3d((1, 0, 1, 0, 0, 0), 0),
            nn.Conv3d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img):
        return self.model(img)
