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

        # Representation Network

        self.conv_layer1 = nn.Conv2d(in_channels, 256, kernel_size=4, stride=2, padding=1, bias=False)

        conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        bn2_2 = nn.BatchNorm2d(256)
        conv_layer2 = [conv2, bn2_2]
        self.conv_layer2 = nn.Sequential(*conv_layer2)
        self.relu2 = nn.ReLU(inplace=True)

        conv3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        bn2_3 = nn.BatchNorm2d(512)
        relu3 = nn.ReLU(inplace=True)
        conv_layer3 = [conv3, bn2_3, relu3]
        self.conv_layer3 = nn.Sequential(*conv_layer3)

        conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        bn2_4 = nn.BatchNorm2d(512)
        conv_layer4 = [conv4, bn2_4]
        self.conv_layer4 = nn.Sequential(*conv_layer4)
        self.relu4 = nn.ReLU(inplace=True)

        conv5 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False)
        bn2_5 = nn.BatchNorm2d(1024)
        relu5 = nn.ReLU(inplace=True)
        conv_layer5 = [conv5, bn2_5, relu5]
        self.conv_layer5 = nn.Sequential(*conv_layer5)

        conv6 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        bn2_6 = nn.BatchNorm2d(1024)
        conv_layer6 = [conv6, bn2_6]
        self.conv_layer6 = nn.Sequential(*conv_layer6)
        self.relu6 = nn.ReLU(inplace=True)

        conv7 = nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1, bias=False)
        bn2_7 = nn.BatchNorm2d(2048)
        relu7 = nn.ReLU(inplace=True)
        conv_layer7 = [conv7, bn2_7, relu7]
        self.conv_layer7 = nn.Sequential(*conv_layer7)

        conv8 = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1, bias=False)
        bn2_8 = nn.BatchNorm2d(2048)
        conv_layer8 = [conv8, bn2_8]
        self.conv_layer8 = nn.Sequential(*conv_layer8)
        self.relu8 = nn.ReLU(inplace=True)

        conv9 = nn.Conv2d(2048, 4096, kernel_size=4, stride=2, padding=1, bias=False)
        bn2_9 = nn.BatchNorm2d(4096)
        relu9 = nn.ReLU(inplace=True)
        conv_layer9 = [conv9, bn2_9, relu9]
        self.conv_layer9 = nn.Sequential(*conv_layer9)

        conv10 = nn.Conv2d(4096, 4096, kernel_size=3, stride=1, padding=1, bias=False)
        bn2_10 = nn.BatchNorm2d(4096)
        conv_layer10 = [conv10, bn2_10]
        self.conv_layer10 = nn.Sequential(*conv_layer10)
        self.relu10 = nn.ReLU(inplace=True)

        # Transform Module

        conv11 = nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0, bias=False)
        relu11 = nn.ReLU(inplace=True)
        trans1 = [conv11, relu11]
        self.trans1 = nn.Sequential(*trans1)
        deconv11 = nn.ConvTranspose3d(2048, 2048, kernel_size=1, stride=1, padding=0, output_padding=0, bias=False)
        relu12 = nn.ReLU(inplace=True)
        trans2 = [deconv11, relu12]
        self.trans2 = nn.Sequential(*trans2)

        # Generation Network

        deconv10 = nn.ConvTranspose3d(2048, 1024, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        bn3_10 = nn.BatchNorm3d(1024)
        relu13 = nn.ReLU(inplace=True)
        deconv_layer10 = [deconv10, bn3_10, relu13]
        self.deconv_layer10 = nn.Sequential(*deconv_layer10)

        deconv9 = nn.ConvTranspose3d(1024, 512, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        bn3_9 = nn.BatchNorm3d(512)
        relu14 = nn.ReLU(inplace=True)
        deconv_layer9 = [deconv9, bn3_9, relu14]
        self.deconv_layer9 = nn.Sequential(*deconv_layer9)

        deconv8 = nn.ConvTranspose3d(512, 512, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False)
        bn3_8 = nn.BatchNorm3d(512)
        relu15 = nn.ReLU(inplace=True)
        deconv_layer8 = [deconv8, bn3_8, relu15]
        self.deconv_layer8 = nn.Sequential(*deconv_layer8)

        deconv7 = nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        bn3_7 = nn.BatchNorm3d(256)
        relu16 = nn.ReLU(inplace=True)
        deconv_layer7 = [deconv7, bn3_7, relu16]
        self.deconv_layer7 = nn.Sequential(*deconv_layer7)

        deconv6 = nn.ConvTranspose3d(256, 256, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False)
        bn3_6 = nn.BatchNorm3d(256)
        relu17 = nn.ReLU(inplace=True)
        deconv_layer6 = [deconv6, bn3_6, relu17]
        self.deconv_layer6 = nn.Sequential(*deconv_layer6)

        deconv5 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
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
        print("x=", x.shape)
        conv1 = self.conv_layer1(x)
        print("conv1=", conv1.shape)
        conv2 = self.conv_layer2(conv1)
        print("conv2=", conv2.shape)
        relu2 = self.relu2(conv1 + conv2)
        print("relu2=", relu2.shape)

        conv3 = self.conv_layer3(relu2)
        print("conv3=", conv3.shape)
        conv4 = self.conv_layer4(conv3)
        print("conv4=", conv4.shape)
        relu4 = self.relu4(conv3 + conv4)

        conv5 = self.conv_layer5(relu4)
        print("conv5=", conv5.shape)
        conv6 = self.conv_layer6(conv5)
        print("conv6=", conv6.shape)
        relu6 = self.relu6(conv5 + conv6)

        conv7 = self.conv_layer7(relu6)
        print("conv7=", conv7.shape)
        conv8 = self.conv_layer8(conv7)
        print("conv8=", conv8.shape)
        relu8 = self.relu8(conv7 + conv8)

        conv9 = self.conv_layer9(relu8)
        print("conv9=", conv9.shape)
        conv10 = self.conv_layer10(conv9)
        print("conv10=", conv10.shape)
        relu10 = self.relu10(conv9 + conv10)

        features = self.trans1(relu10)
        print("features=", features.shape)
        trans_features = features.view(-1, 2048, 2, 4, 4)
        print("trans_features=", trans_features.shape)
        trans_features = self.trans2(trans_features)
        print("trans_features=", trans_features.shape)

        deconv10 = self.deconv_layer10(trans_features)
        print("deconv10=", deconv10.shape)
        deconv9 = self.deconv_layer9(deconv10)
        print("deconv9=", deconv9.shape)
        deconv8 = self.deconv_layer8(deconv9)
        print("deconv8=", deconv8.shape)
        deconv7 = self.deconv_layer7(deconv8)
        print("deconv7=", deconv7.shape)
        deconv6 = self.deconv_layer6(deconv7)
        print("deconv6=", deconv6.shape)
        deconv5 = self.deconv_layer5(deconv6)
        print("deconv5=", deconv5.shape)
        deconv4 = self.deconv_layer4(deconv5)
        print("deconv4=", deconv4.shape)
        deconv3 = self.deconv_layer3(deconv4)
        print("deconv3=", deconv3.shape)
        deconv2 = self.deconv_layer2(deconv3)
        print("deconv2=", deconv2.shape)
        deconv1 = self.deconv_layer1(deconv2)
        print("deconv1=", deconv1.shape)

        out = torch.squeeze(deconv1, 1)
        print("out=", out.shape)
        out = self.output_layer(out)
        print("out=", out.shape)
        return out

