import math
import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES']='1'

from models.my_models import *
sys.path.append(os.path.split(sys.path[0])[0])
import torch
import torch.nn as nn
import torch.nn.functional as F
import parameter as para
import numpy as np
from torch.nn import init
#--------------------------------------------------------------basic conv--------------------------------------------------------#
class DoubleConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=0):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size, stride, padding),
            nn.PReLU(ch_out),

            nn.Conv3d(ch_out, ch_out, kernel_size, stride, padding),
            nn.PReLU(ch_out),
        )
    def forward(self, x):
        x = self.double_conv(x)
        return x

class TripleConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=0):
        super(TripleConv, self).__init__()
        self.triple_conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size, stride, padding),
            nn.PReLU(ch_out),

            nn.Conv3d(ch_out, ch_out, kernel_size, stride, padding),
            nn.PReLU(ch_out),

            nn.Conv3d(ch_out, ch_out, kernel_size, stride, padding),
            nn.PReLU(ch_out),
        )
    def forward(self, x):
        x = self.triple_conv(x)
        return x

class DownConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=0):
        super(DownConv, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size, stride, padding),
            nn.PReLU(ch_out)
        )
    def forward(self, x):
        x = self.down_conv(x)
        return x

class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=0):
        super(UpConv, self).__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose3d(ch_in, ch_out, kernel_size, stride, padding),
            nn.PReLU(ch_out)
        )
    def forward(self, x):
        x = self.up_conv(x)
        return x

class MaxPooling(nn.Module):
    def __init__(self,ch_in=32, ch_out=1, kernel_size=1, stride=1, scale_factor=1):
        super(MaxPooling, self).__init__()
        self.maxpooling = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size, stride),
            nn.Upsample(scale_factor=(math.pow(2, scale_factor-1), math.pow(2, scale_factor), math.pow(2, scale_factor)), mode='trilinear',align_corners=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        x1 = self.maxpooling(x)
        return x1

class Dense_residual_cat_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Dense_residual_cat_block, self).__init__()
        self.conv1 = nn.Conv3d(ch_in, ch_out, kernel_size=1, padding=0)
        self.conv3 = nn.Conv3d(ch_out, ch_out, kernel_size=3, padding=1)
        self.convm = nn.Conv3d(ch_in + ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv3d(ch_out, ch_out, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(ch_out)
        self.bn3 = nn.BatchNorm3d(ch_out)
        self.bn4 = nn.BatchNorm3d(ch_out)
        self.bn5 = nn.BatchNorm3d(ch_out)
        self.bn6 = nn.BatchNorm3d(ch_out)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):#([3, 128+64, 4, 64, 64])
        x_out1 = self.conv1(x)#([3, 128, 4, 64, 64])
        x_out1 = self.bn1(x_out1)
        x_out1 = self.relu(x_out1)

        x_out3 = self.conv3(x_out1)#([3, 128, 4, 64, 64])
        # x_out3 = self.bn3(x_out3)

        x_out4 = self.convm(torch.cat([x_out3,x],dim=1))#([3, 128, 4, 64, 64])
        # x_out4 = self.bn4(x_out4)

        x_out5 = self.conv5(x_out4)#([3, 128, 4, 64, 64])
        # x_out5 = self.bn5(x_out5)

        x_out = x_out1 + x_out3 + x_out5#([3, 128, 4, 64, 64])
        x_out = self.relu(x_out)
        return x_out#([3, 128, 4, 64, 64])
class LCT_new_DR_UNet_new(nn.Module):
#net total parameters: 23522336
    def __init__(self, training):
        super().__init__()

        self.training = training

        # self.encoder_stage1 = DoubleConv(ch_in=1, ch_out=16, kernel_size=3, stride=1, padding=1)
        # self.down_conv1 = DownConv(ch_in=16, ch_out=32, kernel_size=2, stride=2)
        #
        # self.encoder_stage2 = TripleConv(ch_in=32, ch_out=32, kernel_size=3, stride=1, padding=1)
        # self.down_conv2 = DownConv(ch_in=32, ch_out=64, kernel_size=2, stride=2)
        #
        # self.encoder_stage3 = TripleConv(ch_in=64, ch_out=64, kernel_size=3, stride=1, padding=1)
        # self.down_conv3 = DownConv(ch_in=64, ch_out=128, kernel_size=2, stride=2)
        #
        # self.encoder_stage4 = TripleConv(ch_in=128, ch_out=128, kernel_size=3, stride=1, padding=1)
        # self.down_conv4 = DownConv(ch_in=128, ch_out=256, kernel_size=3, stride=1, padding=1)
        #
        # self.decoder_stage1 = TripleConv(ch_in=128, ch_out=256, kernel_size=3, stride=1, padding=1)
        # self.up_conv2 = UpConv(ch_in=256, ch_out=128, kernel_size=2, stride=2)
        #
        # self.decoder_stage2 = TripleConv(ch_in=128+64, ch_out=128, kernel_size=3, stride=1, padding=1)
        # self.up_conv3 = UpConv(ch_in=128, ch_out=64, kernel_size=2, stride=2)
        #
        # self.decoder_stage3 = TripleConv(ch_in=64+32, ch_out=64, kernel_size=3, stride=1, padding=1)
        # self.up_conv4 = UpConv(ch_in=64, ch_out=32, kernel_size=2, stride=2)
        #
        # self.decoder_stage4 = DoubleConv(ch_in=32+16, ch_out=32, kernel_size=3, stride=1, padding=1)

        self.encoder_stage1 = Dense_residual_block(ch_in=1, ch_out=16)
        self.down_conv1 = DownConv(ch_in=16, ch_out=32, kernel_size=2, stride=2)

        self.encoder_stage2 = Dense_residual_block(ch_in=32, ch_out=32)
        self.down_conv2 = DownConv(ch_in=32, ch_out=64, kernel_size=2, stride=2)

        self.encoder_stage3 = Dense_residual_block(ch_in=64, ch_out=64)
        self.down_conv3 = DownConv(ch_in=64, ch_out=128, kernel_size=2, stride=2)

        self.encoder_stage4 = Dense_residual_block(ch_in=128, ch_out=128)
        self.down_conv4 = DownConv(ch_in=128, ch_out=256, kernel_size=3, stride=1, padding=1)

        self.decoder_stage1 = Dense_residual_cat_block(ch_in=128, ch_out=256)
        self.up_conv2 = UpConv(ch_in=256, ch_out=128, kernel_size=2, stride=2)

        self.decoder_stage2 = Dense_residual_cat_block(ch_in=128 + 64 + 64, ch_out=128)
        self.up_conv3 = UpConv(ch_in=128, ch_out=64, kernel_size=2, stride=2)

        self.decoder_stage3 = Dense_residual_cat_block(ch_in=64 + 32 + 32, ch_out=64)
        self.up_conv4 = UpConv(ch_in=64, ch_out=32, kernel_size=2, stride=2)

        self.decoder_stage4 = Dense_residual_cat_block(ch_in=32 + 16 + 16, ch_out=32)

        # 最后大尺度下的映射（256*256），下面的尺度依次递减
        self.map4 = MaxPooling(ch_in=32, ch_out=1, kernel_size=1, stride=1, scale_factor=1)
        # 128*128 尺度下的映射
        self.map3 = MaxPooling(ch_in=64, ch_out=1, kernel_size=1, stride=1, scale_factor=2)
        # 64*64 尺度下的映射
        self.map2 = MaxPooling(ch_in=128, ch_out=1, kernel_size=1, stride=1, scale_factor=3)
        # 32*32 尺度下的映射
        self.map1 = MaxPooling(ch_in=256, ch_out=1, kernel_size=1, stride=1, scale_factor=4)
#----------------------------------------------------attention----------------------------------------------------------------------#
        self.LCT0 = LCTAttention(dim=16, kernel_size=3)
        self.LCT1 = LCTAttention(dim=32, kernel_size=3)
        self.LCT2 = LCTAttention(dim=64, kernel_size=3)
        self.LCT3 = LCTAttention(dim=128, kernel_size=3)
        self.LCT4 = LCTAttention(dim=256, kernel_size=3)

        self.eca = ECAAttention(kernel_size=3)
# ----------------------------------------------------attention----------------------------------------------------------------------#

    def forward(self, inputs):#inputs([3, 1, 16, 256, 256])
        long_range1 = self.encoder_stage1(inputs)  # ([3, 16, 16, 256, 256])

        short_range1 = self.down_conv1(long_range1)  # ([3, 32, 8, 128, 128])
        long_range2 = self.encoder_stage2(short_range1)  # ([3, 32, 8, 128, 128])
        long_range2 = F.dropout(long_range2, para.drop_rate, self.training)  # ([3, 32, 8, 128, 128])

        short_range2 = self.down_conv2(long_range2)  # ([3, 64, 4, 64, 64])
        long_range3 = self.encoder_stage3(short_range2)  # ([3, 64, 4, 64, 64])
        long_range3 = F.dropout(long_range3, para.drop_rate, self.training)  # ([3, 64, 4, 64, 64])

        short_range3 = self.down_conv3(long_range3)  # ([3, 128, 2, 32, 32])
        long_range4 = self.encoder_stage4(short_range3)  # ([3, 128, 2, 32, 32])
        long_range4 = F.dropout(long_range4, para.drop_rate, self.training)  # ([3, 128, 2, 32, 32])


        # long_range4 = self.cot3(long_range4)
        # long_range4 = self.eca(long_range4)
        outputs = self.decoder_stage1(long_range4)  # ([3, 256, 2, 32, 32])
        outputs = F.dropout(outputs, para.drop_rate, self.training)  # ([3, 256, 2, 32, 32])

        output1 = self.map1(outputs)

        short_range6 = self.up_conv2(outputs)  # ([3, 128, 4, 64, 64])
        # long_range3 = self.eca(long_range3)
        LCT_long_range3 = self.LCT2(long_range3)
        outputs = self.decoder_stage2(torch.cat([short_range6, cot_long_range3,long_range3], dim=1))  # ([3, 128, 4, 64, 64])
        outputs = F.dropout(outputs, 0.3, self.training)  # ([3, 128, 4, 64, 64])

        output2 = self.map2(outputs)

        short_range7 = self.up_conv3(outputs)  # ([3, 64, 8, 128, 128])
        # long_range2 = self.eca(long_range2)
        LCT_long_range2 = self.LCT1(long_range2)
        outputs = self.decoder_stage3(torch.cat([short_range7, cot_long_range2,long_range2], dim=1))  # ([3, 64, 8, 128, 128])
        outputs = F.dropout(outputs, 0.3, self.training)  # ([3, 64, 8, 128, 128])

        output3 = self.map3(outputs)

        short_range8 = self.up_conv4(outputs)  # ([3, 32, 16, 256, 256])
        LCT_long_range1 = self.LCT0(long_range1)
        #long_range1 = self.eca(long_range1)
        outputs = self.decoder_stage4(torch.cat([short_range8, cot_long_range1,long_range1], dim=1))  # ([3, 32, 16, 256, 256])

        output4 = self.map4(outputs)  # ([3, 1, 16, 512, 512])

        # outputs = output1 + output2 + output3 + output4

        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4
        # return output4


def inita(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(module.weight.data, 0.25)
        nn.init.constant_(module.bias.data, 0)


net = LCT_new_DR_UNet_new(training=True)
net.apply(inita)
