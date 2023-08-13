import math
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='1'

from models.my_models.conv_block import *
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
#--------------------------------------------------------------basic conv--------------------------------------------------------#
#--------------------------------------------------------------attention conv--------------------------------------------------------#
class CoTAttention(nn.Module):
    def __init__(self, dim=512, kernel_size=3, res=False):
        super().__init__()
        self.res = res
        self.dim = dim
        self.kernel_size = kernel_size
        self.key_embed = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=4, bias=True),
            nn.BatchNorm3d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv3d(dim, dim, 1, bias=True),
            nn.BatchNorm3d(dim)
        )
        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv3d(2 * dim, 2 * dim // factor, 1, bias=True),
            nn.BatchNorm3d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv3d(2 * dim // factor, kernel_size * kernel_size * dim, 1)
        )
    def forward(self, x):
        bs, c, h, w, d = x.shape
        k1 = self.key_embed(x)  # bs,c,h,w
        v = self.value_embed(x).view(bs, c, -1)  # bs,c,h,w
        y = torch.cat([k1, x], dim=1)  # bs,2c,h,w
        att = self.attention_embed(y)  # bs,c*k*k,h,w
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w, d)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # bs,c,h*w
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w, d)
        out = k1 + k2
        if self.res:
            out = out + x
        return out
class ECAAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid = nn.Sigmoid()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        y = self.gap(x) #bs,c,1,1,1
        y = y.squeeze(-1).squeeze(-1).permute(0, 2, 1) #bs,1,c
        y = self.conv(y) #bs,1,c
        y = self.sigmoid(y) #bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1) #bs,c,1,1,1
        return x*y.expand_as(x)
#--------------------------------------------------------------attention conv--------------------------------------------------------#

class LCT_DR_UNet_new(nn.Module):
#
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
        self.cot0 = CoTAttention(dim=16, kernel_size=3)
        self.cot1 = CoTAttention(dim=32, kernel_size=3)
        self.cot2 = CoTAttention(dim=64, kernel_size=3)
        self.cot3 = CoTAttention(dim=128, kernel_size=3)
        self.cot4 = CoTAttention(dim=256, kernel_size=3)

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
        cot_long_range3 = self.cot2(long_range3)
        outputs = self.decoder_stage2(torch.cat([short_range6, cot_long_range3,long_range3], dim=1))  # ([3, 128, 4, 64, 64])
        outputs = F.dropout(outputs, 0.3, self.training)  # ([3, 128, 4, 64, 64])

        output2 = self.map2(outputs)

        short_range7 = self.up_conv3(outputs)  # ([3, 64, 8, 128, 128])
        # long_range2 = self.eca(long_range2)
        cot_long_range2 = self.cot1(long_range2)
        outputs = self.decoder_stage3(torch.cat([short_range7, cot_long_range2,long_range2], dim=1))  # ([3, 64, 8, 128, 128])
        outputs = F.dropout(outputs, 0.3, self.training)  # ([3, 64, 8, 128, 128])

        output3 = self.map3(outputs)

        short_range8 = self.up_conv4(outputs)  # ([3, 32, 16, 256, 256])
        cot_long_range1 = self.cot0(long_range1)
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


net = LCT_DR_UNet_new(training=True)
net.apply(inita)

# 计算网络参数
print('net total parameters:', sum(param.numel() for param in net.parameters()))
# from torchsummary import summary

def main():
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = LCT_DR_UNet_new(training=False).cuda()
    x = torch.randn(1, 1, 16, 256, 256).cuda()
    # x = torch.randn(3,1,16,512,512).cuda()
    output = net(x)
    print(output.shape)  # torch.Size([2, 1, 8, 512, 512])
    # summary(net,(1,8,256,256))

# if __name__ == '__main__':
#     main()
if __name__ == '__main__':
    main()