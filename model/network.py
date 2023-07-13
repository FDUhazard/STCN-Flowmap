"""
network.py - The core of the neural network
Defines the structure and memory operations
Modifed from STM: https://github.com/seoungwugoh/STM

The trailing number of a variable usually denote the stride
e.g. f16 -> encoded features with stride 16
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import *


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ResBlock(1024, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256)  # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256)  # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, f16, f8, f4): # f16:[1024，H/16, W/16]
        x = self.compress(f16)  # x:[512，H/16, W/16]
        x = self.up_16_8(f8, x) # x:[256, H/8, W/8]
        x = self.up_8_4(f4, x) # x:[256. H/4, W/4]

        x = self.pred(F.relu(x))  # [1. H/4, W/4]

        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x  # B,1,H,W

# basic modules，NewDecoder中使用
class Conv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ConvRelu(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        self.add_module('relu', nn.ReLU())
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = Conv(c, c, 3, 1, 1)
        self.conv2 = nn.Sequential(ConvRelu(c, c, 1, 1, 0), Conv(c, c, 1, 1, 0))
        self.conv3 = nn.Sequential(ConvRelu(2, 16, 3, 1, 1), Conv(16, 1, 3, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        c = torch.sigmoid(self.conv2(F.adaptive_avg_pool2d(x, output_size=(1, 1))) + self.conv2(F.adaptive_max_pool2d(x, output_size=(1, 1))))
        x = x * c
        s = torch.sigmoid(self.conv3(torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)))
        x = x * s
        return x

#融合光流特征的Decoder
class NewDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvRelu(1024, 256, 1, 1, 0)
        self.blend1 = ConvRelu(256, 256, 3, 1, 1)
        self.cbam1 = CBAM(256)
        self.conv2 = ConvRelu(512, 256, 1, 1, 0)
        self.blend2 = ConvRelu(256 + 256, 256, 3, 1, 1)
        self.cbam2 = CBAM(256)
        self.conv3 = ConvRelu(256, 256, 1, 1, 0)
        self.blend3 = ConvRelu(256 + 256, 256, 3, 1, 1)
        self.cbam3 = CBAM(256)
        self.predictor = Conv(256, 1, 3, 1, 1)

    def forward(self, f16, f8, f4, s16, s8, s4): # f对应RGB特征，s对应光流特征
        x = self.conv1(f16 + s16)
        x = self.cbam1(self.blend1(x))
        t8 = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        x = torch.cat([self.conv2(f8 + s8), t8], dim=1)
        x = self.cbam2(self.blend2(x))
        t4 = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        x = torch.cat([self.conv3(f4 + s4), t4], dim=1)
        x = self.cbam3(self.blend3(x))
        x = self.predictor(x)
        final_score = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)
        return final_score

class MemoryReader(nn.Module):
    def __init__(self):
        super().__init__()

    def get_affinity(self, mk, qk):
        B, CK, T, H, W = mk.shape
        mk = mk.flatten(start_dim=2)
        qk = qk.flatten(start_dim=2)

        # See supplementary material  a_sq为通道维度的平方和，作为偏移量，对点积相似度进行归一化和标准化
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk  # 点积相似度，为何不是平方相似度？？少了个**2？

        affinity = (2 * ab - a_sq) / math.sqrt(CK)  # B, THW, HW  使得相似度范围在[-1/sqrt(CK),1/sqrt(CK)]

        # softmax operation; aligned the evaluation style
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]  # dim=1取出HW列上的最大值 B,1,HW
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)  # B 1 HW
        affinity = x_exp / x_exp_sum

        return affinity

    def readout(self, affinity, mv, qv):
        B, CV, T, H, W = mv.shape

        mo = mv.view(B, CV, T * H * W)
        mem = torch.bmm(mo, affinity)  # Weighted-sum B, CV, HW
        mem = mem.view(B, CV, H, W)

        mem_out = torch.cat([mem, qv], dim=1)  # B CK+CV H W?

        return mem_out


class STCN(nn.Module):
    def __init__(self, single_object):
        super().__init__()
        self.single_object = single_object

        self.key_encoder = KeyEncoder()
        self.flow_encoder = FlowEncoder()

        if single_object:
            self.value_encoder = ValueEncoderSO()
        else:
            self.value_encoder = ValueEncoder()

        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(1024, keydim=64)
        self.flow_proj = FlowProjection(1024,keydim=64)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.flow_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.memory = MemoryReader()
        self.decoder = Decoder()

    def aggregate(self, prob):  # 作用?
        new_prob = torch.cat([
            torch.prod(1 - prob, dim=1, keepdim=True),
            prob
        ], 1).clamp(1e-7, 1 - 1e-7)
        logits = torch.log((new_prob / (1 - new_prob)))
        return logits

    def encode_key(self, frame):
        # input: b*t*c*h*w
        b, t = frame.shape[:2]

        f16, f8, f4 = self.key_encoder(frame.flatten(start_dim=0, end_dim=1))
        k16 = self.key_proj(f16)  # 维度由1024 -- 64
        f16_thin = self.key_comp(f16) # 维度由1024 -- 512

        # B*C*T*H*W
        k16 = k16.view(b, t, *k16.shape[-3:]).transpose(1, 2).contiguous()

        # B*T*C*H*W
        f16_thin = f16_thin.view(b, t, *f16_thin.shape[-3:])  # 语法含义？5维还是3维
        f16 = f16.view(b, t, *f16.shape[-3:])
        f8 = f8.view(b, t, *f8.shape[-3:])
        f4 = f4.view(b, t, *f4.shape[-3:])

        return k16, f16_thin, f16, f8, f4

    def encode_flow(self, frame):
        # input: b*t*c*h*w
        b, t = frame.shape[:2]

        s16, s8, s4 = self.flow_encoder(frame.flatten(start_dim=0, end_dim=1))
        ks16 = self.flow_proj(s16)  # 维度由1024 -- 64
        s16_thin = self.flow_comp(s16) # 维度由1024 -- 512

        # B*C*T*H*W
        s16 = s16.view(b, t, *s16.shape[-3:]).transpose(1, 2).contiguous()

        # B*T*C*H*W
        s16_thin = s16_thin.view(b, t, *s16_thin.shape[-3:])  # 语法含义？5维还是3维
        s16 = s16.view(b, t, *s16.shape[-3:])
        s8 = s8.view(b, t, *s8.shape[-3:])
        s4 = s4.view(b, t, *s4.shape[-3:])

        return ks16, s16_thin, s16, s8, s4


    def encode_value(self, frame, kf16, mask, other_mask=None):
        # Extract memory key/value for a frame
        if self.single_object:
            f16 = self.value_encoder(frame, kf16, mask)
        else:
            f16 = self.value_encoder(frame, kf16, mask, other_mask)
        return f16.unsqueeze(2)  # B*512*T*H*W

    def segment(self, qk16, qv16, qf8, qf4, mk16, mv16, selector=None):  # 得到query的mask
        # q - query, m - memory
        # qv16 is f16_thin above
        affinity = self.memory.get_affinity(mk16, qk16)

        if self.single_object:
            logits = self.decoder(self.memory.readout(affinity, mv16, qv16), qf8, qf4)
            prob = torch.sigmoid(logits)
        else:
            logits = torch.cat([
                self.decoder(self.memory.readout(affinity, mv16[:, 0], qv16), qf8, qf4),
                self.decoder(self.memory.readout(affinity, mv16[:, 1], qv16), qf8, qf4),
            ], 1)

            prob = torch.sigmoid(logits)
            prob = prob * selector.unsqueeze(2).unsqueeze(2)

        logits = self.aggregate(prob)  # ?
        prob = F.softmax(logits, dim=1)[:, 1:]

        return logits, prob

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_flow':
            return self.encode_flow(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError
