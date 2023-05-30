import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

#-------------------------------------------------#
#   MISH激活函数
#-------------------------------------------------#
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

#---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   Conv2d + BatchNormalization + Mish
#---------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

#---------------------------------------------------#
#   CSPdarknet的结构块的组成部分
#   内部堆叠的残差块
#---------------------------------------------------#
class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super(Resblock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3)
        )

    def forward(self, x):
        return x + self.block(x)

#--------------------------------------------------------------------#
#   CSPdarknet的结构块
#   首先利用ZeroPadding2D和一个步长为2x2的卷积块进行高和宽的压缩
#   然后建立一个大的残差边shortconv、这个大残差边绕过了很多的残差结构
#   主干部分会对num_blocks进行循环，循环内部是残差结构。
#   对于整个CSPdarknet的结构块，就是一个大残差块+内部多个小残差块
#--------------------------------------------------------------------#
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(Resblock_body, self).__init__()
        #----------------------------------------------------------------#
        #   利用一个步长为2x2的卷积块进行高和宽的压缩
        #----------------------------------------------------------------#
        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=2)

        if first:
            #--------------------------------------------------------------------------#
            #   然后建立一个大的残差边self.split_conv0、这个大残差边绕过了很多的残差结构
            #--------------------------------------------------------------------------#
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)

            #----------------------------------------------------------------#
            #   主干部分会对num_blocks进行循环，循环内部是残差结构。
            #----------------------------------------------------------------#
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)  
            self.blocks_conv = nn.Sequential(
                Resblock(channels=out_channels, hidden_channels=out_channels//2),
                BasicConv(out_channels, out_channels, 1)
            )

            self.concat_conv = BasicConv(out_channels*2, out_channels, 1)
        else:
            #--------------------------------------------------------------------------#
            #   然后建立一个大的残差边self.split_conv0、这个大残差边绕过了很多的残差结构
            #--------------------------------------------------------------------------#
            self.split_conv0 = BasicConv(out_channels, out_channels//2, 1)

            #----------------------------------------------------------------#
            #   主干部分会对num_blocks进行循环，循环内部是残差结构。
            #----------------------------------------------------------------#
            self.split_conv1 = BasicConv(out_channels, out_channels//2, 1)
            self.blocks_conv = nn.Sequential(
                *[Resblock(out_channels//2) for _ in range(num_blocks)],
                BasicConv(out_channels//2, out_channels//2, 1)
            )

            self.concat_conv = BasicConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)

        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)

        #------------------------------------#
        #   将大残差边再堆叠回来
        #------------------------------------#
        x = torch.cat([x1, x0], dim=1)
        #------------------------------------#
        #   最后对通道数进行整合
        #------------------------------------#
        x = self.concat_conv(x)

        return x

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad

    return p

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)
    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))

class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = BasicConv(c1, c_, 1, 1)
        self.cv2 = BasicConv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
#---------------------------------------------------#
#   CSPdarknet53 的主体部分
#   输入为一张416x416x3的图片
#   输出为三个有效特征层
#---------------------------------------------------#
class CSPDarkNet(nn.Module):
    def __init__(self,layer):
        super(CSPDarkNet, self).__init__()
        # backbone
        self.inplanes = 64
        # 416,416,3 -> 416,416,32
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.inplanes)
        # self.relu1 = nn.LeakyReLU(0.1)
        self.Focus = Focus(c1=3, c2=32, k=3, s=1)
        self.CBL_1 = Conv(c1=32, c2=64, k=3, s=2)
        self.CSP_1 = C3(c1=64, c2=64, n=layer[0])
        self.CBL_2 = Conv(c1=64, c2=128, k=3, s=2)
        self.CSP_2 = C3(c1=128, c2=128, n=layer[1])
        self.CBL_3 = Conv(c1=128, c2=256, k=3, s=2)
        self.CSP_3 = C3(c1=256, c2=256, n=layer[2])
        self.CBL_4 = Conv(c1=256, c2=512, k=3, s=2)
        self.SPP = SPP(c1=512, c2=512, k=(5, 9, 13))
        self.CSP_4 = C3(c1=512, c2=512, n=layer[3])
        # self.Focus = Focus(c1=3, c2=32 , k=3, s=1)
        # self.CBL_1 = Conv(c1=32, c2=64 , k=3, s=2)
        # self.CSP_1 = BottleneckCSP(c1=64 , c2=64 * 2, n=layer[0])
        # self.CBL_2 = Conv(c1=64 * 2, c2=128 , k=3, s=2)
        # self.CSP_2 = BottleneckCSP(c1=128 , c2=128 * 2, n=layer[1])
        # self.CBL_3 = Conv(c1=128 * 2, c2=256 , k=3, s=2)
        # self.CSP_3 = BottleneckCSP(c1=256 , c2=256 * 2, n=layer[2])
        # self.CBL_4 = Conv(c1=256 * 2, c2=512, k=3, s=2)
        # self.SPP = SPP(c1=512 * 2, c2=512 , k=(5, 9, 13))
        # self.CSP_4 = BottleneckCSP(c1=512 , c2=512 * 2, n=layer[3])
        self.layers_out_filters = [64, 128, 256, 512]

        # head
        # self.CSP_4 = BottleneckCSP(c1=512, c2=512, n=1, shortcut=False)
        #
        # self.CBL_5 = Conv(c1=512, c2=256, k=1, s=1)
        # self.Upsample_5 = nn.Upsample(size=None, scale_factor=2, mode="nearest")
        # self.Concat_5 = Concat(dimension=1)
        # self.CSP_5 = BottleneckCSP(c1=512, c2=256, n=1, shortcut=False)
        #
        # self.CBL_6 = Conv(c1=256, c2=128, k=1, s=1)
        # self.Upsample_6 = nn.Upsample(size=None, scale_factor=2, mode="nearest")
        # self.Concat_6 = Concat(dimension=1)
        # self.CSP_6 = BottleneckCSP(c1=256, c2=128, n=1, shortcut=False)
        # self.Conv_6 = nn.Conv2d(in_channels=128, out_channels=self.output_ch, kernel_size=1, stride=1)
        #
        # self.CBL_7 = Conv(c1=128, c2=128, k=3, s=2)
        # self.Concat_7 = Concat(dimension=1)
        # self.CSP_7 = BottleneckCSP(c1=256, c2=256, n=1, shortcut=False)
        # self.Conv_7 = nn.Conv2d(in_channels=256, out_channels=self.output_ch, kernel_size=1, stride=1)
        #
        # self.CBL_8 = Conv(c1=256, c2=256, k=3, s=2)
        # self.Concat_8 = Concat(dimension=1)
        # self.CSP_8 = BottleneckCSP(c1=512, c2=512, n=1, shortcut=False)
        # self.Conv_8 = nn.Conv2d(in_channels=512, out_channels=self.output_ch, kernel_size=1, stride=1)

        # # detection
        # self.Detect = Detect(nc=self.class_num, anchors=self.anchors)

    def forward(self, x):
        # backbone
        x = self.Focus(x)  # 0
        x = self.CBL_1(x)
        x = self.CSP_1(x)
        x = self.CBL_2(x)
        y1 = self.CSP_2(x)  # 4
        x = self.CBL_3(y1)
        y2 = self.CSP_3(x)  # 6
        x = self.CBL_4(y2)
        x = self.SPP(x)
        y3 = self.CSP_4(x)
        out1 = y1
        out2 = y2
        out3 = y3
        # # head
        # x = self.CSP_4(x)
        #
        # y3 = self.CBL_5(x)  # 10
        # x = self.Upsample_5(y3)
        # x = self.Concat_5([x, y2])
        # x = self.CSP_5(x)
        #
        # y4 = self.CBL_6(x)  # 14
        # x = self.Upsample_6(y4)
        # x = self.Concat_6([x, y1])
        # y5 = self.CSP_6(x)  # 17
        # output_1 = self.Conv_6(y5)  # 18 output_1
        #
        # x = self.CBL_7(y5)
        # x = self.Concat_7([x, y4])
        # y6 = self.CSP_7(x)  # 21
        # output_2 = self.Conv_7(y6)  # 22 output_2
        #
        # x = self.CBL_8(y6)
        # x = self.Concat_8([x, y3])
        # x = self.CSP_8(x)
        # output_3 = self.Conv_8(x)  # 26 output_3

        # output = self.Detect([output_1, output_2, output_3])
        return out1, out2, out3

# def cspdarknet53(pretrained, **kwargs):
#     model = CSPDarkNet([3, 9, 9, 3])
#     if pretrained:
#         if isinstance(pretrained, str):
#             model.load_state_dict(torch.load(pretrained))
#         else:
#             raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
def cspdarknet53():
    model = CSPDarkNet([1, 3, 3, 1])
    return model
