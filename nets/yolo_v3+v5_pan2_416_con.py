from collections import OrderedDict

import torch
import torch.nn as nn

from nets.efficientnet import EfficientNet as EffNet
from nets.darknet import darknet53
from nets.CSPdarknet import cspdarknet53

def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

def conv22d(filter_in, filter_out, kernel_size,stride_size=1,p=None, g=1, act=True):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride_size, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("silu", nn.SiLU(0.1)),
    ]))
#------------------------------------------------------------------------#
#   make_last_layers里面一共有七个卷积，前五个用于提取特征。
#   后两个用于获得yolo网络的预测结果
#------------------------------------------------------------------------#
def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )
    return m

class C3f(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = conv22d(c1, c_, 1, 1)
        self.cv2 = conv22d(c1, c_, 1, 1)
        self.cv3 = conv22d(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        # return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = conv22d(c1, c_, 1, 1)
        self.cv2 = conv22d(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi=0, load_weights = False):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#   
        #   生成darknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.backbone =cspdarknet53()

        self.backbone2 = darknet53()
        out_filters2 = self.backbone2.layers_out_filters
        #------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        #------------------------------------------------------------------------#
        ut_filters2 = self.backbone2.layers_out_filters
        # ------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        # ------------------------------------------------------------------------#
        self.last_layercat0 = make_last_layers([688, 1536],
                                               512 + out_filters2[-1],
                                               len(anchors_mask[0]) * (num_classes + 5))
        self.last_layercat00 =make_last_layers([688, 1376],
                                               1376,
                                               len(anchors_mask[0]) * (num_classes + 5))
        self.last_layercat1_conv = conv2d(688, 344, 1)
        self.last_layercat1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layercat1 = make_last_layers([344, 1496], 1496,
                                               len(anchors_mask[1]) * (num_classes + 5))

        self.last_layercat2_conv = conv2d(344, 172, 1)
        self.last_layercat2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layercat2 = make_last_layers([172, 812], 812,
                                               len(anchors_mask[2]) * (num_classes + 5))

        self.last_layerconcat1_conv = conv2d(720, 344, 1)
        self.last_layerconcat1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layerconcat1 = make_last_layers([344, 1352], 1352,
                                                  len(anchors_mask[1]) * (num_classes + 5))

        # ------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        # ------------------------------------------------------------------------#
        self.last_layer20 = make_last_layers([512, 1024], out_filters2[-1], len(anchors_mask[0]) * (num_classes + 5))

        self.last_layer21_conv = conv2d(512, 256, 1)
        self.last_layer21_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer21 = make_last_layers([256, 768], out_filters2[-2] + 256,
                                             len(anchors_mask[1]) * (num_classes + 5))

        self.last_layer22_conv = conv2d(256, 128, 1)
        self.last_layer22_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer22 = make_last_layers([128, 384], out_filters2[-3] + 128,
                                             len(anchors_mask[2]) * (num_classes + 5))

        # efficinet使用FPN
        self.last_layere0 = make_last_layers([256, 512], 512,len(anchors_mask[0]) * (num_classes + 5))

        self.last_layere1_conv = conv2d(256, 128, 1)
        self.last_layere1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layere1 = make_last_layers([256, 384],
                                             384, len(anchors_mask[1]) * (num_classes + 5))

        self.last_layere2_conv = conv2d(256, 128, 1)
        self.last_layere2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layere2 = make_last_layers([128, 512],
                                             128 + 72, len(anchors_mask[2]) * (num_classes + 5))

        self.down_sample1 = conv22d(172, 344, 3, stride_size=2)  # self.backbone2 = darknet53()
        self.down_sample2 = conv22d(344, 688, 3, stride_size=2)
        self.make_five_conv1 = make_last_layers([344, 688], 688, len(anchors_mask[2]) * (num_classes + 5))
        self.make_five_conv2 = make_last_layers([688, 1376], 1376, len(anchors_mask[2]) * (num_classes + 5))

        # self.c3_ = C3f(c1=1536, c2=688, n=1)
        # self.c3_0 = C3f(c1=512, c2=512, n=1)
        # self.c3_1 = C3f(c1=1024, c2=512, n=1)
        # self.c3_2 = C3f(c1=1624, c2=344, n=1)
        # self.c3_3 = C3f(c1=512, c2=256, n=1)
        # self.c3_4 = C3f(c1=768, c2=256, n=1)
        # self.c3_5 = C3f(c1=812, c2=172, n=1)
        # self.c3_6 = C3f(c1=688, c2=344, n=1)
        # self.c3_7 = C3f(c1=1376, c2=688, n=1)
    def forward(self, x):
        #---------------------------------------------------#   
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256；26,26,512；13,13,1024
        #---------------------------------------------------#
        # ---------------------------------------------------#
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256；26,26,512；13,13,1024
        # ---------------------------------------------------#
        x2, x1, x0 = self.backbone(x)
        x22, x21, x20 = self.backbone2(x)
        # ---------------------------------------------------#
        #   第一个特征层
        #   out0 = (batch_size,255,13,13)
        # ---------------------------------------------------#
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        out0_branch = torch.cat([x0, x20], 1)
        out0_branch = self.last_layercat0[:5](out0_branch)
        # out0 = self.last_layercat0[5:](out0_branch)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        x1_in = self.last_layercat1_conv(out0_branch)
        x1_in = self.last_layercat1_upsample(x1_in)

        # efficinet
        xe21_in = self.last_layere0[:5] (x0)
        xe21_in = self.last_layere1_conv(xe21_in)
        xe21_in = self.last_layere1_upsample(xe21_in)
        xe21_in = torch.cat([xe21_in, x1], 1)
        # v3
        xv21_in = self.last_layer20[:5] (x20)
        xv21_in = self.last_layer21_conv(xv21_in)
        xv21_in = self.last_layer21_upsample(xv21_in)
        xv21_in = torch.cat([xv21_in, x21], 1)

        # 26,26,256 + 26,26,512 -> 26,26,768
        x31 = torch.cat([xe21_in, xv21_in], 1)
        x21_in = torch.cat([x1_in, x31], 1)
        # ---------------------------------------------------#
        #   第二个特征层
        #   out1 = (batch_size,255,26,26)
        # ---------------------------------------------------#
        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out1_branch = self.last_layercat1[:5] (x21_in)
        # out1 = self.last_layerconcat1[5:](out1_branch)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        x22_in = self.last_layercat2_conv(out1_branch)
        x2_in = self.last_layercat2_upsample(x22_in)

        # efficinet
        xe22_in = self.last_layere1[:5] (xe21_in)
        xe22_in = self.last_layere2_conv(xe22_in)
        xe22_in = self.last_layere2_upsample(xe22_in)
        xe22_in = torch.cat([xe22_in, x2], 1)
        # v3
        xv22_in = self.last_layer21[:5] (xv21_in)
        xv22_in = self.last_layer22_conv(xv22_in)
        xv22_in = self.last_layer22_upsample(xv22_in)
        xv22_in = torch.cat([xv22_in, x22], 1)
        # 52,52,128 + 52,52,256 -> 52,52,384
        x32 = torch.cat([xe22_in,  xv22_in], 1)
        x2_in = torch.cat([x2_in, x32], 1)
        # ---------------------------------------------------#
        #   第一个特征层
        #   out3 = (batch_size,255,52,52)
        # ---------------------------------------------------#
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        out2 = self.last_layercat2(x2_in)

        out2_branch = self.last_layercat2[:5] (x2_in)
        out2_branch = self.down_sample1(out2_branch)
        out1pa_branch = torch.cat([out2_branch, out1_branch], 1)
        out1pa_branch = self.make_five_conv1[:5] (out1pa_branch)
        out1 = self.last_layercat1[5:](out1pa_branch)

        out0p_branch = self.down_sample2(out1pa_branch)
        out0p_branch = torch.cat([out0p_branch, out0_branch], 1)
        # out0pan_branch = self.c3_7(out0p_branch)
        out0 = self.last_layercat00(out0p_branch)
        # x2_in.requires_grad_(True)
        # x2_in = checkpoint(lambda _:m(_), x2_in)
        # out2 = checkpoint(self.last_layercat2,x2_in)
        #
        # out2_branch = checkpoint(self.last_layercat2[:5],x2_in)
        # out2_branch = checkpoint(self.down_sample1,out2_branch)
        # out1pa_branch = torch.cat([out2_branch, out1_branch], 1)
        # out1pa_branch = checkpoint(self.make_five_conv1[:5],out1pa_branch)
        # out1 = checkpoint(self.last_layercat1[5:],out1pa_branch)
        #
        # out0p_branch = checkpoint(self.down_sample2,out1pa_branch)
        # out0p_branch = torch.cat([out0p_branch, out0_branch], 1)
        # # out0pan_branch = self.make_five_conv2[:5](out0pa_branch)
        # out0 = checkpoint(self.last_layercat0,out0p_branch)

        return out0, out1, out2
        # # ---------------------------------------------------#
        # #   获得三个有效特征层，他们的shape分别是：
        # #   52,52,256；26,26,512；13,13,1024
        # # ---------------------------------------------------#
        # x22, x21, x20 = self.backbone2(x)
        #
        # # ---------------------------------------------------#
        # #   第一个特征层
        # #   out0 = (batch_size,255,13,13)
        # # ---------------------------------------------------#
        # # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        # out20_branch = self.last_layer0[:5](x20)
        # out20 = self.last_layer0[5:](out20_branch)
        #
        # # 13,13,512 -> 13,13,256 -> 26,26,256
        # x21_in = self.last_layer1_conv(out20_branch)
        # x21_in = self.last_layer1_upsample(x21_in)
        #
        # # 26,26,256 + 26,26,512 -> 26,26,768
        # x21_in = torch.cat([x21_in, x21], 1)
        # # ---------------------------------------------------#
        # #   第二个特征层
        # #   out1 = (batch_size,255,26,26)
        # # ---------------------------------------------------#
        # # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        # out21_branch = self.last_layer1[:5](x21_in)
        # out21 = self.last_layer1[5:](out21_branch)
        #
        # # 26,26,256 -> 26,26,128 -> 52,52,128
        # x22_in = self.last_layer2_conv(out21_branch)
        # x22_in = self.last_layer2_upsample(x22_in)
        #
        # # 52,52,128 + 52,52,256 -> 52,52,384
        # x22_in = torch.cat([x22_in, x22], 1)
        # # ---------------------------------------------------#
        # #   第一个特征层
        # #   out3 = (batch_size,255,52,52)
        # # ---------------------------------------------------#
        # # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        # out22 = self.last_layer2(x22_in)
        #
        # out30 = torch.cat([out0, out20], 1)
        # out31 = torch.cat([out1, out21], 1)
        # out32 = torch.cat([out2, out22], 1)
        # return out30, out31, out32

