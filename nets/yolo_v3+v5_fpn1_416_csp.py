from collections import OrderedDict

import torch
import torch.nn as nn


from nets.darknet import darknet53
from nets.CSPdarknet import cspdarknet53

# class EfficientNet(nn.Module):
#     def __init__(self, phi, load_weights=False):
#         super(EfficientNet, self).__init__()
#         model = EffNet.from_pretrained(f'efficientnet-b{phi}', load_weights)
#         del model._conv_head
#         del model._bn1
#         del model._avg_pooling
#         del model._dropout
#         del model._fc
#         self.model = model
#
#     def forward(self, x):
#         x = self.model._conv_stem(x)
#         x = self.model._bn0(x)
#         x = self.model._swish(x)
#         feature_maps = []
#
#         last_x = None
#         for idx, block in enumerate(self.model._blocks):
#             drop_connect_rate = self.model._global_params.drop_connect_rate
#             if drop_connect_rate:
#                 drop_connect_rate *= float(idx) / len(self.model._blocks)
#             x = block(x, drop_connect_rate=drop_connect_rate)
#             if block._depthwise_conv.stride == [2, 2]:
#                 feature_maps.append(last_x)
#             elif idx == len(self.model._blocks) - 1:
#                 feature_maps.append(x)
#             last_x = x
#         del last_x
#         out_feats = [feature_maps[2],feature_maps[3],feature_maps[4]]
#         return out_feats


def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
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

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1,1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

# class Conv(nn.Module):
#     # Standard convolution
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
def Conv(filter_in, filter_out, kernel_size,stride_size,):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride_size, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("silu", nn.SiLU(0.1)),
    ]))

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
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes):
        super(YoloBody, self).__init__()
        # ---------------------------------------------------#
        #   生成darknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        # ---------------------------------------------------#
        self.backbone = darknet53()

        # ---------------------------------------------------#
        #   out_filters : [64, 128, 256, 512, 1024]
        # ---------------------------------------------------#
        out_filters = self.backbone.layers_out_filters

        # ------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        # ------------------------------------------------------------------------#
        # self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))
        #
        # self.last_layer1_conv = conv2d(512, 256, 1)
        # self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))
        #
        # self.last_layer2_conv = conv2d(256, 128, 1)
        # self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

        self.backbone2 = cspdarknet53()
        out_filters2 = self.backbone2.layers_out_filters
        #------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        #------------------------------------------------------------------------#
        self.last_layer02 = make_last_layers([512, 1536], 1536, len(anchors_mask[0]) * (num_classes + 5))

        self.last_layer12_conv = conv2d(512, 256, 1)
        self.last_layer12_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.last_layer12 = make_last_layers([256*2, 512*2], out_filters2[-2]*2 + 256*4, len(anchors_mask[1]) * (num_classes + 5))
        self.last_layer12 = make_last_layers([256 , 1024], 1024,
                                             len(anchors_mask[1]) * (num_classes + 5))
        self.last_layer22_conv = conv2d(256, 128, 1)
        self.last_layer22_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.last_layer22 = make_last_layers([128*2, 256*2], out_filters2[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))
        self.last_layer22 = make_last_layers([128, 512], 512,
                                         len(anchors_mask[1]) * (num_classes + 5))

        self.CONV_0=Conv(1536,512,1,1)
        self.c3_0=C3(c1=1536, c2=512, n=1)
        self.c3_1 = C3(c1=1024, c2=256, n=1)
        self.c3_2 = C3(c1=1024, c2=128, n=1)
    def forward(self, x):
        #---------------------------------------------------#   
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256；26,26,512；13,13,1024
        #---------------------------------------------------#
        x2, x1, x0 = self.backbone(x)
        x22, x21, x20 = self.backbone2(x)
        #---------------------------------------------------#
        #   第一个特征层
        #   out0 = (batch_size,255,13,13)
        #---------------------------------------------------#
        # ____________V3_FPN___________________________#
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        x30=torch.cat([x0, x20],1)
        out0_branch = self.c3_0(x30)
        out0        = self.last_layer02[5:](out0_branch)


        # 13,13,512 -> 13,13,256 -> 26,26,256
        x11_in = self.last_layer12_conv(out0_branch)
        x1_in = self.last_layer12_upsample(x11_in)

        # 26,26,256 + 26,26,512 -> 26,26,768
        x31= torch.cat([x1, x21],1)
        x21_in = torch.cat([x1_in, x31], 1)
        #---------------------------------------------------#
        #   第二个特征层
        #   out1 = (batch_size,255,26,26)
        #---------------------------------------------------#
        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out1_branch = self.c3_1(x21_in)

        out1        = self.last_layer12[5:](out1_branch)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.last_layer22_conv(out1_branch)
        x2_in = self.last_layer22_upsample(x2_in)

        # 52,52,128 + 52,52,256 -> 52,52,384
        x32 = torch.cat([x2, x22], 1)
        x2_in = torch.cat([x2_in, x32], 1)
        #---------------------------------------------------#
        #   第一个特征层
        #   out3 = (batch_size,255,52,52)
        #---------------------------------------------------#
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        out2_branch = self.c3_2(x21_in)

        out2 = self.last_layer22[5:](out2_branch)

       #-------------------------V5_FPN
        return out0, out1, out2

