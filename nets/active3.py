from collections import OrderedDict

import torch
import torch.nn as nn

from nets.efficientnet import EfficientNet as EffNet
from nets.darknet import darknet53

class EfficientNet(nn.Module):
    def __init__(self, phi, load_weights=False):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{phi}', load_weights)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        out_feats = [feature_maps[2],feature_maps[3],feature_maps[4]]
        return out_feats


def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1,inplace=True)),
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
def conv22d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.SiLU(0.1)),
    ]))

class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi=2, load_weights = False):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#   
        #   生成darknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.backbone = EfficientNet(phi, load_weights = load_weights)

        out_filters = {
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [80, 224, 640],
        }[phi]
        self.backbone2 = darknet53()
        out_filters2 = self.backbone2.layers_out_filters
        #------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        #------------------------------------------------------------------------#
        self.last_layer0            = make_last_layers([688, int(out_filters[-1]+out_filters2[-1])], out_filters[-1]+out_filters2[-1], len(anchors_mask[0]) * (num_classes + 5))

        self.last_layer1_conv       = conv2d(688, 344, 1)
        self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1            = make_last_layers([344, 688], out_filters[-2]+out_filters2[-2]+344, len(anchors_mask[1]) * (num_classes + 5))

        self.last_layer2_conv       = conv2d(344, 172, 1)
        self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2            = make_last_layers([172,344], out_filters[-3]+out_filters2[-3]+172, len(anchors_mask[2]) * (num_classes + 5))

        self.down_sample1 = conv22d(172, 344, 3, stride=2)  # self.backbone2 = darknet53()
        self.down_sample2 = conv22d(344, 688, 3, stride=2)
        self.make_five_conv1 = make_last_layers([344, 688], 688,len(anchors_mask[2]) * (num_classes + 5))
        self.make_five_conv2 = make_last_layers([688, 1376], 1376,len(anchors_mask[2]) * (num_classes + 5))
        # self.backbone2 = darknet53()
        # out_filters2 = self.backbone2.layers_out_filters
        # ------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        # ------------------------------------------------------------------------#
        # self.last_layer20 = make_last_layers([512, 1024], out_filters2[-1], len(anchors_mask[0]) * (num_classes + 5))
        #
        # self.last_layer21_conv = conv2d(512, 256, 1)
        # self.last_layer21_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.last_layer21 = make_last_layers([256, 512], out_filters2[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))
        #
        # self.last_layer22_conv = conv2d(256, 128, 1)
        # self.last_layer22_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.last_layer22 = make_last_layers([128, 256], out_filters2[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))
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
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        x30=torch.cat([x0, x20],1)
        out0_branch = self.last_layer0[:5](x30)
        # out0        = self.last_layer0[5:](out0_branch)


        # 13,13,512 -> 13,13,256 -> 26,26,256
        x11_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x11_in)

        # 26,26,256 + 26,26,512 -> 26,26,768
        x31= torch.cat([x1, x21],1)
        x21_in = torch.cat([x1_in, x31], 1)
        #---------------------------------------------------#
        #   第二个特征层
        #   out1 = (batch_size,255,26,26)
        #---------------------------------------------------#
        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out1_branch = self.last_layer1[:5](x21_in)

        # out1        = self.last_layer1[5:](out1_branch)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)

        # 52,52,128 + 52,52,256 -> 52,52,384
        x32 = torch.cat([x2, x22], 1)
        x2_in = torch.cat([x2_in, x32], 1)
        #---------------------------------------------------#
        #   第一个特征层
        #   out3 = (batch_size,255,52,52)
        #---------------------------------------------------#
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,
        out2_branch = self.last_layer2[:5](x2_in)
        out2 = self.last_layer2[5:](out2_branch)
        # 下采样
        out1p_branch = self.down_sample1(out2_branch)
        out1pa_branch = torch.cat([out1p_branch, out1_branch], 1)
        out1pan_branch = self.make_five_conv1[:5](out1pa_branch)
        out1 = self.last_layer1[5:](out1pan_branch)

        out0p_branch = self.down_sample2(out1pan_branch)
        out0pa_branch = torch.cat([out0p_branch, out0_branch], 1)
        out0pan_branch = self.make_five_conv2[:5](out0pa_branch)
        out0 = self.last_layer0[5:](out0pan_branch)
        # # ---------------------------------------------------#
        # #   获得三个有效特征层，他们的shape分别是：
        # #   52,52,256；26,26,512；13,13,1024
        # # ---------------------------------------------------#
        # x22, x21, x20 = self.backbone(x)
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

        # out30 = torch.cat([out0, out20], 1)
        # out31 = torch.cat([out1, out21], 1)
        # out32 = torch.cat([out2, out22], 1)
        return out0, out1, out2

