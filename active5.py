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
        ("relu", nn.LeakyReLU(0.1)),
    ]))


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

class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi=4, load_weights = False):
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

        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        #------------------------------------------------------------------------#
        self.last_layer0            = make_last_layers([688, int(out_filters[-1]+out_filters2[-1])], out_filters[-1]+out_filters2[-1], len(anchors_mask[0]) * (num_classes + 5))

        self.last_layer1_conv       = conv2d(688, 344, 1)
        self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1            = make_last_layers([344, 688], out_filters[-2]+out_filters2[-2]+344, len(anchors_mask[1]) * (num_classes + 5))

        self.last_layer2_conv       = conv2d(344, 172, 1)
        self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2            = make_last_layers([172,344], out_filters[-3]+out_filters2[-3]+172, len(anchors_mask[2]) * (num_classes + 5))


        # out_filters2 = self.backbone2.layers_out_filters
        # ------------------------------------------------------------------------#

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

        x2, x1, x0 = self.backbone(x)
        x22, x21, x20 = self.backbone2(x)
      
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        x30=torch.cat([x0, x20],1)
        out0_branch = self.last_layer0[:5](x30)
        out0        = self.last_layer0[5:](out0_branch)


        # 13,13,512 -> 13,13,256 -> 26,26,256
        x11_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x11_in)

        # 26,26,256 + 26,26,512 -> 26,26,768
        x31= torch.cat([x1, x21],1)
        x21_in = torch.cat([x1_in, x31], 1)
 
        out1_branch = self.last_layer1[:5](x21_in)

        out1        = self.last_layer1[5:](out1_branch)

        return out0, out1, out2

