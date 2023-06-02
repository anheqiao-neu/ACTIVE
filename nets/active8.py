from collections import OrderedDict

import torch
import torch.nn as nn

from nets.efficientnet import EfficientNet as EffNet
from nets.darknet import darknet53
from torch.utils.checkpoint import checkpoint

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
    def __init__(self, anchors_mask, num_classes, phi=4, load_weights = False):
        super(YoloBody, self).__init__()

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

        self.last_layercat0 = make_last_layers([736, 1472],
                                            1472,
                                            len(anchors_mask[0]) * (num_classes + 5))

        self.last_layercat1_conv = conv2d(736, 368, 1)
        self.last_layercat1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layercat1 = make_last_layers([368, 736], out_filters[-2] + out_filters2[-2] + 344,
                                            len(anchors_mask[1]) * (num_classes + 5))

        self.last_layercat2_conv = conv2d(368, 184, 1)
        self.last_layercat2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layercat2 = make_last_layers([216, 680], 680,
                                            len(anchors_mask[2]) * (num_classes + 5))

        self.last_layerconcat1_conv = conv2d(720, 344, 1)
        self.last_layerconcat1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layerconcat1 = make_last_layers([368, 1456], 1456,
                                               len(anchors_mask[1]) * (num_classes + 5))


        self.last_layer20 = make_last_layers([512, 1024], out_filters2[-1], len(anchors_mask[0]) * (num_classes + 5))

        self.last_layer21_conv = conv2d(512, 256, 1)
        self.last_layer21_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer21 = make_last_layers([256, 512], out_filters2[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))

        self.last_layer22_conv = conv2d(256, 128, 1)
        self.last_layer22_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer22 = make_last_layers([128, 256], out_filters2[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

        # efficinet使用FPN
        self.last_layere0 = make_last_layers([out_filters[-1], int(out_filters[-1]*2)], out_filters[-1],
                                            len(anchors_mask[0]) * (num_classes + 5))

        self.last_layere1_conv = conv2d(out_filters[-1], out_filters[-2], 1)
        self.last_layere1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layere1 = make_last_layers([160, 320],
                                            320 , len(anchors_mask[1]) * (num_classes + 5))

        self.last_layere2_conv = conv2d(out_filters[-2], out_filters[-3], 1)
        self.last_layere2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layere2 = make_last_layers([out_filters[-3], int(out_filters[-2])],
                                            out_filters[-3]+72, len(anchors_mask[2]) * (num_classes + 5))

        self.down_sample1 = conv22d(216, 432, 3, stride=2)  # self.backbone2 = darknet53()
        self.down_sample2 = conv22d(368, 736, 3, stride=2)
        self.make_five_conv1 = make_last_layers([368, 800], 800, len(anchors_mask[2]) * (num_classes + 5))
        self.make_five_conv2 = make_last_layers([688, 1376], 1376, len(anchors_mask[2]) * (num_classes + 5))
    def forward(self, x):

        x2, x1, x0 = self.backbone(x)
        x22, x21, x20 = self.backbone2(x)

        out0_branch = torch.cat([x0, x20], 1)
        out0_branch = self.last_layercat0[:5](out0_branch)
        # out0 = self.last_layercat0[5:](out0_branch)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        x1_in = self.last_layercat1_conv(out0_branch)
        x1_in = self.last_layercat1_upsample(x1_in)

        # efficinet
        xe21_in = self.last_layere0[:5](x0)
        xe21_in = self.last_layere1_conv(xe21_in)
        xe21_in = self.last_layere1_upsample(xe21_in)
        xe21_in = torch.cat([xe21_in, x1], 1)
        # v3
        xv21_in = self.last_layer20[:5](x20)
        xv21_in = self.last_layer21_conv(xv21_in)
        xv21_in = self.last_layer21_upsample(xv21_in)
        xv21_in = torch.cat([xv21_in, x21], 1)

        # 26,26,256 + 26,26,512 -> 26,26,768
        x31 = torch.cat([xe21_in, xv21_in], 1)
        x21_in = torch.cat([x1_in, x31], 1)

        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out1_branch = self.last_layerconcat1[:5](x21_in)
        # out1 = self.last_layerconcat1[5:](out1_branch)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        x22_in = self.last_layercat2_conv(out1_branch)
        x2_in = self.last_layercat2_upsample(x22_in)

        # efficinet
        xe22_in = self.last_layere1[:5](xe21_in)
        xe22_in = self.last_layere2_conv(xe22_in)
        xe22_in = self.last_layere2_upsample(xe22_in)
        xe22_in = torch.cat([xe22_in, x2], 1)
        # v3
        xv22_in = self.last_layer21[:5](xv21_in)
        xv22_in = self.last_layer22_conv(xv22_in)
        xv22_in = self.last_layer22_upsample(xv22_in)
        xv22_in = torch.cat([xv22_in, x22], 1)
        # 52,52,128 + 52,52,256 -> 52,52,384
        x32 = torch.cat([xe22_in,  xv22_in], 1)
        x2_in = torch.cat([x2_in, x32], 1)
 
        out2 = self.last_layercat2(x2_in)

        out2_branch = self.last_layercat2[:5](x2_in)
        out2_branch = self.down_sample1(out2_branch)
        out1pa_branch = torch.cat([out2_branch, out1_branch], 1)
        out1pa_branch = self.make_five_conv1[:5](out1pa_branch)
        out1 = self.last_layercat1[5:](out1pa_branch)

        out0p_branch = self.down_sample2(out1pa_branch)
        out0p_branch = torch.cat([out0p_branch, out0_branch], 1)
        
        out0 = self.last_layercat0(out0p_branch)
        

        return out0, out1, out2
        