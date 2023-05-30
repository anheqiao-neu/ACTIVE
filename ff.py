import time
import torch
import numpy as np
from nets.yolo_efficientnetb4_pan2 import YoloBody

anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
net = YoloBody(anchors_mask, 2)
net.eval()
net.cuda()

# x是输入图片的大小
x = torch.zeros((1,3,416,416)).cuda()
t_all = []

for i in range(100):

    t1 = time.time()
    y = net(x)
    t2 = time.time()
    t_all.append(t2 - t1)

print('average time:', np.mean(t_all) / 1)
print('average fps:',1 / np.mean(t_all))

print('fastest time:', min(t_all) / 1)
print('fastest fps:',1 / min(t_all))