import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from yolo import YOLO
from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map

if __name__ == "__main__":
    '''
    Recall和Precision不像AP是一个面积的概念，在门限值不同时，网络的Recall和Precision值是不同的。
    map计算结果中的Recall和Precision代表的是当预测时，门限置信度为0.5时，所对应的Recall和Precision值。

    此处获得的./map_out/detection-results/里面的txt的框的数量会比直接predict多一些，这是因为这里的门限低，
    目的是为了计算不同门限条件下的Recall和Precision值，从而实现map的计算。
    '''
    Recall and Precision are not the same concept of area as AP. When the threshold value is different, the Recall and Precision values of the network are different.
    The Recall and Precision values in the map calculation results represent the Recall and Precision values corresponding to a threshold confidence level of 0.5 during prediction.

    Obtained here/ Map_ The number of boxes in out/detection results/txt will be slightly higher than that in direct predict, because the threshold here is low,

     The purpose is to calculate Recall and Precision values under different threshold conditions, in order to achieve map calculation.
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定该文件运行时计算的内容
    #   map_mode为0代表整个map计算流程，包括获得预测结果、获得真实框、计算VOC_map。
    #   map_mode为1代表仅仅获得预测结果。
    #   map_mode为2代表仅仅获得真实框。
    #   map_mode为3代表仅仅计算VOC_map。
    #   map_mode为4代表利用COCO工具箱计算当前数据集的0.50:0.95map。需要获得预测结果、获得真实框后并安装pycocotools才行
    #   map_mode is used to specify the content calculated during the runtime of the file
    #   map_mode of 0 represents the entire map calculation process, including obtaining predicted results, obtaining true boxes, and calculating VOC_ Map.
    #   map_mode of 1 represents only obtaining predicted results.
    #   map_mode of 2 represents only obtaining the real box.
    #   map_modee of 3 represents only calculating VOC_ Map.
    #   map_mode of 4 represents using the COCO toolbox to calculate the current dataset's 0.50:0.95map. You need to obtain the predicted results, obtain the real box, and install copytools before proceeding
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = 0
    #-------------------------------------------------------#
    #   此处的classes_path用于指定需要测量VOC_map的类别
    #   一般情况下与训练和预测所用的classes_path一致即可
    #   Classes here_ Path is used to specify the need to measure VOC_ Category of map
    #   Generally, it is related to the classes used for training and prediction_ Consistent path is sufficient
    #-------------------------------------------------------#
    classes_path    = 'model_data/new_class.txt'
    #-------------------------------------------------------#
    #   MINOVERLAP用于指定想要获得的mAP0.x
    #   比如计算mAP0.75，可以设定MINOVERLAP = 0.75。
    #  MINOVERLAP is used to specify the mAP0. x you want to obtain
    #  For example, to calculate mAP0.75, MINOVERLAP=0.75 can be set.
    #-------------------------------------------------------#
    MINOVERLAP      = 0.5
    #-------------------------------------------------------#
    #   map_vis用于指定是否开启VOC_map计算的可视化
    #   Map_ Vis is used to specify whether to enable VOC_ Visualization of Map Computing
    #-------------------------------------------------------#
    map_vis         = True
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集

    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #   Point to the folder where the VOC dataset is located
    #   By default, it points to the VOC dataset in the root directory
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    #-------------------------------------------------------#
    #   结果输出的文件夹，默认为map_out
    #  The folder for outputting results, default to map_ Out
    #-------------------------------------------------------#
    map_out_path    = 'E:\E\E\out1'
    a='41_60_'
    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/%sMain/test.txt"% (a))).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO(confidence = 0.5, nms_iou = 0.3)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".png")
            image       = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".png"))
            yolo.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")

    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:

                #数据集修改
                #Dataset modification
                b='41_60'
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/%s/"%b +image_id+".xml" )).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, path = map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")
