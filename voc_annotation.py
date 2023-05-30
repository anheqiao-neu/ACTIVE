#---------------------------------------------#
#   运行前一定要修改classes
#   如果生成的2007_train.txt里面没有目标信息
#   那么就是因为classes没有设定正确
#---------------------------------------------#
import xml.etree.ElementTree as ET
from os import getcwd
import os

saveBasePath=r'8_101_datatxt/'
if not os.path.exists(saveBasePath):
    os.makedirs(saveBasePath)

a='8_101'
sets=[('2012', 'train'), ('2012', 'val')]
# sets=[('2007', 'test')]
#-----------------------------------------------------#
#   这里设定的classes顺序要和model_data里的txt一样
#-----------------------------------------------------#
classes =['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# classes =['S','Impurity']
def convert_annotation(year, image_id, list_file):
    # in_file = open('VOCdevkit/VOC%s/%s_Annotations/%s.xml'%(year,a, image_id), encoding='utf-8')

    in_file = open('F:\毕业\数据库\VOCdevkit\VOC2012\Annotations/%s.xml' %image_id, encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()
for year, image_set in sets:
    # image_ids = open('VOCdevkit/VOC%s/ImageSets/%s_Main/%s.txt' % (year,a, image_set),encoding='utf-8').read().strip().split()
    image_ids = open('F:\毕业\数据库\VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set), encoding='utf-8').read().strip().split()
    list_file = open(os.path.join(saveBasePath,'%s_%s.txt'%(year, image_set)), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, 2007, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()
