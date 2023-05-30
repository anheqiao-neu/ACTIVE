## ACTIVE: Implementation of the ACTIVE object detection model in Pytorch
---

**Released on May 30, 2023：**   


## 
1. [Environment]
2. [Download]
3. [Train]
4. [Eval]
5. [ACTIVE Network structure]
6. [Reference]


## Environment
torch == 1.2.0

## Dataset  
Dataset is available at: : https://github.com/Demozsj/Detection-Sperm    
 

## Train   
1.Preparation of Datasets

**This article uses VOC format for training, and you need to create your own dataset before training**

Before training, place the label file in "VOCdevkit VOC2007 xx_xx" or "VOCdevkit VOC2007 xx_xx_Annotation". For example, when conducting a 50% crossover experiment, test the label file as "VOCdevkit VOC2007 01_20" and the training label file as "VOCdevkit VOC2007 01_20Annotation".

Before training, place the image file in "VOCdevkit VOC2007 JPEGImages".
2. Processing of Datasets

Modify the 'classes_path' in 'voc_annotation. py' to correspond to 'cls_classes. txt' and run 'voc_annotation. py'.

3. Start network training

**The 'classes_path' and 'phi' in 'train. py' need to be modified, 'classes'_ Path "represents the path," phi=2 "represents branch 2 in active1-4, and" phi=4 "represents branch 2 in active5-8**

After modifying 'classes_path', you can run train.py to start training. After training multiple epochs, the weight file will be generated in 'logs'。  

4. Prediction of training results

Run 'predict. py' for detection. After running, enter the image path to detect it.  

## Eval 
Run 'get_map. py' to obtain the evaluation results, which will be saved in 'map_out'


##ACTIVE Network structure
The corresponding relationships of the eight ACTIVE networks are as follows:   “ACTIVE-I~ACTIVE-VIII” “\net\active1~active8”

## Reference
https://github.com/qqwweee/keras-yolo3  
https://github.com/eriklindernoren/PyTorch-YOLOv3   
https://github.com/BobLiu20/YOLOv3_PyTorch
https://github.com/bubbliiiing/yolo3-pytorch
