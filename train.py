#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import YOLOLoss, weights_init
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, get_classes
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
        #   Whether to use Cuda
        #   No GPU can be set to False
    #-------------------------------#
    Cuda = True
    #--------------------------------------------------------#
    #   训练前一定要修改classes_path，使其对应自己的数据集
    #   Before training, be sure to modify the classes_path to correspond to your own dataset
    #--------------------------------------------------------#
    classes_path    = 'model_data/new_class.txt'
    #---------------------------------------------------------------------#
    #   anchors_path代表先验框对应的txt文件，一般不修改。
    #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        #   anchors_ Path represents the txt file corresponding to the prior box and is generally not modified
        #   anchors_ Mask is used to help code find the corresponding prior box and is generally not modified.
    #---------------------------------------------------------------------#
    anchors_path    = 'model_data/yolo_anchors.txt'
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    #----------------------------------------------------------------------------------------------------------------------------#


    #   如果想要让模型从0开始训练，则设置model_path = ''，下面的pretrain = Fasle，Freeze_Train = Fasle，
    #  If you want the model to start training from 0, set the model_ Path='', pre train=Fasle, Freeze below_ Train=Fasle,
    #----------------------------------------------------------------------------------------------------------------------------#
    #model_path      = 'model_data\efficientnet-b4-6ed6700e.pth'
    model_path = ''
    #------------------------------------------------------#
    #   输入的shape大小，一定要是32的倍数
    #   The size of the input image must be a multiple of 32.
    #------------------------------------------------------#
    input_shape     = [416, 416]
    #----------------------------------------------------#
    #   支路2的版本
    #   Version of Branch 2
    #----------------------------------------------------#
    phi             = 4 
    #可选0~8，根据内存
    #Optional 0-8, depending on memory
    #----------------------------------------------------------------------------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = False

    #----------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #   受到BatchNorm层影响，batch_size最小为2，不能为1。
    #The training is divided into two stages, namely the freezing stage and the thawing stage.
    #Insufficient graphics memory is not related to the size of the dataset. If there is insufficient graphics memory, please reduce the batch size_ Size.
    #Affected by the BatchNorm layer, batch_ The minimum size is 2 and cannot be 1
    #----------------------------------------------------#
    #----------------------------------------------------#
    #   冻结阶段训练参数
    #  Freeze phase training parameters
    #----------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 4
    Freeze_lr           = 1e-3
    #----------------------------------------------------#
    #   解冻阶段训练参数
    #   Training parameters during thawing phase
    #----------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 2
    Unfreeze_lr         = 1e-4
    #------------------------------------------------------#
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    #   Whether to perform freeze training on the training parameters during the thawing stage, 
    #   the default is to freeze the main training first and then thaw the training.
    #------------------------------------------------------#
    Freeze_Train        = True
    #------------------------------------------------------#
    #   用于设置是否使用多线程读取数据
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   内存较小的电脑可以设置为2或者0  
    #Used to set whether to use multithreading to read data
    #Enabling it will accelerate data reading speed, but it will occupy more memory
    #Computers with smaller memory can be set to 2 or 0
    #------------------------------------------------------#
    num_workers         = 0
    #----------------------------------------------------#
    #   获得图片路径和标签
    #   Obtain image paths and labels
    #----------------------------------------------------#
    annotation_path   = '81_101_datatxt/2007_train.txt'


    #----------------------------------------------------#
    #   获取classes和anchor
    #   Obtain classes and anchor
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)

    #------------------------------------------------------#
    #   创建ACTIVE模型
    #   Create ACTIVE model
    #------------------------------------------------------#
    model = YoloBody(anchors_mask, num_classes, phi=phi, load_weights=pretrained)
    print(model)
    # para = sum([np.prod(list(p.size())) for p in model.parameters()])
    # print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))

    if not pretrained:
        weights_init(model)
    if model_path != '':
        #------------------------------------------------------#
  
        #------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)



    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        torch.nn.parallel.DistributedDataParallel
        cudnn.benchmark = True
        model_train = model_train.cuda()

    yolo_loss    = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask)
    loss_history = LossHistory("F:\yolox_result/model/81_101_logs/")

    #---------------------------#
    #   读取数据集对应的txt
    #   Read the txt corresponding to the dataset
    #---------------------------#
    val_split = 0.2
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   UnFreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #The backbone feature extraction network features are universal, and frozen training can accelerate training speed
    #It can also prevent weight damage during the early stages of training.
    #Init_ Epoch is the starting generation
    #Freeze_ Epoch is a generation of frozen training
    #UnFreeze_ Epoch Total Training Generation
    #Prompt for OOM or insufficient graphics memory, please reduce the batch size_ Size
    #------------------------------------------------------#
    if True:
        batch_size  = Freeze_batch_size
        lr          = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch
        
        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset   = YoloDataset(lines[:num_train], input_shape, num_classes, train = True)
        val_dataset     = YoloDataset(lines[num_train:], input_shape, num_classes, train = False)
        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate)
                        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()


    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch
        
        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset   = YoloDataset(lines[:num_train], input_shape, num_classes, train = True)
        val_dataset     = YoloDataset(lines[num_train:], input_shape, num_classes, train = False)
        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate)
                        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = True

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()
