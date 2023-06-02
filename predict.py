#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#   predict.py integrates functions such as single image prediction, camera detection, FPS testing, and directory traversal detection
#   Integrate into a py file and modify the mode by specifying the mode.
#-----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   mode is used to specify the mode for testing:
    #   'predict' represents a single image prediction. If you want to modify the prediction process, 
    #   such as saving images, capturing objects, etc., you can first see the detailed annotations below
    #   'video' indicates video detection, which can be performed by calling a camera or video. Please refer to the comments below for details.
    #   'fps' indicates testing fps, using the image street.jpg in IMG. Please refer to the comments below for details
    #   'dir '_ 'predict' indicates traversing the folder for detection and saving. 
    #   By default, traverse the img folder and save the img_ Out folder, please refer to the notes below for details.
    #----------------------------------------------------------------------------------------------------------#
    mode = "video"
    #-------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #   video_ Path is used to specify the path of the video, when the video_ When path=0, it indicates the detection camera
    #   video_ Save_ Path represents the path where the video is saved, when the video_ Save_ When path='', it means not saving
    #   video_ Fps for saved videos
    #   video_ Path, video_ Save_ Path and video_ Fps is only valid when mode='video'
    #   When saving a video, you need to exit with Ctrl+C or run until the last frame to complete the complete save steps
    #-------------------------------------------------------------------------#
    video_path      = "E:/e/e/S_0003.mp4"
    video_save_path = "E:/E/E/out1/"
    video_fps       = 25.0
    #-------------------------------------------------------------------------#
    #   test_interval用于指定测量fps的时候，图片检测的次数
    #   理论上test_interval越大，fps越准确。
    #   Test_ Interval is used to specify the number of image detections when measuring fps
    #   Theoretically, test_ The larger the interval, the more accurate the fps.
    #-------------------------------------------------------------------------#
    test_interval   = 100
    #-------------------------------------------------------------------------#
    #   dir_origin_path指定了用于检测的图片的文件夹路径
    #   dir_save_path指定了检测完图片的保存路径
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #   dir_ Origin_ Path specifies the folder path for detecting images
    #   dir_ Save_ The path specifies the save path for the detected image
    #   dir_ Origin_ Path and dir_ Save_ Path is only available in mode='dir_ Valid when 'predict'
    #-------------------------------------------------------------------------#
    dir_origin_path = "E:/E/E/dataset/original videos/S_0003.mp4"
    dir_save_path   = "E:/E/E/out1/"

    if mode == "predict":
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
       1、If you want to save the detected image, use r_ Image. save ("img. jpg") can be saved and modified directly in predict.py.
       2、If you want to obtain the coordinates of the prediction box, 
            you can enter yolo.detect_image function reads the top, left, bottom, and right values in the drawing section.
       3、If you want to use the prediction box to intercept the target, 
            you can enter yolo.detect_image function utilizes the obtained top, left, bottom, and right values in the drawing section
            Use the matrix method to intercept from the original image.
       4、If you want to write additional words on the prediction graph, such as the number of specific targets detected, 
            you can enter yolo.detect_ Image function, in the drawing section for predicted_ Class to make a judgment,
            For example, to determine if predicted_ Class=='car ': This determines whether the current target is a car, and then records the quantity. Write with draw.text  
      '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            # Read a certain fram
            ref,frame=capture.read()
            # 格式转变，BGRtoRGB
            # Format transformation, BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 转变成Image
            # Convert to Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            # Conduct testing
            frame = np.array(yolo.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            # RGBtoBGR meets the opencv display format
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
                
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
