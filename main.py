# import os

# import cv2
# from EVM_color import EVM_color
# vid = '/home/li/3Tdata/data/magnification_video/face0.avi'
# capture = cv2.VideoCapture(vid)
# folder = '/home/li/3Tdata/clion_project/VideoMag/data/face/'
# frame_num = len(os.listdir(folder))
#
#
# EVM_Mag = EVM_color(50.0/60.0,60.0/60.0,30,3,50)
# GassPryList=[]
#
# for i in range(frame_num):
#     frame = cv2.imread(os.path.join(folder,str(i) + '.png'))
#     GaussPry = EVM_Mag.Gauss_Pry(frame)
#     GassPryList.append(GaussPry)
#
# IdealData = EVM_Mag.idealFilter(GassPryList)
# magData = EVM_Mag.magnification(IdealData)
# EVM_Mag.recover(magData)

# from EVM_motion import EVM_motion
# baby_breath_folder = '/home/li/3Tdata/clion_project/VideoMag/data/baby/'
# file_num = len(os.listdir(baby_breath_folder))
#
# alpha = 10
# lambda_c = 16
# r1 = 0.4
# r2 = 0.05
# chromAttenuation = 0.1
# w = 960                     # 图像的宽度
# h = 544                     # 图像的高度
# magMotion = EVM_motion(alpha,lambda_c,r1,r2,chromAttenuation,w,h)
#
# for i in range(file_num):
#     frame = cv2.imread(os.path.join(baby_breath_folder,str(i) + '.png'))
#     mag_frame = magMotion.magnification_motion(frame)
#
#
#     cv2.imshow("frame",mag_frame)
#     cv2.waitKey(1)


import cv2
from EVM_color_video import EVM_Color_Video

vid_path = '/home/li/3Tdata/matlab_project/EVM_Matlab/face.mp4'
capture = cv2.VideoCapture(0)

frame_num = capture.get(cv2.CAP_PROP_FRAME_COUNT)
alpah = 100
bufferFrame = 300               # 做滤波的视频缓冲帧
EVM_Mag = EVM_Color_Video(50.0/60.0,60.0/60.0,30,3,alpah,bufferFrame)

while capture.isOpened():
    ret,frame = capture.read()
    if not ret:
        break

    magFrame = EVM_Mag.MagColor(frame)

    if len(magFrame) > 0:
        cv2.imshow('magFrame',magFrame)

    cv2.imshow('frame',frame)
    cv2.waitKey(1)