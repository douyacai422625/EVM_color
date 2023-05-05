import os

import cv2
from EVM import EVM

# vid = '/home/li/3Tdata/data/magnification_video/face0.avi'
# capture = cv2.VideoCapture(vid)
folder = '/home/li/3Tdata/clion_project/VideoMag/data/face/'
frame_num = len(os.listdir(folder))


EVM_Mag = EVM(50.0/60.0,60.0/60.0,30,3,50)
GassPryList=[]

for i in range(frame_num):
    frame = cv2.imread(os.path.join(folder,str(i) + '.png'))
    GaussPry = EVM_Mag.Gauss_Pry(frame)
    GassPryList.append(GaussPry)

IdealData = EVM_Mag.idealFilter(GassPryList)
magData = EVM_Mag.magnification(IdealData)
EVM_Mag.recover(magData)
