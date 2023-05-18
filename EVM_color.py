import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft
import scipy.fftpack as fftpack

class EVM_color:
    def __init__(self,low_freq,high_freq,fps,level,alpha):
        self.low = low_freq
        self.high = high_freq
        self.fps = fps

        # rgb 转 NTSC的参数
        self.T_inv = np.linalg.inv(np.array([[1.0, 0.956, 0.621],[1.0, -0.272, -0.647],[1.0, -1.106, 1.703]])).transpose()
        self.T = np.array([[1.0, 0.956, 0.621],[1.0, -0.272, -0.647],[1.0, -1.106, 1.703]])
        self.levels = level
        self.alpha = alpha
        self.kernel = np.array([[0.062500000000000, 0.250000000000000, 0.375000000000000, 0.250000000000000, 0.062500000000000]])

        self.NTFSFrameList = []

    def RGB2NTSC(self,frame):
        rgb_frame = frame[:,:,::-1]
        reshape_size = rgb_frame.shape[0]*rgb_frame.shape[1]
        reshape_frame = np.reshape(rgb_frame,(reshape_size,3)) / 255.0

        NTFSFrame = np.dot(reshape_frame , self.T_inv)
        NTFSFrame = np.reshape(NTFSFrame,frame.shape)

        return NTFSFrame
    def NTSC2BGR(self,mag_frame):
        reshape_size = mag_frame.shape[0] * mag_frame.shape[1]
        reshape_frame = np.reshape(mag_frame, (reshape_size, 3))
        RGB_img = np.dot(reshape_frame, self.T.transpose())
        RGB_img = np.maximum(RGB_img, 0)
        pos = np.where(RGB_img > 1)
        for r in pos[0]:
            RGB_img[r, :] /= max(RGB_img[r, :])

        matRGB = (np.reshape(RGB_img, mag_frame.shape) * 255.0).astype(np.uint8)
        matBGR = cv2.cvtColor(matRGB, cv2.COLOR_RGB2BGR)
        return matBGR
    def GaussBlurDn(self,img):
        for _ in range(self.levels + 1):
            expand_img = cv2.copyMakeBorder(img, 2, 2, 0, 0, cv2.BORDER_REFLECT_101)
            filter_img = cv2.filter2D(expand_img, -1, self.kernel.transpose())
            img = filter_img[2:filter_img.shape[0] - 2:2, :, :]
            #再对列进行空间滤波
            expand_img = cv2.copyMakeBorder(img, 0, 0, 2, 2, cv2.BORDER_REFLECT_101)
            filter_img = cv2.filter2D(expand_img, -1, self.kernel)
            img = filter_img[:, 2:filter_img.shape[1] - 2:2, :]
        return img

    def Gauss_Pry(self,frame):
        NTFSFrame = self.RGB2NTSC(frame)
        self.NTFSFrameList.append(NTFSFrame)
        img = self.GaussBlurDn(NTFSFrame)

        return img

    ################################  创造理想带通滤波器  ###########################################
    def idealFilter(self,img_list):
        tensor = np.array(img_list)
        fft_data = np.fft.fft(tensor,axis=0)
        frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / self.fps)
        bound_low = (np.abs(frequencies - self.low)).argmin()
        bound_high = (np.abs(frequencies - self.high)).argmin()
        fft_data[:bound_low+1] = 0
        fft_data[bound_high:] = 0
        ifft_data = np.real(np.fft.ifft(fft_data,axis=0))
        return ifft_data

    #######################################  放大 ############################################
    def magnification(self,frameTensor):
        frameTensor *= self.alpha
        return frameTensor

    def recover(self,mag_frame_list):
        Mag_fame = []
        for init_frame,mag_frame in zip(self.NTFSFrameList,mag_frame_list):
            mag_frame = cv2.resize(mag_frame,(init_frame.shape[1],init_frame.shape[0]),cv2.INTER_CUBIC)
            mag_frame += init_frame

            matBGR = self.NTSC2BGR(mag_frame)

            Mag_fame.append(matBGR)
        return Mag_fame