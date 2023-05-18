import numpy as np
import cv2
import math

class EVM_motion:
    def __init__(self,alpha,lambda_c,r1,r2,chromAttenuation,w,h):
        self.lambda_c = lambda_c
        self.r1 = r1
        self.r2 = r2
        self.chromAttenuation = chromAttenuation
        self.alpha = alpha
        self.filt1 = np.array([[0.088388347648318,0.353553390593274,0.530330085889911,0.353553390593274,0.088388347648318]]).transpose()

        self.T = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647], [1.0, -1.106, 1.703]])
        self.T_inv = np.linalg.inv(np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647], [1.0, -1.106, 1.703]])).transpose()

        min_wh = min(w,h)
        self.max_ht = int(np.floor(np.log2(min_wh))-2)

        self.NTFSFrameList = []
        self.lowpass1 = []
        self.lowpass2 = []

        self.exaggeration_factor = 2
        self.delta = self.lambda_c / 8.0 / (1 + self.alpha)
        self.lamda = math.sqrt((w ** 2 + h ** 2)) / 3.0
    def RGB2NTSC(self, frame):
        rgb_frame = frame[:, :, ::-1]
        reshape_size = rgb_frame.shape[0] * rgb_frame.shape[1]
        reshape_frame = np.reshape(rgb_frame, (reshape_size, 3)) / 255.0
        NTFSFrame = np.dot(reshape_frame, self.T_inv)
        NTFSFrame = np.reshape(NTFSFrame, frame.shape)
        return NTFSFrame

    def NTSC2BGR(self,NTSC):
        reshape_size = NTSC.shape[0] * NTSC.shape[1]
        reshape_frame = np.reshape(NTSC, (reshape_size, 3))
        RGB_img = np.dot(reshape_frame, self.T.transpose())
        RGB_img = np.maximum(RGB_img, 0)
        pos = np.where(RGB_img > 1)
        for r in pos[0]:
            RGB_img[r, :] /= max(RGB_img[r, :])

        matRGB = (np.reshape(RGB_img, NTSC.shape) * 255.0).astype(np.uint8)
        matBGR = cv2.cvtColor(matRGB, cv2.COLOR_RGB2BGR)

        return matBGR

    def Laplas_pry(self,img):
        gauss_list = []
        laplas_list = []
        gauss_list.append(img)
        for _ in range(self.max_ht):
            # 先对列进行空间滤波
            expand_img = cv2.copyMakeBorder(img, 0, 0, 2,2, cv2.BORDER_REFLECT_101)
            filter_img = cv2.filter2D(expand_img, -1, self.filt1.transpose())
            img = filter_img[:, 2:filter_img.shape[1] - 2:2, :]

            expand_img = cv2.copyMakeBorder(img, 2, 2, 0,0, cv2.BORDER_REFLECT_101)
            filter_img = cv2.filter2D(expand_img, -1, self.filt1)
            img = filter_img[2:filter_img.shape[0] - 2:2, :, :]
            gauss_list.append(img)
        laplas_list.append(gauss_list[-1])
        for i in range(len(gauss_list)-1,1,-1):
            cur_img = gauss_list[i]
            next_img = gauss_list[i-1]
            next_zeros = np.zeros((next_img.shape[0],cur_img.shape[1],cur_img.shape[2]))
            next_zeros[0:next_zeros.shape[0]:2,:,:] = cur_img
            expand_img = cv2.copyMakeBorder(next_zeros, 2, 2, 0, 0, cv2.BORDER_REFLECT_101)
            filter_y = cv2.filter2D(expand_img,-1,self.filt1)
            img = filter_y[2:filter_y.shape[0] - 2, :, :]

            next_zeros = np.zeros((next_img.shape[0], next_img.shape[1], cur_img.shape[2]))
            next_zeros[:, 0:next_zeros.shape[1]:2, :] = img
            expand_img = cv2.copyMakeBorder(next_zeros, 0, 0, 2, 2, cv2.BORDER_REFLECT_101)
            filter_x = cv2.filter2D(expand_img, -1, self.filt1.transpose())
            img = filter_x[:,2:filter_x.shape[1]-2,:]

            hi2 = next_img - img

            laplas_list.append(hi2)
        laplas_list.append(gauss_list[0])
        return laplas_list


    def magnification_motion(self,frame):
        NTFSFrame = self.RGB2NTSC(frame)
        laplas_list = self.Laplas_pry(NTFSFrame)

        if len(self.lowpass1) == 0:
            self.lowpass1 = laplas_list
            self.lowpass2 = laplas_list
            BGRImg = self.NTSC2BGR(NTFSFrame)
        else:
            filtered = self.IIRFilter(laplas_list)
            mag_recover_img = self.magn_recover(filtered)
            BGRImg = self.NTSC2BGR(mag_recover_img)


        return BGRImg

    def IIRFilter(self,laplas_list):
        self.lowpass1 = [(1 - self.r1) * sub_lowpass1 + self.r1 * sub_laplas_list for sub_lowpass1, sub_laplas_list in
                         zip(self.lowpass1[:-1], laplas_list[:-1])]
        self.lowpass2 = [(1 - self.r2) * sub_lowpass2 + self.r2 * sub_laplas_list for sub_lowpass2, sub_laplas_list in
                         zip(self.lowpass2[:-1], laplas_list[:-1])]
        filtered = [tmp1 - tmp2 for tmp1, tmp2 in zip(self.lowpass1, self.lowpass2)]
        filtered.append(laplas_list[-1])
        self.lowpass1.append(laplas_list[-1])
        self.lowpass2.append(laplas_list[-1])
        return filtered


    def magn_recover(self,filtered):
        for i in range(1,len(filtered)):
            currAlpha = (self.lamda * (2 ** (-i)) / self.delta / 8 - 1) * self.exaggeration_factor
            if i < (len(filtered)-1):
                if currAlpha > self.alpha:
                    filtered[i] *= self.alpha
                else:
                    filtered[i] *= currAlpha

                if i == 1:
                    pre_upimg = np.copy(filtered[i])
                else:
                    pre_upimg += filtered[i]

                next_filter_size = filtered[i+1].shape
                pre_upimg = self.upsize(pre_upimg,next_filter_size)
            else:
                pre_upimg[:,:,1] *= self.chromAttenuation
                pre_upimg[:,:,2] *= self.chromAttenuation
                final_mag_img = filtered[-1] + pre_upimg

        return final_mag_img

    def upsize(self,filtered,next_filter_size):
        tmp = np.zeros((next_filter_size[0], filtered.shape[1], filtered.shape[2]))
        tmp[0:tmp.shape[0]:2, 0:tmp.shape[1], :] = filtered
        expand_img = cv2.copyMakeBorder(tmp, 2, 2, 0, 0, cv2.BORDER_REFLECT_101)
        filter_y = cv2.filter2D(expand_img, -1, self.filt1)
        img = filter_y[2:filter_y.shape[0] - 2, :, :]

        next_zeros = np.zeros((next_filter_size))
        next_zeros[:, 0:next_zeros.shape[1]:2, :] = img
        expand_img = cv2.copyMakeBorder(next_zeros, 0, 0, 2, 2, cv2.BORDER_REFLECT_101)
        filter_x = cv2.filter2D(expand_img, -1, self.filt1.transpose())
        img = filter_x[:, 2:filter_x.shape[1] - 2, :]

        return img
