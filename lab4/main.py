import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from bmp_utils import BMPReader

class Img:
    def __init__(self, img_dir):
        self.img = cv2.imread(img_dir)
        self.gray_img = cv2.split(self.img)[0]

    def pepperandSalt(self, percetage):
        '''
        在灰度图的基础上产生椒盐噪声
        percentage:产生噪声点的占比
        '''
        src = self.gray_img.copy()
        NoiseImg = src
        NoiseNum = int(percetage*src.shape[0]*src.shape[1])
        for i in range(NoiseNum):
            randX = random.randint(0, src.shape[0]-1)
            randY = random.randint(0, src.shape[1]-1)
            if random.randint(0, 1) <= 0.5:
                NoiseImg[randX, randY] = 0
            else:
                NoiseImg[randX, randY] = 255
        return NoiseImg
    
    def gaussianNoise(self, mean=0, var=0.001):
        ''' 
            添加高斯噪声
            mean : 均值 
            var : 方差
        '''
        image = self.gray_img.copy()
        image = np.array(image/255, dtype=float)
        # 拟合正态函数的分布 
        # image.shape 指示输出的大小
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out*255)
        #cv.imshow("gasuss", out)
        return out

    def gray_median_filter(self, img=None, pad_size=1):
        """
        对于单通道的图像(灰度图)进行中值滤波
        :param self.gray_img: - 灰度图
        :param pad_size: - 执行滤波前在图像周围补0的维数，同时指示滤波器大小(2*pad_size + 1)
        :return: 返回滤波后的图像，大小与原图一致
        """
        img = self.gray_img if img is None else img
        img = np.pad(img.copy(), pad_size, pad_with, padder=0)
        img_filter = img.copy()
        median = int((2 * pad_size + 1) ** 2 / 2)
        h, w = np.shape(img)
        h_low, h_high = pad_size, h - pad_size
        w_low, w_high = pad_size, w - pad_size
        for i in range(h_low, h_high):
            for j in range(w_low, w_high):
                i_low, i_high = i - pad_size, i + pad_size + 1
                j_low, j_high = j - pad_size, j + pad_size + 1
                img_filter[i, j] = np.median(
                    img[i_low:i_high, j_low:j_high].copy().reshape((1, -1))[0])  # [median]
        img_filter = img_filter[h_low:h_high, w_low:w_high]
        return img_filter

    def gray_mean_filter(self, filter, pad_size=1, img=None):
        """
        对于单通道的图像(灰度图)进行均值滤波
        :param self.gray_img: - 灰度图
        :param filter: - 滤波器，n维方阵
        :param pad_size: - 执行滤波前在图像周围补0的维数
        :return: 返回滤波后的图像，大小与原图一致
        """
        img = self.gray_img if img is None else img
        img = np.pad(img.copy(), pad_size, pad_with, padder=0)
        img_filter = img.copy()
        h, w = np.shape(img)
        h_low, h_high = pad_size, h - pad_size
        w_low, w_high = pad_size, w - pad_size
        for i in range(h_low, h_high):
            for j in range(w_low, w_high):
                i_low, i_high = i - pad_size, i + pad_size + 1
                j_low, j_high = j - pad_size, j + pad_size + 1
                # 使用相关求取均值
                img_filter[i, j] = np.sum(
                    img[i_low:i_high, j_low:j_high] * filter)
        img_filter = img_filter[h_low:h_high, w_low:w_high]

        return img_filter

    def gray_roberts_sharp(self, throd=0, norm=1, img=None):
        """
        对于单通道的图像(灰度图)使用Roberts算子进行锐化
        :param self.gray_img: - 灰度图
        :param throd: - 锐化门限，若执行完Roberts计算后小于throd则置0
        :param norm: - 计算使用的范数类型，默认L1范数
        :return: 返回锐化后的图像，大小小于原图像
        """
        img = self.gray_img if img is None else img
        img = img.copy().astype(np.int32)
        h, w = np.shape(img)
        img_sharp = np.zeros((h - 1, w - 1))

        for i in range(h - 1):
            for j in range(w - 1):
                array = [img[i, j] - img[i + 1, j + 1],
                         img[i, j + 1] - img[i + 1, j]]
                img_sharp[i, j] = np.linalg.norm(array, ord=norm)

        img_sharp = np.where(img_sharp > throd, img_sharp, 0).astype(np.uint8)
        return img_sharp

    def gray_sobel_sharp(self, throd=0, norm=1, img=None):
        """
        对于单通道的图像(灰度图)使用Sobel算子进行锐化
        :param self.gray_img: - 灰度图
        :param throd: - 锐化门限，若执行完Sobel计算后小于throd则置0
        :param norm: - 计算使用的范数类型，默认L1范数
        :return: 返回锐化后的图像，大小小于原图像
        """
        img = self.gray_img if img is None else img
        img = img.copy().astype(np.int32)
        h, w = np.shape(img)
        img_sharp = np.zeros((h - 2, w - 2))

        # 水平和垂直的Sobel算子
        r_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        c_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                array = [np.sum(img[i - 1:i + 2, j - 1:j + 2] * r_filter),
                         np.sum(img[i - 1:i + 2, j - 1:j + 2] * c_filter)]
                img_sharp[i - 1, j - 1] = np.linalg.norm(array, ord=norm)
        img_sharp = np.where(img_sharp > throd, img_sharp, 0).astype(np.unit8)
        return img_sharp

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


if __name__ == '__main__':
    img_path = 'res/luna.bmp'
    file_name = img_path.split('/')[-1]
    prefix = file_name.split('.')[0]
    
    # bmp = BMPReader(img_path)
    
    # cv2.imshow("bmpreader")
    
    img = Img(img_path)

    img_gray = img.gray_img
    cv2.imwrite(prefix + '_gray.bmp', img_gray)

    img_gaussian_noise = img.gaussianNoise()
    cv2.imwrite(prefix + '_gaussian_noise.bmp', img_gaussian_noise)

    img_pepperand_salt_noise = img.pepperandSalt(0.01)
    cv2.imwrite(prefix + '_pepperand_salt_noise.bmp', img_pepperand_salt_noise)

    start = time.time()
    img_median_filter= img.gray_median_filter(pad_size=1, img=img_pepperand_salt_noise)
    cv2.imwrite(prefix + '_median_filter.bmp', img_median_filter)
    print('中值滤波耗时 ', time.time() - start, 's')

    filter = np.array([[1, 1, 1],
       [1, 1, 1],
       [1, 1, 1]]) / 9
    start = time.time()
    img_mean_filter= img.gray_mean_filter(filter,pad_size=1, img=img_gaussian_noise)
    cv2.imwrite(prefix + '_mean_filter.bmp', img_mean_filter)
    print('均值滤波耗时 ', time.time() - start, 's')

    img_roberts_sharp = img.gray_roberts_sharp(norm=1)
    img_roberts_sharp = img.gray_roberts_sharp(norm=1)
    cv2.imwrite(prefix + '_roberts_sharp_b.bmp', img_roberts_sharp)

    img_sobel_sharp = img.gray_sobel_sharp(norm=1)
    cv2.imwrite(prefix + '_sobel_sharp.bmp', img_sobel_sharp)
