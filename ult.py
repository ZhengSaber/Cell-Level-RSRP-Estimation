import os, time, pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import warnings
warnings.filterwarnings("ignore")


def get_filename(root_dir, debug=False):
    filenames = []
    sample_cnt = 0
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for name in files:
            sample_cnt += 1
            # print("files: ",os.path.join(root, name),name)
            file_name = name
            file_content = os.path.join(root, name)
            filenames.append([file_name,file_content])
            # if debug:
            #     if sample_cnt == 1:
            #         break
    return filenames

def pre_processing(pb_data):
    x = pb_data['X'] - pb_data['Cell X']
    y = pb_data['Y'] - pb_data['Cell Y']

    x = x / 5
    y = y / 5
    A = np.radians(pb_data['Azimuth'])
    x_ = x * np.cos(A) - y * np.sin(A)
    y_ = x * np.sin(A) + y * np.cos(A)
    pb_data['x_'] = x_
    pb_data['y_'] = y_
    pb_data = pb_data.iloc[list(pb_data["x_"] > -32)]
    pb_data = pb_data.iloc[list(pb_data["x_"] < 32)]
    pb_data = pb_data.iloc[list(pb_data["y_"] > 0)]
    pb_data = pb_data.iloc[list(pb_data["y_"] < 64)]
    return pb_data

def RMSE(A,B):
    return np.sqrt(np.nanmean(np.power((A - B), 2)))
def MAPE(A,B):
    return np.nanmean(np.abs((A - B)/B)) * 100
    # return np.nanmean((np.abs(A - B))) * 100


def read_dataset(path, name):
    dataset = np.load(path + name)
    # print(np.shape(dataset))
    dataset = np.transpose(np.array(dataset),(0,2,3,1))
    print(np.shape(dataset))
    return dataset

def maxmin_norm_mask(data, mask):
    data = (data-data[mask.astype('bool')].min()) \
                 / (data[mask.astype('bool')].max() - data[mask.astype('bool')].min())

    # data = (data-0.5)*2
    data = data * mask

    return data

def RSRP_maxmin_norm(data, ):
    data = (data-data.min())/(data.max() - data.min())
    data = 2*data-1
    # data = (data - (-110)) / 40
    return data

def maxmin_norm(data):
    data = (data-data.min())/(data.max() - data.min())
    # data = 2 * data - 1
    # data = (data - (-110)) / 40
    return data

def MSE(A,B):
    err = np.mean((A-B)**2)
    return err


from scipy.signal import convolve2d
def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))


def numpy_to_png_dype(arr):
    arr = maxmin_norm(arr) * 255
    return arr

import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
def calculate_fid(act1, act2):
# calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
# calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)*2.0)
# calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
# check and correct imaginary numbers from sqrtif iscomplexobj(covmean):
    covmean = covmean.real
# calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# act1 = random(10*2048)
# act1 = act1.reshape((10,2048))
# act2 = random(10*2048)
# act2 = act2.reshape((10,2048))
# fid = calculate_fid(act1, act1)
# print('FID (same): %.3f' % fid)
# fid = calculate_fid(act1, act2)
# print('FID (different): %.3f' % fid)

import math
def psnr(img1, img2):
   mse = np.nanmean((img1/1.0 - img2/1.0) ** 2 )
   if mse < 1.0e-10:
      return 100
   return 10 * math.log10(255.0**2/mse)
