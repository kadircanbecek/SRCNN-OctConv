import numpy as np
import cv2
import os, os.path
import math
from tensorflow.keras import backend as K
from sklearn.feature_extraction.image import extract_patches


def bicubic_psnr(input):
    def psnr(y_true, y_pred):
        max_pixel = 255.0
        return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(input - y_true), axis=-1)))) / 2.303

    return psnr


def psnr(y_true, y_pred):
    max_pixel = 255.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303


def read_data(path_data):
    image_path_list_hr = get_filenames(path_data)
    data_raw = []

    # loop through image_path_list to open each image
    for imagePath in image_path_list_hr:
        image = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
        data_raw.append(image)
        if image is None:
            print("Error loading: " + imagePath)
            # end this loop iteration and move on to next image
            continue
    return data_raw


def read_data(path_list):
    data_raw = []

    # loop through image_path_list to open each image
    for imagePath in path_list:
        image = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
        data_raw.append(image)
        if image is None:
            print("Error loading: " + imagePath)
            # end this loop iteration and move on to next image
            continue
    return data_raw


def get_filenames(path_data):
    image_path_list_hr = []

    valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]  # specify your valid extensions here
    valid_image_extensions = [item.lower() for item in valid_image_extensions]

    for file in os.listdir(path_data):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue
        image_path_list_hr.append(os.path.join(path_data, file))
    image_path_list_hr.sort()
    return image_path_list_hr


def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return np.uint8(ycbcr)


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


def sample(image, sample_size, stride):
    '''
    :param image: a 2d image with (height, width, channels)
    :param sample_size: sample size for output sample (sample_size, sample_size, channels)
    :param stride: stride for cropping (stride, stride , channel)
    :return: sampled_image with size (sample_count,sample_size,sample_size,channels)
    '''
    patch_size = (sample_size, sample_size, image.shape[-1])
    stride_size = (stride, stride, image.shape[-1])
    sampled_image = extract_patches(image, patch_size, stride_size).reshape([-1] + list(patch_size))
    return sampled_image


def preprocessing(image, train=True, sample_size=32, stride=14):
    '''

    :param image: a 2d image with (height, width, channels)
    :param train: is to be trained or not
    :param sample_size: sample size for output sample (sample_size, sample_size, channels)
    :param stride: stride for cropping (stride, stride , channel)
    :return:
    '''
    data = rgb2ycbcr(image)
    if train:
        data = sample(np.array(data), sample_size, stride)
    return data


def psnr_for_loss(loss):
    return 10 * math.log10(255 ** 2 / loss)
