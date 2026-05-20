"""
read_data.py - GeoTIFF数据读取与CNN数据集构建

功能：
1. 读取多个影响因子的GeoTIFF文件，构建多通道张量
2. 将每个像素扩展为 n×n 的图像块（滑动窗口），用于CNN输入
3. 划分训练集/测试集
4. 提供数据保存与导出功能

影响因子（7个）：
DEM（高程）、断层距离、坡度、NDVI、道路距离、河流距离、坡向
"""

import warnings
import numpy as np
from numpy.core.numeric import ones_like
import pandas as pd
import openpyxl
import random
from osgeo import gdal, gdalconst, ogr
from PIL import Image
import os


config = {
    # 各因子GeoTIFF路径
    "data_path": [
        r".\cnn_data\dem.tif",
        r".\cnn_data\fault.tif",
        r".\cnn_data\slope.tif",
        r".\cnn_data\ndvi.tif",
        r".\cnn_data\road.tif",
        r".\cnn_data\river.tif",
        r".\cnn_data\aspect.tif"
    ],

    "data_max": [9, 9, 9, 9, 9, 9, 9, 9],  # 各因子归一化最大值
    "label_path": r".\label_data\label1.tif",

    "feature": 7,       # 影响因子数量
    "width": 2118,       # 研究区栅格宽度（像素）
    "height": 1817,      # 研究区栅格高度（像素）
    "size": 11,          # 像素窗口大小（11×11）
    "batch_size": 32,    # 训练批次大小
    "epochs": 80,        # 训练轮次
    "model_parameter": "model_batchsize32_size_17"
}


def resample_tif(img):
    """若GeoTIFF尺寸不一致，进行重采样"""
    warnings.filterwarnings("ignore")
    img = np.array(Image.fromarray(img).resize((config["width"], config["height"])))
    return img


def read_data_from_tif(tif_path):
    """读取单个GeoTIFF文件为numpy数组"""
    tif = gdal.Open(tif_path)
    w, h = tif.RasterXSize, tif.RasterYSize
    img = np.array(tif.ReadAsArray(0, 0, w, h).astype(np.float32))
    if w != config["width"] and h != config["height"]:
        imgs = resample_tif(img)
    return img


def get_feature_data():
    """
    读取所有影响因子，构建 [feature, width, height] 的归一化张量
    """
    tif_paths = config["data_path"]
    data = np.zeros((config["feature"], config["width"], config["height"])).astype(np.float32)
    for i, tif_path in enumerate(tif_paths):
        img = read_data_from_tif(tif_path)
        data[i, :, :] = (img - 0) / config["data_max"][i]  # Min-Max归一化
    return data


def save_to_excel(numpy_data, file):
    """将numpy数组保存为Excel文件"""
    data = np.squeeze(np.array(numpy_data))
    data = pd.DataFrame(data)
    writer = pd.ExcelWriter(file)
    data.to_excel(writer, 'page_1', float_format='%.8f')
    writer.save()
    writer.close()
    return 0


class CreateDataset():
    """
    像素扩展数据集构建器

    将每个像素扩展为 n×n 大小的图像块：
    以目标像素为中心，提取周围 n×n 区域作为CNN输入

    参数：
        tensor_data: [feature, width, height] 的多通道张量
        n: 窗口大小（奇数），如 11 表示 11×11 的窗口
    """
    def __init__(self, tensor_data, n):
        self.data = tensor_data
        self.n = int(n)
        self.p = int((n - 1) / 2)
        self.F = tensor_data.shape[0]
        self.w = tensor_data.shape[1]
        self.h = tensor_data.shape[2]

    def create_new_tensor(self):
        """边缘填充，保证所有像素都能提取完整的 n×n 窗口"""
        new_tensor = np.zeros((self.F, self.w + self.n - 1, self.h + self.n - 1))
        new_tensor[:, self.p:self.w + self.p, self.p:self.h + self.p] = self.data
        return new_tensor

    def pixel_to_image(self, data):
        """将每个像素提取为 [feature, n, n] 的图像块"""
        images = []
        for i in range(config["width"]):
            for j in range(config["height"]):
                images.append(data[:, i:i + self.n, j:j + self.n])
        return images


def read_label():
    """
    读取标签GeoTIFF，返回展平的标签列表
    标签编码：0=背景, 1=训练集非滑坡, 2=训练集滑坡, 3=测试集非滑坡, 4=测试集滑坡
    """
    label = read_data_from_tif(config["label_path"])
    labels = []
    count_0, count_1, count_2, count_3, count_4 = 0, 0, 0, 0, 0
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i, j] == 0:
                labels.append(0)
                count_0 += 1
            elif label[i, j] == 1:
                labels.append(1)
                count_1 += 1
            elif label[i, j] == 2:
                labels.append(2)
                count_2 += 1
            elif label[i, j] == 3:
                labels.append(3)
                count_3 += 1
            elif label[i, j] == 4:
                labels.append(4)
                count_4 += 1
            else:
                labels.append(-1)
    print("标签分布 — 背景:{}, 训练非滑坡:{}, 训练滑坡:{}, 测试非滑坡:{}, 测试滑坡:{}".format(
        count_0, count_1, count_2, count_3, count_4))
    return labels


def split_train_test(images, labels, mode="train"):
    """
    按标签划分训练集和测试集
    0,1 → 训练集（非滑坡/滑坡）
    2,3 → 测试集（非滑坡/滑坡）
    """
    train_images, train_labels = [], []
    valid_images, valid_labels = [], []
    for i in range(len(labels)):
        if labels[i] == 0 or labels[i] == 1:
            train_images.append(images[i][:, :, :])
            train_labels.append(labels[i])
        elif labels[i] == 2 or labels[i] == 3:
            valid_images.append(images[i][:, :, :])
            valid_labels.append(labels[i] - 2)

    if mode == "train":
        print(len(train_images), len(train_labels))
        return train_images, train_labels
    else:
        print(len(valid_images), len(valid_labels))
        return valid_images, valid_labels


def shuffle_data(images, labels):
    """同步随机打乱图像和标签"""
    randnum = random.randint(0, len(images))
    random.seed(randnum)
    random.shuffle(images)
    random.seed(randnum)
    random.shuffle(labels)
    return images, labels


def train_data():
    """构建训练数据集"""
    tensor_data = get_feature_data()
    creator = CreateDataset(tensor_data, config["size"])
    data = creator.create_new_tensor()
    images = creator.pixel_to_image(data)
    labels = read_label()

    train_images, train_labels = split_train_test(images, labels, mode="train")
    train_images, train_labels = shuffle_data(train_images, train_labels)

    return (np.array(train_images).reshape((-1, config["feature"], config["size"], config["size"])),
            np.array(train_labels).reshape((-1, 1)))


def test_data():
    """构建测试数据集"""
    tensor_data = get_feature_data()
    creator = CreateDataset(tensor_data, config["size"])
    data = creator.create_new_tensor()
    images = creator.pixel_to_image(data)
    labels = read_label()

    images, labels = split_train_test(images, labels, mode="valid")

    return (np.array(images).reshape((-1, config["feature"], config["size"], config["size"])),
            np.array(labels).reshape((-1, 1)))


def pred_data():
    """构建全研究区预测数据集"""
    tensor_data = get_feature_data()
    creator = CreateDataset(tensor_data, config["size"])
    data = creator.create_new_tensor()
    data = creator.pixel_to_image(data)
    return data


def save_to_tif(pred_result, save_path):
    """将预测结果保存为GeoTIFF"""
    img = pred_result.reshape((config["width"], config["height"]))
    im_geotrans, im_prof = [], []

    for tif_path in config["data_path"]:
        tif = gdal.Open(tif_path)
        im_geotrans.append(tif.GetGeoTransform())
        im_prof.append(tif.GetProjection())

    if 'int8' in img.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in img.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(img.shape) == 3:
        im_bands, im_height, im_width = img.shape
    else:
        im_bands, (im_height, im_width) = 1, img.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(save_path, im_width, im_height, im_bands, datatype)
    dataset.SetGeoTransform(im_geotrans[-1])
    dataset.SetProjection(im_prof[-1])
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(img)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(img[i])
    del dataset
    print('保存成功')


if __name__ == "__main__":
    test_data()
