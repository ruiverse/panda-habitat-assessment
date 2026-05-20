"""
create_label.py - 滑坡/非滑坡标签数据生成

功能：根据滑坡点和非滑坡点的Excel坐标数据，生成标签GeoTIFF文件
标签编码：
    0: 背景（未标注区域）
    1: 训练集非滑坡点
    2: 训练集滑坡点
    3: 测试集非滑坡点
    4: 测试集滑坡点
"""

import numpy as np
from osgeo import gdal, osr
import pandas as pd


config = {
    "source_data": r".\cnn_data\dem.tif",
    "label_path": r".\label_data\label1.tif",
    "LS_train": r".\label_data\08landslide_train.xlsx",
    "Non_LS_train": r".\label_data\08nolandslide_train.xlsx",
    "LS_test": r".\label_data\08landslide_test.xlsx",
    "Non_LS_test": r".\label_data\08nolandslide_test.xlsx"
}


def get_sample_info(file):
    """读取Excel中的滑坡/非滑坡样本点坐标"""
    data = pd.read_excel(file)
    data = data._values
    info = data[:, 0:2]
    print(info.shape)
    return info


def get_tif_info():
    """获取基准GeoTIFF的坐标范围和尺寸"""
    data = gdal.Open(config["source_data"])
    w, h = data.RasterXSize, data.RasterYSize
    print(w, h)

    info = data.GetGeoTransform()
    x = info[0] + np.arange(w, dtype=float) * info[1]
    y = info[3] + np.arange(h, dtype=float) * info[5]

    print(x[0], y[0])
    print(x[-1], y[-1])
    return x, y, w, h


def find_ii_jj(x, y, xx, yy):
    """根据地理坐标查找对应的像素行列号"""
    ii, jj = 0, 0
    for j in range(len(x) - 1):
        if xx >= x[j] and xx < x[j + 1]:
            jj = j
            break
    for i in range(len(y) - 1):
        if yy <= y[i] and yy > y[i + 1]:
            ii = i
            break
    return ii, jj


def normalization(image):
    """标签值重映射"""
    result = np.select(
        [image == 1, image == 2, image == 3, image == 4, image == 0],
        [0.0, 1.0, 2.0, 3.0, 4.0],
        default=-100.0
    )
    return result


def save_to_tif(pred_result, save_path):
    """将numpy数组保存为GeoTIFF（保留原始坐标系和投影）"""
    img = pred_result
    tif = gdal.Open(config["source_data"])
    im_geotrans = tif.GetGeoTransform()
    im_prof = tif.GetProjection()

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
    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_prof)
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(img)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(img[i])
    del dataset
    print('标签文件保存成功')


def create_label():
    """
    主函数：生成标签GeoTIFF
    x 是经度（代表列），y 是纬度（代表行）
    """
    x, y, w, h = get_tif_info()
    label = np.zeros((h, w))

    LS_tra = get_sample_info(config["LS_train"])
    Non_LS_tra = get_sample_info(config["Non_LS_train"])
    LS_tes = get_sample_info(config["LS_test"])
    Non_LS_tes = get_sample_info(config["Non_LS_test"])

    for i in range(len(Non_LS_tra)):
        ii, jj = find_ii_jj(x, y, Non_LS_tra[i, 0], Non_LS_tra[i, 1])
        label[ii, jj] = 1

    for i in range(len(LS_tra)):
        ii, jj = find_ii_jj(x, y, LS_tra[i, 0], LS_tra[i, 1])
        label[ii, jj] = 2

    for i in range(len(Non_LS_tes)):
        ii, jj = find_ii_jj(x, y, Non_LS_tes[i, 0], Non_LS_tes[i, 1])
        label[ii, jj] = 3

    for i in range(len(LS_tes)):
        ii, jj = find_ii_jj(x, y, LS_tes[i, 0], LS_tes[i, 1])
        label[ii, jj] = 4

    data = normalization(label)
    save_to_tif(data, config["label_path"])


if __name__ == "__main__":
    create_label()
