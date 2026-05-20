"""
rf_habitat.py - 随机森林大熊猫生境评价模型

功能：
1. 矢量数据转栅格（训练标签生成）
2. 多因子GeoTIFF数据提取为训练文本
3. 随机森林模型训练与评估
4. 全研究区生境敏感性预测与GeoTIFF输出

评价因子（11个，7大类）：
- 地质灾害敏感性（CNN滑坡易发性结果）
- 地形敏感性（高程、坡度、坡向、曲率）
- 气候敏感性（温度、降水）
- 植被敏感性（NDVI、植被种类）
- 动物痕迹敏感性（大熊猫足迹可达性）
- 水资源敏感性（河流距离）
- 人类活动敏感性（道路距离、居民点距离）

敏感性分级：
1 = 不敏感, 2 = 微敏感, 3 = 中敏感, 4 = 高敏感, 5 = 极敏感
"""

import numpy as np
import os
import pickle
from osgeo import gdal, ogr
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, cohen_kappa_score


# ==================== 数据准备 ====================

def vector_to_raster(vector_file, raster_template, output_raster):
    """
    矢量数据转栅格

    将训练样本点的矢量文件（.shp）转为与基准栅格对齐的GeoTIFF
    """
    vector_ds = ogr.Open(vector_file)
    layer = vector_ds.GetLayer()

    raster_ds = gdal.Open(raster_template)
    geotransform = raster_ds.GetGeoTransform()
    projection = raster_ds.GetProjection()
    cols = raster_ds.RasterXSize
    rows = raster_ds.RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.Create(output_raster, cols, rows, 1, gdal.GDT_Float32)
    output_ds.SetGeoTransform(geotransform)
    output_ds.SetProjection(projection)
    band = output_ds.GetRasterBand(1)
    band.SetNoDataValue(-9999)

    gdal.RasterizeLayer(output_ds, [1], layer, options=["ATTRIBUTE=buffer_ked"])
    print("栅格标签创建成功！")

    vector_ds = None
    output_ds = None
    raster_ds = None


def create_training_data(tif_path, label_path, txt_path):
    """
    从多波段GeoTIFF和标签栅格中提取训练数据

    输出格式：每行 = 11个因子值 + 标签值（逗号分隔的txt文件）
    """
    dataset = gdal.Open(tif_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    tif_data = dataset.ReadAsArray(0, 0, width, height)

    dataset = gdal.Open(label_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    label_data = dataset.ReadAsArray(0, 0, width, height)

    min_value = np.min(label_data)
    max_value = np.max(label_data)

    # 归一化标签（仅对有效值）
    normalized_data = (label_data - 1.0) / (max_value - 1.0)

    if os.path.exists(txt_path):
        os.remove(txt_path)

    with open(txt_path, 'w') as f:
        for i in range(normalized_data.shape[0]):
            for j in range(normalized_data.shape[1] - 1):
                if normalized_data[i][j] >= 0:
                    var = ""
                    for k in range(tif_data.shape[0]):
                        var += str(tif_data[k][i][j]) + ","
                    var += str(normalized_data[i][j])
                    f.write(var + '\n')

    print("训练数据创建成功！")


# ==================== 模型训练 ====================

def train(txt_path, model_save_path):
    """
    随机森林模型训练

    参数：
        txt_path: 训练数据文件路径
        model_save_path: 模型保存路径（pickle格式）
    """
    data = np.loadtxt(txt_path, dtype=float, delimiter=',')
    train_data, train_label = np.split(data, indices_or_sections=(11,), axis=1)
    train_data = train_data[:, 0:11]

    # 划分训练集和测试集（9:1）
    train_x, test_x, train_y, test_y = model_selection.train_test_split(
        train_data, train_label,
        random_state=6, train_size=0.9, test_size=0.1
    )

    # 随机森林回归（500棵树）
    classifier = RandomForestRegressor(
        n_estimators=500,
        bootstrap=True,
        max_features='sqrt'
    )

    print(f"训练数据: {train_x.shape}")
    print(f"训练标签: {train_y.ravel().shape}")

    classifier.fit(train_x, train_y.ravel())

    print(f"训练集准确率: {classifier.score(train_x, train_y):.4f}")
    print(f"测试集准确率: {classifier.score(test_x, test_y):.4f}")

    with open(model_save_path, "wb") as f:
        pickle.dump(classifier, f)

    print(f"模型已保存至: {model_save_path}")


# ==================== 预测与评估 ====================

def calculate_accuracy(testlabel_path, classify_matrix):
    """计算分类精度（混淆矩阵 + Kappa系数）"""
    data = gdal.Open(testlabel_path)
    values = data.GetRasterBand(1).ReadAsArray()

    confusion_mat = confusion_matrix(
        values[values != 0], classify_matrix[values != 0]
    )
    kappa = cohen_kappa_score(
        values[values != 0], classify_matrix[values != 0]
    )
    overall_accuracy = np.sum(np.diagonal(confusion_mat)) / np.sum(confusion_mat)

    print(f"混淆矩阵:\n{confusion_mat}")
    print(f"分类总体精度: {overall_accuracy:.4f}")
    print(f"Kappa系数: {kappa:.4f}")


def predict(tif_path, rf_path, save_tif_path):
    """
    全研究区生境敏感性预测

    加载训练好的随机森林模型，对11波段GeoTIFF逐像素预测，
    输出生境敏感性GeoTIFF
    """
    dataset = gdal.Open(tif_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    geotrans = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    tif_data = dataset.ReadAsArray(0, 0, width, height)

    with open(rf_path, "rb") as f:
        rf_model = pickle.load(f)

    # 多波段数据展平为 [像素数, 波段数]
    data = np.zeros((tif_data.shape[0], tif_data.shape[1] * tif_data.shape[2]))
    for i in range(tif_data.shape[0]):
        data[i] = tif_data[i].flatten()
    data = data.swapaxes(0, 1)

    # 去除背景像素
    non_background = data[~np.all(data == 0, axis=1)]
    pre = rf_model.predict(non_background)

    # 还原为二维栅格
    output = np.zeros(tif_data.shape[1] * tif_data.shape[2])
    output[~np.all(data == 0, axis=1)] = pre
    output = output.reshape(tif_data.shape[1], tif_data.shape[2])

    print("分类结束，正在输出结果...")
    write_tif(output, geotrans, proj, save_tif_path)


def write_tif(im_data, im_geotrans, im_proj, path):
    """将numpy数组写入GeoTIFF"""
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if dataset is not None:
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset
    print(f"结果已保存至: {path}")


# ==================== 主程序 ====================

if __name__ == "__main__":
    # 1. 矢量转栅格（生成训练标签）
    # vector_to_raster(r".\RF-data\point-train.shp",
    #                  r".\cnn_data\dem.tif",
    #                  r".\RF-data\label_train.tif")

    # 2. 提取训练数据
    # create_training_data(r".\RF-data\layerstack-11.tif",
    #                      r".\RF-data\label_train.tif",
    #                      r".\RF-data\train_data.txt")

    # 3. 训练随机森林
    txt_path = r".\RF-data\train_data.txt"
    rf_save_path = r"model_RF.pickle"
    train(txt_path, rf_save_path)

    # 4. 预测
    # predict(r".\RF-data\layerstack-11.tif", rf_save_path, "habitat_result.tif")
