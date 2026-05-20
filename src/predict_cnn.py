"""
predict_cnn.py - SE-CNN 滑坡易发性预测与评估

功能：
1. 在测试集上评估模型性能（AUC/ACC/Kappa/F1等）
2. 对全研究区逐像素预测滑坡概率
3. 将预测结果保存为GeoTIFF格式的滑坡易发性图
"""

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.autograd import Variable

from read_data import config, test_data, train_data, pred_data, save_to_tif, save_to_excel
from model import LandslideCNN
import evaluate as evaluate_method


def calculate_correct(outputs, targets):
    """单样本预测正确判断"""
    outputs = outputs.cpu().detach().numpy()
    predict = 0 if outputs[0, 0] >= outputs[0, 1] else 1
    return 1 if predict == targets[0] else 0


def eval_model(y_pred, test_y_1D, save_roc_path='roc_curve.csv'):
    """完整模型评估"""
    probability = [prob[1] for prob in y_pred]
    print(f'测试样本数: {len(probability)}')
    evaluate_method.full_evaluation(test_y_1D, probability, save_roc_path)


def test():
    """在测试集上评估模型"""
    images, labels = test_data()
    print(f"测试集: images {images.shape}, labels {labels.shape}")

    net = LandslideCNN()
    net.load_state_dict(torch.load(config["model_parameter"]))
    net.eval()

    result = []
    acc = 0
    for i in range(images.shape[0]):
        img = Variable(torch.tensor(
            images[i], dtype=torch.float32
        ).reshape((1, config["feature"], config["size"], config["size"])))
        out = net(img)
        acc += calculate_correct(out, labels[i, :])
        result.append(out.detach().numpy())

    result = np.squeeze(np.array(result), axis=1)
    print(f"测试集精度: {acc / len(images) * 100:.2f}%")
    eval_model(result, labels)


def predict():
    """
    全研究区滑坡易发性预测

    对研究区每个像素生成滑坡概率值，输出GeoTIFF格式的滑坡易发性图
    """
    images = pred_data()

    device = torch.device('cpu')
    net = LandslideCNN()
    net.to(device=device)
    net.load_state_dict(torch.load(config["model_parameter"]))
    net.eval()

    result = []
    batch_size = 2500

    for i in range(int(len(images) / batch_size) + 1):
        img = Variable(torch.tensor(
            images[i * batch_size:batch_size * (i + 1)],
            dtype=torch.float32
        ).reshape((-1, config["feature"], config["size"], config["size"])))
        img = img.to(device=device)

        out = net(img)
        out = out.cpu().detach().numpy()
        result.append(out[:, 1])  # 取滑坡概率

        if i % 100 == 0:
            print(f"进度: {i:6d}/{len(images):6d}, 当前概率: {out[0, 1]:.5f}")

    # 合并结果
    result1 = np.array(result[:-1])
    result2 = np.array(result[-1:])
    result = np.append(result1, result2)

    # 保存结果
    save_to_tif(result, "CNN2d_landslide_susceptibility.tif")
    print("滑坡易发性图已保存")


if __name__ == "__main__":
    test()
    # predict()
