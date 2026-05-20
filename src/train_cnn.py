"""
train_cnn.py - SE-CNN 滑坡易发性评价模型训练

训练流程：
1. 加载7个影响因子的GeoTIFF数据
2. 构建像素级 11×11 滑动窗口数据集
3. 使用 SE-CNN 模型进行二分类训练（滑坡/非滑坡）
4. 保存训练曲线和模型参数

超参数：
- 优化器: SGD (lr=0.01, momentum=0.9)
- 损失函数: CrossEntropyLoss
- 批次大小: 32
- 训练轮次: 80
- 像素窗口: 11×11
"""

import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable

from read_data import config, train_data, test_data, save_to_excel
from model import LandslideCNN
import evaluate as evaluate_method


def calculate_correct(outputs, targets):
    """计算批次内预测正确的样本数"""
    outputs = outputs.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    correct = 0
    for i in range(outputs.shape[0]):
        predict = 0 if outputs[i, 0] >= outputs[i, 1] else 1
        if predict == targets[i]:
            correct += 1
    return correct


def calculate_MAE(outputs, targets):
    """计算批次内的RMSE"""
    outputs = outputs.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    mae = evaluate_method.get_RMSE(targets, outputs[:, 1])
    return mae


def train():
    """SE-CNN 模型训练主函数"""
    device = torch.device('cpu')
    net = LandslideCNN()
    net.to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-2, weight_decay=0, momentum=0.9)

    # 加载数据
    imgs, labs = train_data()
    imgs_test, labs_test = test_data()
    imgs = np.array(imgs).astype(np.float32)
    labs = np.squeeze(np.array(labs).astype(np.long))
    imgs_test = np.array(imgs_test).astype(np.float32)
    labs_test = np.squeeze(np.array(labs_test).astype(np.long))

    print(f"训练集: imgs {imgs.shape}, labs {labs.shape}")
    print(f"测试集: imgs_test {imgs_test.shape}, labs_test {labs_test.shape}")

    Loss, MAE, ACC_train = [], [], []

    for epoch in range(config["epochs"]):
        train_loss, train_acc, train_mae = 0, 0, 0
        net.train()

        train_dataset = data_utils.TensorDataset(
            torch.from_numpy(imgs).float(),
            torch.from_numpy(labs).long()
        )
        train_loader = data_utils.DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True
        )

        for batch_id, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            train_acc += calculate_correct(outputs, targets.long())
            train_mae += calculate_MAE(outputs, targets.long())

        print("Epoch {:3d} | Loss {:.5f} | ACC {:.2f}% | MAE {:.2f}%".format(
            epoch, train_loss / (batch_id + 1),
            train_acc / len(imgs) * 100,
            train_mae / len(imgs) * 100))

        Loss.append(train_loss / (batch_id + 1))
        MAE.append(train_mae / (batch_id + 1))
        ACC_train.append(train_acc / len(imgs) * 100)

    # 保存训练曲线
    os.makedirs("Loss_curve", exist_ok=True)
    save_to_excel(Loss, "Loss_curve/CNN2D_train_LOSS.xlsx")
    save_to_excel(MAE, "Loss_curve/CNN2D_train_MAE.xlsx")
    save_to_excel(ACC_train, "Loss_curve/ACC_train.xlsx")

    # 保存模型
    torch.save(net.state_dict(), config["model_parameter"])
    print(f"模型已保存至: {config['model_parameter']}")


if __name__ == "__main__":
    import os
    train()
