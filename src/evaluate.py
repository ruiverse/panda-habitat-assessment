"""
evaluate.py - 模型评估指标工具集

提供滑坡易发性评价模型的全面评估指标：
- AUC (Area Under ROC Curve)
- ACC (Accuracy)
- Kappa 系数
- F1-Score
- Precision / Recall
- RMSE
- MCC (Matthews Correlation Coefficient)
- IOA (Index of Agreement)
- Sensitivity / Specificity
- ROC 曲线导出
"""

from sklearn import metrics
import numpy as np
from scipy.stats import ks_2samp
import pandas as pd


def pre_class(y_probability, threshold=0.5):
    """将概率值转为二分类标签"""
    return [1 if i > threshold else 0 for i in y_probability]


def get_auc(y_real, y_probability):
    """计算 AUC 值"""
    return metrics.roc_auc_score(y_real, y_probability)


def get_acc(y_real, y_probability):
    """计算分类精度"""
    pred_class = pre_class(y_probability)
    return metrics.accuracy_score(y_real, pred_class)


def get_precision(y_real, y_probability):
    """计算精确率"""
    pred_class = pre_class(y_probability)
    return metrics.precision_score(y_real, pred_class)


def get_recall(y_real, y_probability):
    """计算召回率"""
    pred_class = pre_class(y_probability)
    return metrics.recall_score(y_real, pred_class)


def get_f1(y_real, y_probability):
    """计算 F1 分数"""
    pred_class = pre_class(y_probability)
    return metrics.f1_score(y_real, pred_class)


def get_mcc(y_real, y_probability):
    """计算 Matthews 相关系数"""
    pred_class = pre_class(y_probability)
    return metrics.matthews_corrcoef(y_real, pred_class)


def get_RMSE(y_real, y_probability):
    """计算均方根误差"""
    pred_class = pre_class(y_probability)
    mse = metrics.mean_squared_error(y_real, pred_class)
    return mse ** 0.5


def get_MAE(y_real, y_probability):
    """计算平均绝对误差"""
    mae = metrics.mean_absolute_error(y_real, y_probability)
    return mae


def get_kappa(y_real, y_probability):
    """计算 Cohen's Kappa 系数"""
    pred_class = pre_class(y_probability)
    return metrics.cohen_kappa_score(y_real, pred_class)


def get_ROC(data_input_y, y_probability, save_path):
    """计算并导出 ROC 曲线数据"""
    fpr, tpr, thresholds = metrics.roc_curve(data_input_y, y_probability)
    fpr, tpr = pd.DataFrame(fpr), pd.DataFrame(tpr)
    roc = pd.concat([fpr, tpr], axis=1)
    roc.to_csv(save_path)


def get_IOA(y_real, y_probability):
    """计算一致性指数 (Index of Agreement)"""
    y_pred = pre_class(y_probability)
    y_real_average = np.average(y_real)
    y_pred_average = np.average(y_pred)
    top = sum((y_pred[i] - y_real[i]) ** 2 for i in range(len(y_real)))
    down = sum((np.fabs(y_real[i] - y_real_average) +
                np.fabs(y_pred[i] - y_pred_average)) ** 2
               for i in range(len(y_real)))
    return 1 - top / down


def get_TPR_FPR(y_real, y_probability):
    """计算灵敏度 (Sensitivity) 和特异度 (Specificity)"""
    y_pred = pre_class(y_probability)
    tn, fp, fn, tp = metrics.confusion_matrix(y_real, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)
    return sensitivity, specificity


def full_evaluation(y_real, y_probability, save_roc_path=None):
    """
    完整评估报告

    参数：
        y_real: 真实标签
        y_probability: 预测概率
        save_roc_path: ROC曲线保存路径（可选）
    """
    acc = get_acc(y_real, y_probability)
    auc = get_auc(y_real, y_probability)
    kappa = get_kappa(y_real, y_probability)
    ioa = get_IOA(y_real, y_probability)
    mcc = get_mcc(y_real, y_probability)
    precision = get_precision(y_real, y_probability)
    recall = get_recall(y_real, y_probability)
    f1 = get_f1(y_real, y_probability)
    rmse = get_RMSE(y_real, y_probability)
    sensitivity, specificity = get_TPR_FPR(y_real, y_probability)

    print(f"ACC = {acc:.4f}  AUC = {auc:.4f}  Kappa = {kappa:.4f}")
    print(f"IOA = {ioa:.4f}  MCC = {mcc:.4f}")
    print(f"Precision = {precision:.4f}  Recall = {recall:.4f}  F1 = {f1:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"Sensitivity = {sensitivity:.4f}  Specificity = {specificity:.4f}")

    if save_roc_path:
        get_ROC(y_real, y_probability, save_roc_path)
        print(f"ROC曲线已保存至: {save_roc_path}")
