# 🐼 基于深度学习的卧龙大熊猫震后生境变化评价研究

> 四川省大学生创新创业训练计划（省级大创项目）  
> 项目负责人（5人团队） | 2022–2024  
> 成果：软件著作权（第二作者）— 考虑地质灾害易发性的野生大熊猫生境选择分析软件

---

## 📋 项目简介

2008年汶川8.0级特大地震诱发大量滑坡、崩塌等次生地质灾害，对距震中不到20km的卧龙国家级自然保护区造成严重生态破坏，大熊猫生境破碎化进一步加重。

本项目以卧龙自然保护区为研究区，构建了**两阶段评价模型**：

**阶段一：SE-CNN 滑坡易发性评价**
- 基于7个地质影响因子（高程、坡度、坡向、NDVI、断层/河流/道路距离）
- 使用 SE注意力机制 + CNN，对研究区每个像素预测滑坡概率
- 输出滑坡敏感性分区图（5级：极低→极高）

**阶段二：随机森林 生境敏感性评价**
- 将阶段一的滑坡易发性结果作为11个评价因子之一
- 结合地形、气候、植被、大熊猫足迹可达性、水资源、人类活动等因子
- 使用随机森林（500棵决策树）构建生境评价模型
- 输出大熊猫生境敏感性分区图（5级：不敏感→极敏感）

---

## 🔧 技术栈

| 类别 | 工具 |
|------|------|
| 深度学习 | **PyTorch**（SE-CNN 滑坡模型） |
| 机器学习 | **scikit-learn**（随机森林生境模型） |
| 遥感处理 | **GDAL/OGR**（GeoTIFF 读写、矢量转栅格） |
| 遥感预处理 | ENVI 5.3（辐射定标、大气校正） |
| 空间分析 | ArcGIS 10.8（缓冲分析、成本距离、因子制图） |
| 数据处理 | NumPy、Pandas |
| 模型评估 | AUC、Kappa、F1-Score、ROC曲线 |

---

## 🧠 模型架构

### SE-CNN 滑坡易发性模型

```
输入: [batch, 7, 11, 11] — 7个影响因子，11×11像素窗口

SE-Block(7ch) → Conv2d(7→64) → BN → Tanh → MaxPool(2×2)
SE-Block(64ch) → Conv2d(64→132) → BN → Tanh → MaxPool(2×2)
SE-Block(132ch) → Conv2d(132→196) → BN → Tanh
Flatten → Linear(784→384) → Linear(384→2) → Sigmoid

输出: [batch, 2] — 非滑坡/滑坡概率
```

**SE注意力模块**（Squeeze-and-Excitation）：通过全局平均池化 + 两层全连接，自适应学习各影响因子通道的重要性权重。

### 随机森林生境评价模型

```
输入: 11个评价因子（含CNN滑坡易发性结果）
模型: RandomForestRegressor(n_estimators=500, max_features='sqrt')
划分: 训练集90% / 测试集10%
输出: 生境敏感性连续值 → 重分类为5级
```

---

## 📊 研究区与数据

**研究区：** 四川卧龙国家级自然保护区（东经102°52′–104°53′，北纬30°42′–31°38′）

### CNN 滑坡评价因子（7个）

| 因子 | 分辨率 | 来源 |
|------|--------|------|
| 高程 (DEM) | 30m | SRTM |
| 坡度 | 30m | DEM派生 |
| 坡向 | 30m | DEM派生 |
| NDVI | 30m | Landsat |
| 断层距离 | 30m | 地质图矢量化 |
| 道路距离 | 30m | OpenStreetMap |
| 河流距离 | 30m | 水系矢量数据 |

### RF 生境评价因子（11个，7大类）

地质灾害敏感性、地形敏感性、气候敏感性、植被敏感性、动物痕迹敏感性、水资源敏感性、人类活动敏感性

标签数据来源：全国第四次大熊猫调查报告（国家林业局，2015年）

---

## 📁 项目结构

```
panda-habitat-assessment/
├── src/                        # 源代码
│   ├── create_label.py        # 标签数据生成（坐标→GeoTIFF）
│   ├── read_data.py           # 数据读取与CNN数据集构建
│   ├── model.py               # SE-CNN 模型定义
│   ├── train_cnn.py           # CNN 训练流程
│   ├── predict_cnn.py         # CNN 测试与全区预测
│   ├── evaluate.py            # 评估指标工具集
│   └── rf_habitat.py          # 随机森林生境评价（完整流程）
├── data/                       # 数据目录（未上传，见说明）
│   └── README.md              # 数据来源与下载说明
├── results/                    # 结果图
│   └── README.md              # 结果文件说明
├── requirements.txt            # Python依赖
├── LICENSE                     # MIT许可证
└── README.md                   # 本文件
```

---

## 🚀 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/ruiverse/panda-habitat-assessment.git
cd panda-habitat-assessment

# 创建虚拟环境
conda create -n panda python=3.9
conda activate panda

# 安装依赖
pip install -r requirements.txt

# GDAL 建议通过conda安装
conda install -c conda-forge gdal
```

### 运行流程

```bash
cd src

# 阶段一：CNN 滑坡易发性评价
python create_label.py       # 1. 生成标签
python train_cnn.py          # 2. 训练SE-CNN模型
python predict_cnn.py        # 3. 测试评估 + 全区预测

# 阶段二：随机森林 生境评价
python rf_habitat.py         # 4. 训练RF + 全区预测
```

---

## 🏆 项目成果

- ✅ **软件著作权**：考虑地质灾害易发性的野生大熊猫生境选择分析软件（第二作者）
- ✅ **省级大创项目**：顺利结题
- ✅ **创新点**：将CNN滑坡易发性结果作为生境评价因子，首次将强震地质灾害与大熊猫生境选择定量关联

---

## 📚 参考文献

1. 欧阳志云等. 卧龙自然保护区大熊猫生境评价[J]. 生态学报, 2001.
2. 白文科等. 卧龙自然保护区大熊猫空间利用格局动态变化特征[J]. 兽类学报, 2017.
3. 徐卫华等. 基于遥感和GIS的秦岭山系大熊猫生境评价[J]. 遥感技术与应用, 2006.

---

## 👩‍💻 作者

**ruiverse**  
中国科学院国家空间科学中心 · 硕士在读  
GitHub: [@ruiverse](https://github.com/ruiverse)

---

## 📄 许可证

[MIT License](LICENSE)
