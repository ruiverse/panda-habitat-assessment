# 数据说明

本项目使用的数据均为公开数据，不依赖于现场调查。

## CNN 滑坡易发性评价数据（`cnn_data/`）

7个影响因子（30m分辨率GeoTIFF）：

| 因子 | 文件名 | 来源 |
|------|--------|------|
| 高程 (DEM) | dem.tif | SRTM 30m |
| 坡度 | slope.tif | DEM派生 |
| 坡向 | aspect.tif | DEM派生 |
| NDVI | ndvi.tif | Landsat 8 OLI |
| 断层距离 | fault.tif | 地质图矢量化 |
| 道路距离 | road.tif | OpenStreetMap |
| 河流距离 | river.tif | 水系矢量数据 |

标签数据来源：2008年汶川地震滑坡点位（遥感解译 + 历史记录）

## RF 生境评价数据（`RF-data/`）

11个评价因子（7大类），详见论文。

标签数据来源：全国第四次大熊猫调查报告（国家林业局，2015年）

## 数据下载

由于数据体积较大（~200MB），未包含在仓库中。
如需复现，请联系作者获取数据，或从以下来源自行下载：
- SRTM DEM: https://earthexplorer.usgs.gov/
- Landsat: https://www.gscloud.cn/
- 大熊猫调查数据: 国家林业局第四次大熊猫调查报告
