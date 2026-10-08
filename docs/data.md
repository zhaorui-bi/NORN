# 数据准备与坐标

## 输入观测

CSV 必须具有以下列：

| 列 | 定义 |
|---|---|
| `observation_id` | 唯一记录 ID，重复 ID 会报错 |
| `present_lon` / `present_lat` | 现代经纬度，单位度；经度周期处理 |
| `age_lower_ma` / `age_upper_ma` | 年龄支持区间，Ma；点年龄上下限相同 |
| `thickness_km` | 正厚度或已声明为厚度代理的值，km |

支持原项目中文 XLSX，由 `data/observations.py` 的列映射处理。原始文件只读。非有限值、非法纬度和非正厚度不参与训练；存在 QC 列时，默认另外剔除非法位置、负代表年龄、小于 2 km 及域外点年龄记录。不会按验证误差筛选记录。

`source_semantics` 必须声明为 `unverified_proxy`、`verified_thickness` 或 `synthetic`。声明 `verified_thickness` 由数据使用者提供依据；软件不能替代来源验证。

## 现代 DAT

输入为 64,800 行、3 列 `longitude latitude negative_encoded_thickness`。原生格点：180 纬度 × 360 经度，纬度 `-89.5…89.5` 升序，经度 `0.5…359.5` 升序。加载时核验形状、坐标次序和负编码符号，转为正厚度 km。现代误差尺度由配置指定，当前样例 3 km，并未声明已校准。

## 板块重建

`coordinate_mode=paleo` 需要与研究范围相容的 `.rot` 和 `.gpml`/`.gpmlz` 静态分区文件，以及 `kinematics` 依赖。每个观测在其每个年龄节点重建，位置由旋转引擎产生；不使用 Excel 已有古坐标列作为训练输入。

原工作区样例使用 Müller2019 文件：

```text
data/external/muller2019/Rotations/Muller_etal_2019_CombinedRotations.rot
data/external/muller2019/StaticPolygons/Muller_etal_2019_Global_StaticPlatePolygons.gpmlz
```

这些外部文件不随源码包分发。提供自己的许可数据，或从原板块模型提供者获取匹配版本。准备 manifest 保存实际文件 SHA256，而不是只记录模型名称。静态分区的刚性旋转不代表已经解析所有变形网络、材料出生/消亡或拓扑事件。

`position_failure=drop` 将任一年龄节点无法定位的记录整条剔除；`error` 则停止并报告。选择 `coordinate_mode=present` 是明确的统计对照，不会被描述为古位置重建。

## 网格与特征

内部球谐网格为 Gauss 纬度北到南，经度从 0 开始。现代输入使用球面积交叠映射到内部网格，其面积权重与 Gauss 求积权重一致，并处理经度接缝。现代损失和导出通过双线性、周期经度查询回到原生格点。

8 通道顺序：`sin_lat`、`cos_lat_cos_lon`、`cos_lat_sin_lon`、`modern_thickness_div40`、`modern_continental_mask`、`static_plate_edge_distance_div3000`、`deformation_coverage`、`cos_lat_squared`。大陆掩膜以现代厚度 20 km 为阈值，是数据派生特征。静态分区距离来自当前模型分区格点边缘，不代表经过分类的动态洋脊／海沟距离。无变形覆盖资料时通道为零，manifest 的支持标记为 false。

## 划分与证据

默认按现代位置构造约 100 km 网格的地理代理组，再以固定种子划分验证组。该代理不等于独立研究／样品 ID，也不保证相邻组之间具有空间缓冲。

可提供 `holdout_csv`，必须含 `observation_id` 和指定布尔列。选中的记录所在**整个地理组**都移出训练；因此测试记录数可能大于最初挑战框选的记录数。测试不会在训练中自动评分。现代资料保留，属于允许现代端点的任务。

## 冻结产物

`norn prepare` 写入无 pickle 的 NPZ：输入特征、现代观测、每条记录的目标／误差／划分、年龄节点及权重、各节点古位置和 JSON 元数据。旁边 JSON 保存来源哈希、保留的年龄质量、位置失败数量和特征支持状态。

`data.prepared` 可直接引用该冻结 NPZ；模型格点、年代域和语义需相容。训练 checkpoint 保存数据哈希，恢复和评估要求使用同一冻结数据。调整误差尺度、位置模型或划分时，应重新准备数据并开始新的运行。
