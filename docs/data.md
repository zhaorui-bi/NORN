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

当前正式 GMT 配置使用 Zahirovic2022 原始旋转框架，并对全部同龄 GMT 要素作数值注册，见 [GMT／标签协议](gmt_tags.md)。历史 0.3/0.4 样例使用 Müller2019 文件：

```text
data/external/muller2019/Rotations/Muller_etal_2019_CombinedRotations.rot
data/external/muller2019/StaticPolygons/Muller_etal_2019_Global_StaticPlatePolygons.gpmlz
```

这些外部文件不随源码包分发。提供自己的许可数据，或从原板块模型提供者获取匹配版本。准备 manifest 保存实际文件 SHA256，而不是只记录模型名称。静态分区的刚性旋转不代表已经解析所有变形网络、材料出生/消亡或拓扑事件。

`position_failure=drop` 将任一年龄节点无法定位或缺少旋转线路的记录整条剔除；`error` 则停止并报告。明确关闭 pyGPlates 缺失板块 ID 时的默认恒等旋转回退，不能将现代位置偷偷当古位置。选择 `coordinate_mode=present` 是明确的统计对照，不会被描述为古位置重建。

## 网格与特征

内部球谐网格为 Gauss 纬度北到南，经度从 0 开始。现代输入使用球面积交叠映射到内部网格，其面积权重与 Gauss 求积权重一致，并处理经度接缝。现代损失和导出通过双线性、周期经度查询回到原生格点。

动态输入形状为 **`(N_anchor,8,nlat,nlon)`**，`inputs[m]` 与 `feature_ages[m]` 严格配对。8 通道顺序：`sin_lat`、`cos_lat_cos_lon`、`cos_lat_sin_lon`、`modern_reference_div40`、`continental_mask`、`plate_boundary_distance_div3000`、`deformation_coverage`、`cos_lat_squared`。

`geometry_mode=dynamic` 要求同一源模型的 `rotations`、`static_polygons`、`continental_polygons`、`topologies`（单一 GPML 或目录）。默认 `geometry_source=gmt` 还要求两个完整的 GMT 目录，板界／网络输入直接由同龄 GMT 产生，原 GPML 用于逐龄注册校验；声明 GMT 后不能静默忽略。大陆掩膜由每个年龄重建的大陆多边形生成；板界距离由每个年龄的板块／网络边界计算（0.5° 加密顶点近邻，封顶 3000 km）；网络轮廓只提供覆盖，不提供应变。现代参考通道沿刚性板块回映到现代位置取样，非零年龄网络内及未分配区域以零表示缺失，并记录支持数量；它是条件特征，不是古厚度标签。现代损失仍只使用真实 0 Ma 产品。

`geometry_mode=static` 仅为显式的统计／合成对照，重复固定特征，不应声称支持移动大陆视频。动态数据准备另见 [动态重建](dynamic.md)。

## 划分与证据

默认按现代位置构造约 100 km 网格的地理代理组，再以固定种子划分验证组。该代理不等于独立研究／样品 ID，也不保证相邻组之间具有空间缓冲。

可提供 `holdout_csv`，必须含 `observation_id` 和指定布尔列。选中的记录所在**整个地理组**都移出训练；因此测试记录数可能大于最初挑战框选的记录数。测试不会在训练中自动评分。现代资料保留，属于允许现代端点的任务。

## 冻结产物

`norn prepare` 写入无 pickle、**schema 2** 的 NPZ：逐年代输入特征、`feature_ages`、逐年代 `plate_ids`、现代观测、每条记录的目标／误差／划分、年龄节点及权重、各节点古位置和 JSON 元数据。旁边 JSON 保存来源哈希（含两个 GMT 源、静态／大陆 shapefile 属性和侧文件）、逐龄注册误差、零面积网络及参考支持统计、年龄质量和位置失败数量。原始资料只读。

`data.prepared` 可直接引用该冻结 NPZ；模型格点、年代域、几何模式／来源（GMT 或 GPML）、年龄排序和语义需相容。checkpoint schema 2 冻结全部逐龄特征及板块 ID，独立推理不重新读取外部板块文件。恢复和评估要求同一数据哈希；调整误差尺度、位置模型、动态几何或划分时，应重新准备数据并开始新运行。0.2 静态数据及权重会被拒绝，不能直接续训。
