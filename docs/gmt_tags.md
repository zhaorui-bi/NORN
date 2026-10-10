# 同龄 GMT 输入与物理项标签（0.5）

## 唯一受控开关

正式默认是 `no_physics`。`physics` 是可选的一致性扩展，不是另一个
backbone，也不把上一年龄预测递推到下一年龄。两版共享 **同一份冻结 NPZ**、
SFNO、数据 loss、划分、seed、优化器／学习率计划及验证 MAE 选模。

```bash
# 一次性获取与工作区 GMT 数值相容的、哈希固定的模型支持文件
uv run --no-sync python scripts/fetch_gmt_model.py

# 准备一次；0–60 Ma 的两个 GMT 源必须同时齐全
norn prepare --config configs/a40.json --output processed/gmt_dataset.npz

# 默认 baseline：不导入、读取、实例化或调用任何物理目标
norn train --config configs/a40.json --tag no_physics

# 可选：从同一数据构造明确的刚性板内假设，再启用物理项
norn prepare-rigid-priors --config configs/a40.json \
  --dataset processed/gmt_dataset.npz --output processed/gmt_rigid_priors.npz
norn train --config configs/a40.json --tag physics \
  --physics-constraints processed/gmt_rigid_priors.npz
```

也可运行 `bash scripts/a40.sh no_physics` 或 `bash scripts/a40.sh physics`。
该脚本只评分验证集；`no_physics` 分支不生成物理先验。输出分别为
`outputs/gmt_a40/no_physics/`、`outputs/gmt_a40/physics/`，不会覆盖历史结果。
命令行 `--output` 可显式覆盖目录；`--dry-run` 展示实际配置。

`Config.with_tag()` 返回副本，不修改原配置。`no_physics` 清除全部物理
路径／权重和时间平滑；直接配置 `data_only` 却留下这些内容仍会报错。
`physics` 必须提供先验文件；若没有声明非零物理权重，显式采用轨迹、预算
各 0.05，其他权重为零。它不改变模型、数据 loss 或选模准则。
`configs/a40_physics.json` 与默认 A40 配置只在物理项、目标模式和输出目录上不同。
checkpoint／manifest 记录 `experiment_tag`；禁止跨标签 resume。

## 两份 GMT 确实进入网络输入

`data.geometry_source="gmt"` 要求 `geometry_mode="dynamic"`，并同时声明
`plate_gmt_dir`、`deformation_gmt_dir`。声明 GMT 后不能被 GPML 路径静默忽略。
工作区两个目录各有 0–66 Ma 的 67 文件；只选择 **0–60 Ma 的各 61 文件**。
缺龄、同龄重复、时间头不一致、非零 ANCHOR 或缺失来源属性都会报错。

每个固定 Gauss 网格上的 `X(a)` 使用：

- 板块 GMT 的同龄多边形分区及外边界；
- 变形 GMT 的同龄网络覆盖与边界，覆盖包括导出的 inactive networks，
  **不是内部应变、速度或岩浆率**；
- 匹配旋转下重建的大陆多边形；
- 同龄刚性分区回映取样的现代厚度条件特征，非零龄网络内、未分区或缺少
  旋转线路的位置明确缺失，不作为古厚度标签。

GMT 已是重建后的年龄坐标；栅格化时不能再旋转一次。观测位置在其真实
年龄节点用相同旋转／静态分区重建。输入、模型年龄、查询和视频图层同龄配对。
非整数年龄使用相邻锚点插值；板块 ID 用最近锚点，不能声称重新解析了事件拓扑。

解析保留完整 `@D` 属性、`@P` 的多个外环及其后 `@H` 孔洞；不把第二个外环
误当孔洞，也不丢弃坏坐标。零面积初生裂谷在原模型注册通过后保留边界距离、
不给区域覆盖，并记录数量；不是偷偷用 GPML 图层替换 GMT。

## 参考框架不能只凭模型名字认定

这两个源的头部声明 **Zahirovic2022、ANCHOR=0**，不是 Müller2019。
较新的 PMM `CombinedRotations.rot` 又更新了绝对参考框架；仅在 30／60 Ma
的初查就出现数百公里偏移。默认使用原导出旋转：

```text
Zahirovic_etal_2022_OptimisedMantleRef_and_NNRMantleRef.rot
SHA256 a2edf515af7f7e4771d58f438b398cb9332826bacffcd0c7606331615a3f4348
```

`fetch_gmt_model.py` 固定下载版本／哈希，不覆盖已有文件。支持来源：
[GPlates PMM](https://github.com/GPlates/plate-model-manager)、
[原旋转文件固定修订](https://github.com/Computational-Planet/reconstructable-tile-layer/blob/05c0417e067a1bedca8e81f93d374a29dab83fb7/apps/reconstructable-tile-layer-demo/public/rotations/Zahirovic_etal_2022_OptimisedMantleRef_and_NNRMantleRef.rot)、
[模型资料 DOI](https://doi.org/10.5281/zenodo.4729045)。外部数据不随源码分发。

准备阶段还在**全部 61 年龄**解析原始 GPML，通过 feature ID 验证两个 GMT
导出的完整要素集合、板块 ID、全部边界顶点及边中点，距离容差 **1 m**。
同文件名但不同旋转内容也会失败。GPML 用于这一注册校验和大陆重建；
板界距离／网络覆盖／同龄分区的实际输入仍来自 GMT。顶点集合相同但顺序
被打乱的多边形不能通过边中点校验。

动态清单记录 122 GMT 文件哈希、模型和 shapefile 属性侧文件哈希、每龄支持
及注册误差。刚性物理先验另绑定同一数据 SHA256 和几何来源，不能复用
Müller2019 旧先验或其他划分／节点位置的数据。

## 科学与历史结果边界

0.4 的 B0/B3 消融使用 Müller2019 GPML，旧配置 `a40_ml_*` 和所有产物保留；
历史受约束配置是 `configs/a40_physics_muller2019_legacy.json`。
0.5 沿用验证侧选择的 B0 网络／数据 loss，但 **GMT 的正式 500 步结果必须
重新训练**。短运行只验证执行路径，不是精度或物理项增益结论。

剔除缺少旋转线路的记录是定位质量规则，不是按验证误差选择样本。两标签
使用同一剔除后的数据。原地理组测试已打开，不应重新用它调参；需要新的
独立评估才能给出最终泛化结论。当前资料仍为 `unverified_proxy`；几何配对
和代码回归通过不验证古厚度真值，也不证明内部变形过程正确。
