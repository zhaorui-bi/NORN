# 逐年代动态板块重建与视频

0.5 默认 `no_physics` 与可选 `physics` 都共享同龄双 GMT 输入。默认使用[纯 ML baseline](ml_baseline.md)；详细开关／参考框架核对见 [GMT 与标签](gmt_tags.md)。运动几何是输入，独立于是否启用一致性项。

## 核心变化

原实现向所有年龄广播同一份现代空间输入。0.3 改为

$$H_\theta(x,a)=\mathrm{softplus}(F_\theta(X(a),a)(x))+\epsilon.$$

- **网格固定**：内部北到南 Gauss 纬度／周期经度；不是把经纬度网格本身移动。
- **几何移动**：每个年龄独立重建大陆、解析板界和网络轮廓，并绑定该年龄的空间特征。
- **厚度反演**：同一套网络参数在全部年代共享，由古观测和现代监督拟合（0.3 的物理扩展是单独的历史路径）；不是把现代厚度直接旋转当作历史真值。
- **时间配对**：`inputs[m]`、`feature_ages[m]`、FiLM 年龄和输出 `H[m]` 必须一致。两遍梯度重计算使用完全相同的配对。

地质年龄范围仅为 0–60 Ma。0→60 的视频是现代向过去的重建；历史物理扩展才使用向现代推进的时间 `tau=60-age`，纯 ML 目标不使用该方程。地质年龄不是 diffusion 加噪／去噪步数，也没有把上一张预测图强制作为下一张图的输入。

## 数据与来源一致性

动态模式使用同一模型下的四种资料：

| 配置项 | 用途 |
|---|---|
| `rotations` | 同一参考框架的板块旋转 |
| `static_polygons` | 现代样品分区及真实观测年龄节点的古位置 |
| `continental_polygons` | 每个锚点的移动大陆掩膜 |
| `topologies` | 原 GPML 或含相互引用文件的目录；逐龄校验 GMT 要素、顶点和边 |
| `plate_gmt_dir` | 实际输入的同龄板块多边形／边界 |
| `deformation_gmt_dir` | 实际输入的同龄网络覆盖／边界 |

A40 配置采用与两个 GMT 数值注册相容的 **Zahirovic2022 原始旋转框架、ANCHOR=0**。全部 61 年龄核验，不凭文件名接受更新后的旋转模型；GMT 不再被二次重建。多个外环和孔洞保留，原模型中的零面积初生网络保留边界但无覆盖。shp 的 dbf/shx/prj／GPlates 属性侧文件和所用 GMT 都参与哈希。旧 Müller2019 GPML 路径保留在历史 `a40_ml_*` 配置中。已有 Excel 古坐标不替代主模型位置。

8 个通道中位置编码保持固定；大陆掩膜、板界距离、网络覆盖随年龄变化。现代参考值在同龄刚性板块点回映到现代后取样；非零年龄网络内／未分配点参考缺失并以零输入，支持数量记录在清单中。这只是条件特征，不增加任何古标签；现代损失只对应真实 0 Ma 产品。

大陆掩膜是点中心几何覆盖，导出后可有插值产生的分数；不是古海岸线／海平面模型。板界距离用 0.5° 加密后的顶点近邻，最大段内误差约半段长度（约 28 km），不是精确距离或洋脊／海沟分类。网络轮廓只说明覆盖，不能用于推导内部应变或岩浆生产率。

## 冻结与训练

```bash
uv sync --locked --extra cuda --extra kinematics --extra export --extra viz
norn prepare --config configs/a40.json --output processed/gmt_dataset.npz
norn train --config configs/a40.json --tag no_physics
```

或使用 `bash scripts/a40.sh no_physics` / `bash scripts/a40.sh physics` 执行准备、训练、验证、地图及视频导出（需要 ffmpeg）。默认输出为 `outputs/gmt_a40/no_physics`／`physics`，不会覆盖 `outputs/ml_control_a40`、`outputs/dynamic_a40` 或 `outputs/release_a40`。

输入 shape 为 `61×8×180×360`。schema-2 NPZ/checkpoint 保存年龄顺序和全部逐龄几何；schema 1 会拒绝加载，必须重新准备并训练。即使网络通道数没有变化，输入语义也发生变化，不能沿用旧权重声称动态重建。

默认划分政策仍为地理组留出，不在本次架构修改中改变挑战／验证记录，以免混淆对照。训练不评分测试集。真实资料来源仍为 `unverified_proxy`，动态图层不自动使标签来源得到验证。

在可选 `physics` 路径中，材料轨迹和移动区域预算是明确的刚性无净源先验；默认 baseline 不计算这些项。新版本在全部物理节点筛查同龄边界距离／网络覆盖；只观察现代位置远离边界是不够的。该几何筛查不能替代独立地质过程证据。

## 导出与视频

```bash
norn infer --checkpoint outputs/gmt_a40/no_physics/best.pt --ages all \
  --output outputs/gmt_a40/no_physics/inference --device cuda --formats npz dat netcdf
norn animate --checkpoint outputs/gmt_a40/no_physics/best.pt --ages all \
  --output outputs/gmt_a40/no_physics/evolution_0_60Ma.mp4 --device cuda \
  --fps 6 --vmin 0 --vmax 80
```

- NPZ/NetCDF 同时导出厚度和同龄大陆覆盖、板界距离、网络覆盖、板块 ID。
- 默认仅用大陆覆盖遮罩显示大陆厚度；`--all-crust` 同时显示洋壳。遮罩不改变科学输出中的全球厚度值。
- 白线来自重建大陆轮廓，灰线来自同龄格点板块 ID；没有叠加固定现代海岸线。
- 全视频固定色标，避免逐帧自动缩放制造虚假厚度变化。
- MP4 使用系统 ffmpeg；改成 `.gif` 可使用 Pillow。另保存 0／中间／末帧预览 PNG，以及逐帧年龄、色标、来源和哈希 JSON。
- 连续厚度／几何图层按锚点插值；离散板块 ID 使用最近锚点，不平均 ID。非整数帧不是重新解析过的拓扑，跨事件插值可能不可靠；真正更细时间步需重新准备几何并训练。

动态图层确保模型能够使用已给定的运动几何，不保证反演厚度已经唯一或正确。视频不是新增观测、优化收敛证明或可信度图。

## 回归证据

`tests/test_gmt_geometry.py` 另检查两份 GMT 的实际输入、完整要素及同名错误框架拒绝、零面积网络、多外环／孔洞、两标签共享输入及数据 loss、禁止跨标签恢复和删除来源后的独立推理。`tests/test_dynamic_geometry.py` 在临时目录生成一个 60 Myr 旋转 90° 的已知板块，检查大陆／板块 ID／现代参考随年龄移动、古观测定位一致、动态输入两遍梯度等价，以及删除全部原始板块文件后的地图、几何和 GIF 导出。真实 A40 运行证据另见工作区 `outputs/dynamic_a40/` 的日志和清单；历史 0.2 指标不能作为此版本的精度证据。
