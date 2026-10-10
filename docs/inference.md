# 推理接口

```python
from norn_earth.inference import NornPredictor, export_prediction

predictor = NornPredictor("best.pt", device="auto")
values = predictor.predict_points(lon=[-120, 240], lat=[40, 40], age=[13.4, 13.4])
maps = predictor.predict_grid([0, 30, 60])
geometry = predictor.predict_geometry([0, 30, 60])
export_prediction(predictor, [0, 13.4, 60], "predictions", formats=("npz", "dat", "netcdf"))
```

`lon`、`lat`、`age` 可广播；返回的点值具有广播后的形状。负经度与对应的 0–360 经度等价。`age` 单位 Ma，必须在 checkpoint 的年代域内；禁止外推。

点坐标是该年龄的古位置，参考框架由 checkpoint 中的旋转模型来源确定。将现代样品位置作为古年龄查询前，应使用同一板块重建模型转换。地图给出各年代在该框架下的空间场，不是把现代坐标固定为材料身份。

默认 CUDA 卷积可使用 TF32，与 CPU 不保证位级一致；需要严格 float32 对照时，在预测前设置 `torch.backends.cudnn.allow_tf32=False` 和 `torch.backends.cuda.matmul.allow_tf32=False`。数值核对脚本分别记录默认路径误差和严格 float32 误差，不混淆精度策略。

所有查询使用与训练相同的锚点厚度线性插值和球面网格双线性取样。不会额外调用一个只在训练中使用的局部 MLP。推理内部缓存 61 张锚点场，缓存只属于当前不可变 checkpoint。

`predict_grid` 返回：

| 键 | 形状 / 含义 |
|---|---|
| `thickness_km` | `(N_age,180,360)`，正厚度 |
| `age_ma` | 查询年龄 |
| `latitude` | `-89.5…89.5`，南到北 |
| `longitude` | `0.5…359.5` |
| `continental_fraction` | `(N_age,180,360)`，同龄大陆覆盖，0–1 |
| `plate_boundary_distance_km` | 同龄板界距离，封顶 3000 km |
| `deformation_coverage` | 解析网络覆盖，不是应变 |
| `plate_id` | 最近年龄锚点／最近空间节点的板块 ID，未分配为 −1 |

checkpoint schema 2 包含全部模型配置、参数、逐龄输入特征、年龄索引和板块 ID。不同年代必须使用各自的 `X(a)`；0.2 静态权重不兼容。连续厚度和几何通道在相邻锚点间插值，离散 ID 不插值；跨拓扑事件的插值只是显示近似，真实更细步长需要重新解析几何、准备并训练。全图或古坐标点查询无需原始观测或物理文件；使用 CPU 时不会依赖原 GPU 设备。需要安装与发布版本相容的 `sfno` 依赖。

导出的元数据声明 `interval_kind=point_estimate` 和 `calibrated_uncertainty=false`。年龄积分处理观测年龄误差，本身不能提供已校准厚度区间。

CLI 点查询 CSV：

```csv
longitude,latitude,age_ma
-120,40,13.4
85,32,40
```

```bash
norn predict-points --checkpoint best.pt --csv queries.csv --output values.csv --device cpu
```

## 移动板块与厚度演化视频

```bash
norn animate --checkpoint outputs/dynamic_a40/best.pt --ages all \
  --output outputs/dynamic_a40/evolution_0_60Ma.mp4 --device cuda \
  --fps 6 --vmin 0 --vmax 80
```

需要 `viz` 依赖和系统 `ffmpeg`；`.gif` 可不使用 ffmpeg。默认用同龄大陆掩膜只着色大陆地壳，白线为同龄大陆轮廓、灰线为格点化板块边缘，`--all-crust` 同时着色洋壳。所有帧固定色标；不叠加现代静态海岸线。默认年龄递增，0→60 Ma 表示向过去重建，不是优化迭代或 diffusion 去噪。旁边保存三个年代的 PNG 预览和帧年龄／来源／色标 JSON；已有视频不覆盖。

视频只允许动态几何 checkpoint。年代间变化是点估计，不是已验证的全球历史；1 Myr 间距也不是可辨识分辨率。详见 [动态重建](dynamic.md)。

使用 `norn evaluate` 评分时，必须提供训练时冻结的同一个数据 NPZ，并通过哈希核验。评分使用年龄边缘 NLL 和年龄权重下预测均值的 MAE，避免用域外代表年龄做隐式外推。
