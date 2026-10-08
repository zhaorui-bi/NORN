# 推理接口

```python
from norn_earth.inference import NornPredictor, export_prediction

predictor = NornPredictor("best.pt", device="auto")
values = predictor.predict_points(lon=[-120, 240], lat=[40, 40], age=[13.4, 13.4])
maps = predictor.predict_grid([0, 30, 60])
export_prediction(predictor, [0, 13.4, 60], "predictions", formats=("npz", "dat", "netcdf"))
```

`lon`、`lat`、`age` 可广播；返回的点值具有广播后的形状。负经度与对应的 0–360 经度等价。`age` 单位 Ma，必须在 checkpoint 的年代域内；禁止外推。

点坐标是该年龄的古位置，参考框架由 checkpoint 中的旋转模型来源确定。将现代样品位置作为古年龄查询前，应使用同一板块重建模型转换。地图给出各年代在该框架下的空间场，不是把现代坐标固定为材料身份。

所有查询使用与训练相同的锚点厚度线性插值和球面网格双线性取样。不会额外调用一个只在训练中使用的局部 MLP。推理内部缓存 61 张锚点场，缓存只属于当前不可变 checkpoint。

`predict_grid` 返回：

| 键 | 形状 / 含义 |
|---|---|
| `thickness_km` | `(N_age,180,360)`，正厚度 |
| `age_ma` | 查询年龄 |
| `latitude` | `-89.5…89.5`，南到北 |
| `longitude` | `0.5…359.5` |

checkpoint 包含全部模型配置、参数和输入特征。全图或古坐标点查询无需原始观测或物理文件；使用 CPU 时不会依赖原 GPU 设备。需要安装与发布版本相容的 `sfno` 依赖。

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

使用 `norn evaluate` 评分时，必须提供训练时冻结的同一个数据 NPZ，并通过哈希核验。评分使用年龄边缘 NLL 和年龄权重下预测均值的 MAE，避免用域外代表年龄做隐式外推。
