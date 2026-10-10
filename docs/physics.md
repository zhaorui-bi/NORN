# 可选物理约束 NPZ 接口

默认 `--tag no_physics` 完全不导入或加载本接口。`--tag physics --physics-constraints ...` 才启用；两标签共享同龄双 GMT 输入、模型和数据 loss，见 [GMT／标签协议](gmt_tags.md)。

所有物理量由外部资料或明确的先验假设提供，不能根据最终测试误差生成适用掩膜。软件没有自动填补未知通量或自由逐像素源项。

NPZ 使用 `allow_pickle=False` 读取；`metadata` 为一个 JSON 字符串标量，至少包括：

```json
{
  "schema_version": 1,
  "time_coordinate": "forward_tau=max_age-age",
  "kind": "independently_supported_constraints",
  "evidence_ids": {
    "trajectory": ["region_A_kinematics"],
    "budget": ["region_B_kinematics"],
    "birth": ["ridge_C_retained_production"],
    "source": ["region_D_net_source_prior"]
  }
}
```

同时保存资料来源、哈希、单位、区域身份、适用范围和跨事件分段依据。`evidence_ids` 中每个因子一个 ID；source 列表声明源先验的父证据。启用角色之间共享同一 ID 会报错；同角色的重复 ID 按组平均，不能因增加求积点而增加证据量。

令 R/B/M 为轨迹／预算／出生因子数，T 为时间节点数，P 为内部面积点数，E 为边界点数，C 为低维源系数数。时间数组使用**递减地质年龄**，对应递增物理时间。坐标为各节点所处年龄的球面位置，单位度。

## 有界源系数（可选）

| 数组 | 形状 | 定义 |
|---|---|---|
| `source_lower`, `source_upper` | `(C,)` | 有限上下界 |
| `source_prior_mean` | `(C,)` | 严格位于上下界内部 |
| `source_prior_cov` | `(C,C)` | 正定、对称的系数协方差 |
| `source_smooth_pairs` | `(N_pair,2)` | 可选，施加平滑的系数索引对 |

系数默认具有 km/Myr 的净厚度源意义，基函数应与其单位相容。最多 128 个固定区域／时间系数；启用学习源项必须给出非零 `source_prior_weight`。不分别宣称识别了总生产、保留比例和损失。

没有学习源项时 C=0，`*_source_basis` 仍提供末维 0 的显式数组，并声明源项已知及其无源假设。未知源项不能仅以空基函数代替。

## 材料轨迹

| 数组 | 形状 | 定义 |
|---|---|---|
| `trajectory_active`, `trajectory_source_known` | `(R,)` bool | 有效域及源信息支持 |
| `trajectory_lon`, `trajectory_lat`, `trajectory_age` | `(R,T)` | 轨迹节点，T≥2 |
| `trajectory_divergence_per_myr` | `(R,T)` | 水平散度，1/Myr |
| `trajectory_source_basis` | `(R,T,C)` | 净源基函数 |
| `trajectory_sigma_km` | `(R,)` | 轨迹厚度变化的正误差尺度 |

实现 $H_{end}-H_{start}-\int(q-H\,div\,v)d\tau$。源未知或域无效的因子不启用。

## 弱形式体积预算

| 数组 | 形状 | 定义 |
|---|---|---|
| `budget_active`, `budget_source_known`, `budget_flux_known` | `(B,)` bool | 有效域、源和完整边界通量支持 |
| `budget_age` | `(B,T)` | 递减年龄 |
| `budget_lon`, `budget_lat`, `budget_area_km2` | `(B,T,P)` | 移动控制域的内部点和正面积权重 |
| `budget_source_basis` | `(B,T,P,C)` | 内部净源基函数 |
| `budget_boundary_lon`, `budget_boundary_lat` | `(B,T,E)` | 边界求积点 |
| `budget_relative_normal_velocity_km_myr` | `(B,T,E)` | $(v-v_b)\cdot n$，外向为正 |
| `budget_boundary_length_km` | `(B,T,E)` | 非负边长权重 |
| `budget_reference_volume_km3` | `(B,)` | 固定正参考体积 $H_{ref}A_{ref}$ |
| `budget_sigma_normalized` | `(B,)` | 无量纲残差的正误差尺度 |

实现 $(\Delta V+I_{flux}-I_{source})/V_{ref}$。未知通量不置零；任一支持掩膜 false 时，整个因子移除，其无效数组允许含 NaN。活动因子里的 NaN 会报错。

动态模式的 `prepare-rigid-priors` 会在每个物理积分年龄检查轨迹／材料域点：远离同龄板界，并在网络轮廓覆盖之外（GMT 模式使用同龄导出，包含 inactive outlines）。派生先验绑定 `geometry_source`、几何源哈希和冻结数据 `dataset_sha256`；不允许将旧 GPML 先验用于 GMT 或改变划分／节点位置后的数据。不能只以现代板内位置筛选 60 Myr 的整条路径；这些检查仍不构成独立的无变形或无源证据。

对刚性共动材料域，`v=v_b` 是模型内可计算的关系，可以明确提供零相对通量；它与“缺少通量资料所以填零”不同。`prepare-rigid-priors` 生成的样例还假设净源为零，因此必须作为模型先验解释，不能声称源汇已经测量。

## 洋壳出生先验

| 数组 | 形状 | 定义 |
|---|---|---|
| `birth_active` | `(M,)` bool | 适用掩膜 |
| `birth_lon`, `birth_lat`, `birth_age` | `(M,)` | 出生查询位置和年龄 |
| `birth_retained_production_km2_myr` | `(M,)` | 正的单位脊长保留熔体体积产率 |
| `birth_full_spreading_km_myr` | `(M,)` | 正的总扩张速度 |
| `birth_sigma_km` | `(M,)` | 正厚度先验尺度 |

比较预测厚度与 `Q_retained/u_full`。相同生产率资料如果已经作为源项先验与条件预算使用，不能再使用另一个 ID 把其派生厚度当成独立证据。

## 配置和示例

```json
{
  "physics": {
    "constraints": "physics.npz",
    "trajectory_weight": 0.1,
    "budget_weight": 0.1,
    "birth_weight": 0.1,
    "source_prior_weight": 0.01,
    "source_smooth_weight": 0.0
  }
}
```

`norn demo` 在 [data/demo.py](../src/norn_earth/data/demo.py) 中生成可执行的完整示例。测试使用已知真值检查源项符号、体积预算、反向传播及来源重复计数。实际研究仍需要独立核验物理文件的地质适用性。
