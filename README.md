# NORN｜诺恩

**年代不确定性感知、动态几何条件化的全球古地壳厚度反演；当前以纯 ML baseline 为主。**

NORN 默认用年代条件球面神经算子（SFNO）表示 0–60 Ma 的厚度历史，也提供实验性的函数化 Tucker 潜空间模型与条件生成。**每个年代绑定自己的板块几何输入 `X(a)`：网格固定，大陆和板界移动，厚度由模型预测。** 默认 SFNO 的 `--tag no_physics` 仅使用观测和现代监督，不导入物理目标或计算时间平滑；`--tag physics` 只增加显式一致性项。两版都使用同一份逐年代板块／变形区 GMT 输入，不是只加时间 embedding。

> 当前版本：**0.5.0，研究软件 Beta**。0.3/0.4 的 schema-2 权重仍可独立推理，但不能冒充 GMT 重训练结果或跨物理标签恢复；0.2 静态数据／权重不兼容。资料仍为 `unverified_proxy`；默认输出点估计，实验性条件采样提供未校准的潜变量／预测区间。

## 使用 uv 安装

环境统一由 `pyproject.toml` 与 `uv.lock` 管理。锁定配置支持 **Linux x86_64、Python 3.9–3.12**，默认 Python 3.11；CPU 与 CUDA 12.1 使用各自的官方 PyTorch 索引。

```bash
# 如未安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/zhaorui-bi/NORN.git
cd NORN

# CPU 训练、推理与 NetCDF 导出
uv sync --locked --extra cpu --extra export --no-dev

# NVIDIA A40 / CUDA 12.1；与上面的 CPU 配置二选一
# uv sync --locked --extra cuda --extra kinematics --extra export --extra viz --no-dev

source .venv/bin/activate
norn --version
norn --help
```

`uv sync --locked --no-dev` 只安装数据准备所需的基础依赖，不安装 PyTorch。真实古位置重建需要 `--extra kinematics`（pygplates）；绘图可选 `--extra viz`（matplotlib）；NetCDF 导出需要 `--extra export`（xarray）。不要同时启用 `cpu` 与 `cuda`，也不要使用 `--all-extras`。

无需激活环境时，已同步的环境可通过 `uv run --no-sync norn ...` 使用。

## 合成数据自检示例

示例在本地生成已知真值、现代 DAT、古观测和完整物理约束，不需要下载真实数据。CUDA 可替换为 `cpu`。

```bash
norn demo --output outputs/demo --device cuda
norn train --config outputs/demo/config.json
norn infer --checkpoint outputs/demo/run/best.pt \
  --ages 0 13.4 30 60 --output outputs/demo/predictions \
  --device cuda --formats npz dat netcdf
norn evaluate --checkpoint outputs/demo/run/best.pt \
  --dataset outputs/demo/run/dataset.npz --split validation \
  --output outputs/demo/validation.json --device cuda
```

该示例实际启用轨迹、弱形式预算、出生厚度先验和有界净源系数的联合优化。它用于验证软件计算与可控反演，不能用作真实地质过程的精度证据。

## 真实数据训练

所有路径相对于 **JSON 配置文件所在目录** 解析，命令行 `--output` 则相对于当前目录解析。设备由配置或 `--device` 明确选择，发布入口不依赖机器专属路径或外部授权 JSON。

需要的观测 CSV 列：

```csv
observation_id,present_lon,present_lat,age_lower_ma,age_upper_ma,thickness_km
sample_001,-120.0,40.0,12.0,18.0,35.0
sample_002,85.0,32.0,40.0,40.0,60.0
```

经度可使用 `[-180,180]` 或 `[0,360)`；单位分别为度、Ma 和 km。也可以直接读取本项目原始中文列名的 XLSX。现代数据使用 `180×360` 原生格点的三列 DAT，按南到北、经度从 `0.5°` 到 `359.5°` 排列，第三列为历史数据约定的**负编码厚度**；内部和导出结果均使用正厚度。

从下面的模板新建配置文件（例如 `my_config.json`），填写观测、现代 DAT、两个 GMT 目录和匹配的旋转／静态分区／大陆／原始拓扑路径：

```json
{
  "grid": {"nlat": 180, "nlon": 360},
  "model": {"backbone": "sfno", "width": 32, "blocks": 6, "lmax": 32, "mmax": 32},
  "data": {
    "prepared": "processed/gmt_dataset.npz",
    "observations": "data/observations.csv",
    "modern": "data/modern.dat",
    "rotations": "data/external/zahirovic2022/Rotations/<rotation .rot>",
    "static_polygons": "data/external/zahirovic2022/StaticPolygons/<static polygons .shp>",
    "continental_polygons": "data/external/zahirovic2022/ContinentalPolygons/<continental polygons .shp>",
    "topologies": "data/external/zahirovic2022/Topologies",
    "geometry_mode": "dynamic",
    "geometry_source": "gmt",
    "plate_gmt_dir": "data/plate_gmt",
    "deformation_gmt_dir": "data/deformation_gmt",
    "coordinate_mode": "paleo",
    "source_semantics": "unverified_proxy",
    "validation_fraction": 0.1
  },
  "training": {"objective": "data_only", "selection_metric": "validation_mae",
               "observation_huber_weight": 0.0, "steps": 500, "device": "auto",
               "anchor_batch_size": 4},
  "output_dir": "outputs/reconstruction"
}
```

`geometry_mode=dynamic`、`geometry_source=gmt` 在每个锚点读取同龄板块和变形网络，不再二次旋转 GMT。只有明确的统计／合成对照允许 `static`。

```bash
norn prepare --config my_config.json --output processed/gmt_dataset.npz
norn train --config my_config.json --tag no_physics --dry-run
norn train --config my_config.json --tag no_physics
```

`coordinate_mode: paleo` 会在每个真实年龄节点通过 pygplates 重建古位置。无法定位的记录整条剔除或报错，由 `position_failure` 控制；不会无提示地退回现代坐标。年龄超出重建域的部分会被条件化并记录保留质量，域外点年龄不参与训练，推理禁止外推。

GMT 源需实际声明 **Zahirovic2022、ANCHOR=0**，不能混用 Müller2019 或更新后偏移的绝对参考框架。准备过程在全部年龄逐要素核对原始 GPML 的 ID、顶点和边中点（1 m 容差），保留多个外环／孔洞及零面积裂谷的边界，并冻结来源哈希。

可选物理对照（同模型、同 loss、同数据、同种子／计划／选模，只增加显式一致性项）：

```bash
norn prepare-rigid-priors --config my_config.json \
  --dataset processed/gmt_dataset.npz --output processed/gmt_rigid_priors.npz
norn train --config my_config.json --tag physics \
  --physics-constraints processed/gmt_rigid_priors.npz
```

默认配置采用验证侧更好的纯数据 SFNO 对照：180×360 Gauss 网格、6 个模块、32 通道、`lmax=mmax=32`，目标仅为年龄边缘 NLL + 现代数据 NLL，按验证 MAE 选模，不读取物理先验。

## 断点恢复和模型选择

```bash
# 有限步数检查，仍按配置的完整训练计划计算学习率
norn train --config my_config.json --output outputs/check --stop-after 10

# 恢复同一训练计划
norn train --config my_config.json --output outputs/check \
  --resume outputs/check/last.pt
```

每次运行保存：

| 文件 | 内容 |
|---|---|
| `best.pt` | 根据验证侧分数选择的参数，包含配置、输入特征和来源信息 |
| `last.pt` | 最新参数、优化器和随机状态，可继续训练 |
| `training_log.csv` | 原始目标、各损失分量、验证指标、梯度范数和耗时 |
| `run_manifest.json` | 配置、环境、数据哈希和目标模式（纯基线无物理来源项） |
| `training_summary.json` | 实际完成步数、最终状态指标、GPU 峰值显存 |
| `dataset.npz` | 自动准备时保存的冻结数据；使用外部 prepared 文件时保留在原路径 |

恢复时检查数据哈希、模型结构、目标及损失／优化配置；不能跨 backbone 或目标恢复。schema-2 旧权重可推理，schema-1 静态权重拒绝加载。默认评估验证集；训练不会自动打开测试集。地理组划分是缺少研究 ID 时的代理，现代端点参与训练，因此不能把这些分数当作"完全未见现代资料"的空间泛化结果。

## 推理和导出

checkpoint 内包含推理所需的空间特征。只复制 `best.pt` 到另一台安装了 `sfno` 依赖的机器，即可生成地图，无需原始 Excel、现代 DAT 或物理约束文件。

```bash
norn infer --checkpoint best.pt --ages all --output predictions --device cpu
norn infer --checkpoint best.pt --ages 0 13.4 60 --output predictions \
  --formats npz dat netcdf --device cuda
```

输出地图为原生 `180×360` 网格，纬度南到北，经度 `0.5°…359.5°`，厚度单位 km：

- `thickness.npz`：`thickness_km[age,latitude,longitude]`，含年龄、坐标、同龄大陆覆盖／板界距离／变形覆盖／板块 ID 和 JSON 元数据。
- `thickness_<age>Ma.dat`：64,800 行 `lon lat thickness_km`，正厚度。
- `thickness.nc`：带坐标的 NetCDF；需要 `export` 可选依赖。
- `inference_manifest.json`：checkpoint 哈希、适用假设和点估计标识。

点位接口接受**所查询年龄的坐标**，不是未经重建的现代样品坐标：

```python
from norn_earth.inference import NornPredictor

predictor = NornPredictor("best.pt", device="cuda")
thickness = predictor.predict_points(lon=[-120, 85], lat=[40, 32], age=[13.4, 40])
maps = predictor.predict_grid([0, 30, 60])
```

也可使用 `norn predict-points --checkpoint best.pt --csv queries.csv --output predictions.csv`，输入列为 `longitude,latitude,age_ma`。

`norn animate` 使用 checkpoint 内同龄几何绘制移动大陆和板块边缘，固定色标显示预测厚度；默认播放 0→60 Ma（现代向过去）。MP4 需要系统 `ffmpeg`；也可输出 `.gif`。视频旁保存三时刻预览 PNG 与来源／色标 JSON。现代参考图不是历史标签，视频也不是优化迭代或 diffusion 去噪过程。

## 方法学概述（非默认 baseline）

默认纯 ML baseline 之外，代码保留受观测和物理关系联合约束的时空反演扩展：年龄混合似然对古厚度记录的真实年龄做边缘化；薄层近似下以时间 $\tau=60-a$ 连接厚度演化、材料运动、面积变形与净源汇（$H_2=H_1/J_A$，伸展减薄、压缩增厚）；对运动边界的控制域显式计算弱形式体积预算；过程先验包括洋壳出生厚度 $H_{\rm birth}=Q_{\rm retained}/u_{\rm full}$ 与带幅值／符号／变化约束的低维净源系数，可与网络参数联合优化。这些物理项在 `--tag physics` 下启用；无独立定量资料时不启用真实变形／源汇项，未知边界通量屏蔽整个预算。完整推导见 Git 历史中的 `docs/design/NORN_Methods_CN.pdf`。

`1° / 1 Myr` 是输出采样间距，不是已经验证的地质分辨率。动态板界距离来自同龄解析拓扑的密集边界顶点；变形覆盖来自网络轮廓，不等于内部应变或散度。历史厚度校准置信区间本版不提供，输出明确标为点估计。

## 仓库布局

```text
NORN/
├── src/norn_earth/        # 可安装的库和 CLI
│   ├── data/             # 数据准备、观测和板块重建
│   ├── geometry/         # 球面网格、可微取样和保守映射
│   ├── models/           # SFNO、函数化 Tucker、GP 与潜空间扩散
│   ├── physics/          # 物理约束、适用掩膜和低维源项
│   ├── losses/           # 统一的观测与联合目标
│   ├── evaluation/       # 指标与选模
│   ├── training/         # 两遍训练、checkpoint 与恢复
│   ├── utils/            # 哈希、清单等工具
│   ├── cli.py            # norn 命令行入口
│   ├── config.py         # 配置解析与校验
│   ├── inference.py      # 独立 checkpoint 推理
│   └── visualization.py  # 动画导出
├── pyproject.toml        # 依赖、CPU/CUDA 索引和开发工具
├── uv.lock               # 完整依赖锁定
├── .python-version       # 默认 Python 版本
├── .pre-commit-config.yaml
└── .gitignore
```

旧实验材料可通过 Git 历史追溯。本轮重构方案见 [NORN_REFACTOR_PLAN_CN.md](docs/NORN_REFACTOR_PLAN_CN.md)，可运行示例见 [generative_demo.py](examples/generative_demo.py)，验证记录见 [NORN_REFACTOR_VALIDATION_CN.md](docs/NORN_REFACTOR_VALIDATION_CN.md)，与 CRAI／SDIFT 的实现和证据核查见 [NORN_LITERATURE_AUDIT_CN.md](docs/NORN_LITERATURE_AUDIT_CN.md)。

## 连续低秩表示与条件生成（实验性）

本轮借鉴 [CRAI](https://www.nature.com/articles/s41561-020-0582-5) 的缺测验证思路及 [SDIFT](https://arxiv.org/html/2505.09284v3) 的函数化张量／时间相关潜空间思路，新增独立实现。新模型使用球面三维坐标因子、逐年代几何条件因子和低维时间核心；不使用经纬度接缝处不连续的坐标编码。当前厚度查询继续使用统一的空间／锚点插值，输出采样间距不代表已验证分辨率。

在前面的完整配置中修改以下字段即可训练；其余数据、目标、选模和导出接口一致：

```json
{
  "model": {
    "backbone": "functional_tucker", "width": 32,
    "spatial_rank": 8, "context_rank": 4,
    "head_mode": "input_residual",
    "latent_lengthscale_ma": 10.0, "latent_amplitude": 1.0,
    "latent_scale_km": 10.0
  },
  "training": {"latent_gp_weight": 0.001}
}
```

`latent_gp_weight=0` 为无核心 GP 正则的对照；正值只增加统计先验，不能解释成物理守恒。`latent_scale_km` 将无量纲核心映射到厚度量级。SFNO 仍为默认对照；新架构需要以验证数据决定是否适用。

验证配置新增 `data.group_column`（例如 CSV 中完整的 `study_id`）与 `data.test_age_window_ma: [20,30]`。前者用于按真实研究来源分组，缺省仍为地理代理；后者将与年龄窗相交的记录整条移出训练，窄年龄记录进入测试、其余进入 buffer。`test_max_age_width_ma` 默认 5 Ma。两种验证问题应分开设计：时间窗测试不自动意味着来源独立或现代资料完全留出。改变协议后必须重新 prepare，旧 prepared 数据会拒绝加载。

```bash
norn prepare --config tucker.json --output processed/tucker_dataset.npz
# tucker.json 的 data.prepared 指向上述文件
norn train --config tucker.json --tag no_physics --output outputs/tucker
norn sample-posterior --checkpoint outputs/tucker/best.pt \
  --dataset processed/tucker_dataset.npz --output outputs/tucker/gp_posterior \
  --draws 32 --chains 2 --burnin 100 --thin 2 --device cpu
norn evaluate-posterior --checkpoint outputs/tucker/best.pt \
  --dataset processed/tucker_dataset.npz \
  --samples outputs/tucker/gp_posterior/latent_samples.npz \
  --split validation --output outputs/tucker/predictive_validation.json --device cpu
```

GP 采样采用带接受拒绝步骤的 pCN，冻结拟合后的因子网络，仅用训练记录和现代端点约束核心。因为数据项按记录／现代区块归一，目标明确称作**条件广义后验**。它不包含因子网络、板块模型和代理关系的全部不确定性。`sampling_manifest.json` 记录接受率、基础 split R-hat 和先验参数；短链诊断不能证明收敛。`posterior_summary.npz` 是**北到南 Gauss 网格**，包含坐标、均值和潜在厚度分位区间；它与 `infer` 的原生网格点估计分开保存。`evaluate-posterior` 加入 Student-t 观测噪声和年龄混合后评估留出观测的预测区间，不能把这种覆盖率当作潜在真实古厚度的覆盖率。

### 独立历史集合与扩散先验

只有一段地球历史时，先用上面的 GP 方案。扩散训练需要独立的合成／已核验模拟历史。使用同一冻结解码器编码后才能合并核心，不能拼接分别训练的模型参数。首版要求所有历史共享 checkpoint 的板块几何、现代条件输入和网格；不同几何集合的条件先验学习尚未实现。

`encode-ensemble` 输入 NPZ 包含：

| 数组 | 内容 |
|---|---|
| `thickness_km`, `observed_mask` | `[history, age, latitude, longitude]`；掩膜为 0/1，未观测值可为 NaN |
| `age_ma`, `latitude`, `longitude` | 与 checkpoint 相同的年龄和北到南 Gauss 网格坐标 |
| `history_id` | 每个历史样本的唯一 ID |
| `group_id` | 独立物理过程的父 ID；同一过程的不同窗口必须同组 |
| `split` | `train`、`validation` 或 `test`；按父历史隔离 |
| `metadata` | JSON 字符串，含 `source_semantics: synthetic` 或 `verified_simulation` |

至少需要两个独立训练组和一个验证组；这是输入完整性条件，不代表这些数量足以训练可信生成先验。代码拒绝重复核心和跨划分的父历史组，标准化仅使用训练历史。

```bash
norn encode-ensemble --checkpoint outputs/tucker/best.pt \
  --fields field_ensemble.npz --output processed/latent_ensemble.npz --device cpu
norn train-prior --ensemble processed/latent_ensemble.npz \
  --output outputs/diffusion_prior.pt --lengthscale-ma 10 --amplitude 1 --device cpu
norn sample-posterior --checkpoint outputs/tucker/best.pt \
  --dataset processed/tucker_dataset.npz --prior outputs/diffusion_prior.pt \
  --output outputs/tucker/diffusion_posterior --device cpu
```

扩散模型在 GP 白化核心上学习序列去噪，再通过真实非线性年代混合似然进行近似观测引导。它不是 SDIFT 的闭式 MPDPS 复现，也不保证精确后验或已校准覆盖率。先验绑定因子权重、输入几何、网格和年代的指纹；微调因子或更换几何后必须重新编码、训练先验。

从零运行完整合成示例（无需真实古位置或观测资料）：

```bash
uv run --no-sync python examples/generative_demo.py \
  --output outputs/generative_demo --device cpu
uv run --no-sync python -m pytest -q tests
```

示例比较现代参考、SFNO、函数化 Tucker，检验整时间窗留出，并完成 GP 采样、集合编码、扩散训练、观测引导和预测评估。完整场误差之外，另保存训练查询支持掩膜及其补集上的面积加权误差；现代条件输入仍保留。它使用独立解析合成历史，不证明真实地质恢复精度或方法优越性。

## 许可

软件采用 [Apache License 2.0](LICENSE)。原始观测、外部板块模型和运行产物不随源码包分发，其许可与来源由各数据提供者决定。
