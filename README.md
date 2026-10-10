# NORN｜诺恩

**年代不确定性感知、动态几何条件化的全球古地壳厚度反演；当前以纯 ML baseline 为主。**

NORN 用年代条件球面神经算子表示 0–60 Ma 的厚度历史。**每个年代绑定自己的板块几何输入 `X(a)`：网格固定，大陆和板界移动，厚度由模型预测。**0.5 默认 `no_physics` 仅使用观测和现代监督，不导入物理目标或计算时间平滑；`--tag physics` 只增加显式一致性项。两版都使用同一份逐年代板块／变形区 GMT 输入，不是只加时间 embedding。

![NORN 纯 ML baseline：逐年代输入、SFNO 候选、数据拟合损失和验证 MAE 选模](docs/design/figures/norn_ml_baseline.svg)

当前开关和参考框架校验见 [GMT 与标签协议](docs/gmt_tags.md)，数据 loss／历史消融见 [纯 ML baseline](docs/ml_baseline.md)，几何和视频见 [动态重建](docs/dynamic.md)。[历史方法说明](docs/method.md)、[中文 Methods PDF](docs/design/NORN_Methods_CN.pdf) 和 [LaTeX](docs/design/NORN_Methods_CN.tex) 中的物理扩展保留追溯，不属于默认基线目标。

> 当前版本：**0.5.0，研究软件 Beta**。0.3/0.4 的 schema-2 权重仍可独立推理，但不能冒充 GMT 重训练结果或跨物理标签恢复。0.2 静态数据／权重不兼容。资料仍为 `unverified_proxy`；输出点估计，不提供已校准置信区间。

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

`uv sync --locked --no-dev` 只安装数据准备所需的基础依赖，不安装 PyTorch。真实古位置重建需要 `--extra kinematics`；绘图可选 `--extra viz`。开发测试使用默认 `dev` 依赖组。不要同时启用 `cpu` 与 `cuda`，也不要使用 `--all-extras`。

无需激活环境时，已同步的环境可通过 `uv run --no-sync norn ...` 使用。完整的环境切换、依赖更新、开发与构建步骤见 [环境管理](docs/environment.md)。

## 历史受约束计算示例（不是纯 ML baseline）

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

经度可使用 `[-180,180]` 或 `[0,360)`；单位分别为度、Ma 和 km。也可以直接读取本项目原始中文列名的 XLSX。现代数据使用 `180×360` 原生格点的三列 DAT，按南到北、经度从 `0.5°` 到 `359.5°` 排列，第三列为历史数据约定的**负编码厚度**；内部和导出结果均使用正厚度。详见 [数据格式与来源](docs/data.md)。

先复制 [通用配置](configs/reconstruction.json)，填写观测、现代 DAT、两个 GMT 目录和匹配的旋转／静态分区／大陆／原始拓扑路径。`geometry_mode=dynamic`、`geometry_source=gmt` 在每个锚点读取同龄板块和变形网络，不再二次旋转 GMT。只有明确的统计／合成对照允许 `static`；历史 GPML 配置仍可独立使用。

```bash
norn prepare --config configs/reconstruction.json --output processed/gmt_dataset.npz
norn train --config configs/reconstruction.json --tag no_physics --dry-run
norn train --config configs/reconstruction.json --tag no_physics
```

`coordinate_mode: paleo` 会在每个真实年龄节点通过 pygplates 重建古位置。无法定位的记录整条剔除或报错，由 `position_failure` 控制；不会无提示地退回现代坐标。年龄超出重建域的部分会被条件化并记录保留质量，域外点年龄不参与训练，推理禁止外推。

本工作区的 A40 复现实验使用 [configs/a40.json](configs/a40.json)：

```bash
# 外部模型哈希固定；原始 GMT 和观测由用户提供
uv run --no-sync python scripts/fetch_gmt_model.py
norn prepare --config configs/a40.json --output processed/gmt_dataset.npz
norn train --config configs/a40.json --tag no_physics

# 可选物理对照：同模型、同 loss、同数据、同种子／计划／选模
norn prepare-rigid-priors --config configs/a40.json \
  --dataset processed/gmt_dataset.npz --output processed/gmt_rigid_priors.npz
norn train --config configs/a40.json --tag physics \
  --physics-constraints processed/gmt_rigid_priors.npz

norn infer --checkpoint outputs/gmt_a40/no_physics/best.pt --ages all \
  --output outputs/gmt_a40/no_physics/inference --device cuda --formats npz dat netcdf
norn animate --checkpoint outputs/gmt_a40/no_physics/best.pt --ages all \
  --output outputs/gmt_a40/no_physics/evolution_0_60Ma.mp4 --device cuda --fps 6 --vmin 0 --vmax 80
```

默认配置采用验证侧更好的纯数据 SFNO 对照：180×360 Gauss 网格、6 个模块、32 通道、`lmax=mmax=32`，目标仅为年龄边缘 NLL + 现代数据 NLL，按验证 MAE 选模，不读取物理先验。

两个 GMT 源实际声明 **Zahirovic2022、ANCHOR=0**，不能混用 Müller2019 或更新后偏移的绝对参考框架。准备过程在全部年龄逐要素核对原始 GPML 的 ID、顶点和边中点（1 m 容差），保留多个外环／孔洞及零面积裂谷的边界，并冻结来源哈希。

历史 gated／Huber 组合的验证 MAE **9.34 km**，旧 SFNO 对照 **9.16 km**，未改善。这是 0.4 Müller2019 GPML 结果，配置 `a40_ml_*` 和运行脚本保留，**不是 0.5 GMT 的正式训练分数**。新 GMT 计划沿用选定的 B0 网络／数据 loss。`configs/a40_physics.json` 是同条件的可选物理配置；旧受约束配置保留为 `a40_physics_muller2019_legacy.json`。

## 断点恢复和模型选择

```bash
# 有限步数检查，仍按配置的完整训练计划计算学习率
norn train --config configs/a40.json --output outputs/check --stop-after 10

# 恢复同一训练计划
norn train --config configs/a40.json --output outputs/check \
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

恢复时检查数据哈希、模型结构、目标及损失／优化配置；不能跨 backbone 或目标恢复。schema-2 旧权重可推理，schema-1 静态权重拒绝加载。默认评估验证集；训练不会自动打开测试集。地理组划分是缺少研究 ID 时的代理，现代端点参与训练，因此不能把这些分数当作“完全未见现代资料”的空间泛化结果。

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

也可使用 `norn predict-points --checkpoint best.pt --csv queries.csv --output predictions.csv`，输入列为 `longitude,latitude,age_ma`。更多说明见 [推理接口](docs/inference.md)。

`norn animate` 使用 checkpoint 内同龄几何绘制移动大陆和板块边缘，固定色标显示预测厚度；默认播放 0→60 Ma（现代向过去）。MP4 需要系统 `ffmpeg`；也可输出 `.gif`。视频旁保存三时刻预览 PNG 与来源／色标 JSON。现代参考图不是历史标签，视频也不是优化迭代或 diffusion 去噪过程。

## 历史受约束方法学（非默认 baseline）

本节为保留的物理扩展设计，不进入当前纯 ML baseline；当前实际目标见 [纯 ML 文档](docs/ml_baseline.md)。

本节依据原项目 `methods/NORN_Methods_CN.tex` 概述完整方法设计。含符号表、编号公式与完整推导的正文见 [中文 Methods](docs/design/NORN_Methods_CN.pdf)；发布代码的计算细节和源码对应关系见 [方法实现说明](docs/method.md)。

NORN 将全球古地壳厚度重建表述为受观测和物理关系约束的时空反演。框架图 A–C 使用逐年代几何条件与年代条件 SFNO 表示正厚度场，卷积编码器、球谐谱分支和局部卷积分支共同提取空间结构，年龄通过 MLP 与 FiLM 调节隐藏通道：

$$H_\theta(x,a)=\mathrm{softplus}(F_\theta(X(a),a)(x))+\epsilon.$$

0–60 Ma 的 61 个锚点共享网络参数，连续年龄在相邻锚点之间线性插值。厚度历史由所有证据联合拟合；更换观测集合通常需要重新优化。

框架图 D 对古厚度记录的真实年龄进行边缘化，在每个年龄节点查询重建古位置，而不是把年龄区间复制成多条独立标签：

$$p(y_i\mid\theta)\approx\sum_k w_{ik}\,
t_\nu\!\left(y_i;H_\theta(x_i(a_{ik}),a_{ik}),\sigma_i\right),
\qquad \sum_k w_{ik}=1.$$

每条记录贡献一个混合似然项。现代数据通过空间块平均的高斯负对数似然约束 0 Ma；误差尺度按记录或来源设定。

框架图 E 在相同参考密度的薄层近似下，采用向现代推进的时间 $\tau=60-a$，连接厚度演化、材料运动、面积变形与净源汇：

$$\frac{\partial H}{\partial\tau}+\nabla_s\cdot(H\mathbf v)=q,
\qquad q=q_{\rm mag}-q_{\rm loss}.$$

材料轨迹一致性通过以下残差进入反演；无源条件下的面积变形关系为 $H_2=H_1/J_A$，伸展导致减薄，压缩导致增厚：

$$r_{\rm traj}=H(x_2,\tau_2)-H(x_1,\tau_1)
-\int_{\tau_1}^{\tau_2}\left(q-H\nabla_s\cdot\mathbf v\right)\,d\tau.$$

对边界以 $\mathbf v_b$ 运动的控制域，弱形式体积预算显式计算相对边界通量与内部净源，不假定全球活动地壳总体积恒定：

$$\frac{dV_\Omega}{d\tau}=-\oint_{\partial\Omega}
H(\mathbf v-\mathbf v_b)\cdot\mathbf n\,dl+\int_\Omega q\,dA,
\qquad V_\Omega=\int_\Omega H\,dA.$$

过程先验包括洋壳出生厚度 $H_{\rm birth}=Q_{\rm retained}/u_{\rm full}$ 与低维净源表示 $q_{\rm net}(x,\tau)=\sum_j c_j(\tau)\psi_j(x)$。系数受到幅值、符号和变化约束；当前软件采用外部指定的时空基函数和有界系数，并支持完整协方差先验。相关约束共享证据预算，未知重要通量时屏蔽完整预算因子。

框架图 F 将数据、轨迹、区域预算和过程先验写入同一优化目标，联合更新网络参数与受限源系数：

$$\mathcal L=\lambda_o\mathcal L_{age}+\lambda_0\mathcal L_{modern}
+\lambda_k\mathcal L_{traj}+\lambda_v\mathcal L_{budget}
+\lambda_p\mathcal L_{prior}+\lambda_r\mathcal R.$$

训练使用两遍梯度重计算与 AdamW，在两遍之间保持参数固定，并记录原始联合目标。完整方法不采用自由逐像素噪声或源项头。

## 当前实现范围

| 功能 | 软件实现 | 当前真实数据配置 |
|---|---|---|
| 年代条件 SFNO、正厚度、连续年龄查询 | 已接入统一训练与推理 | 启用 |
| 年龄混合似然、现代块平均 NLL | 已实现 | 启用 |
| 年龄节点的板块重建古位置 | 已实现 | 启用 |
| 材料轨迹与弱形式预算 | 可微损失，支持显式有效域 | 启用声明假设的刚性板内先验 |
| 变形与源项 | 通过轨迹散度及源基函数进入同一方程 | 缺独立资料，未启用定量变形／源汇 |
| 洋壳出生 `Q_retained/u_full` 先验 | 已实现，合成例子验证 | 缺独立生产率资料，未启用 |
| 有界低维源项联合优化 | 已实现，合成例子验证 | 未估计真实岩浆生产或损失 |
| 历史厚度校准置信区间 | 本版不提供 | 输出明确标为点估计 |

`1° / 1 Myr` 是输出采样间距，不是已经验证的地质分辨率。动态板界距离来自同龄解析拓扑的密集边界顶点；变形覆盖来自网络轮廓，不等于内部应变或散度。无独立定量资料时不启用真实变形／源汇项，未知边界通量屏蔽整个预算，不能用零填充。

## 验证与开发

```bash
uv sync --locked --extra cpu --extra export
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests scripts
uv run --no-sync ruff format --check src tests scripts
uv build --no-sources
```

测试包含周期经度、真实年龄插值、球面网格保守映射、完整目标的两遍梯度等价性、正向物理时间与源汇符号、未知通量屏蔽、证据重复计数、有界源项、断点恢复和离开原数据后的独立推理。CI 分别运行基础依赖和 CPU SFNO 环境。本地 A40 的实际配置、结果与产物见 [验证记录](docs/validation.md)。

历史 **0.2 静态输入版**曾在本地 A40 完成真实数据 **500 步训练**，联合目标 `13.52 → 5.01`，峰值显存约 **1.51 GiB**；同一 checkpoint 成功导出 **61 张 180×360 地图**，NPZ／DAT／NetCDF 数值核对通过。回归测试覆盖核心数值与完整训练／推理流程。这些结果验证软件运行与计算路径，地学精度仍需独立资料检验。

## 仓库布局

```text
norn/
├── src/norn_earth/        # 可安装的库和 CLI
│   ├── data/             # 数据准备、观测和板块重建
│   ├── geometry/         # 球面网格、可微取样和保守映射
│   ├── models/           # SFNO 与统计基线
│   ├── physics/          # 物理约束、适用掩膜和低维源项
│   ├── losses/           # 统一的观测与联合目标
│   └── training/         # 两遍训练、checkpoint 与恢复
├── configs/              # 通用、A40 和统计重建配置
├── scripts/              # 示例、A40 复现与导出核对脚本
├── tests/                # 单元、数值回归及端到端测试
├── docs/                 # 方法、数据、接口及验证文档
├── pyproject.toml        # 依赖、CPU/CUDA 索引和开发工具
├── uv.lock               # 完整依赖锁定
├── .python-version       # 默认 Python 版本
└── .github/workflows/    # 测试和构建 CI
```

目录组织参考 [OlmoEarth](https://github.com/allenai/olmoearth_pretrain) 的库、脚本、文档和测试分工；NORN 没有使用其模型权重或训练实现。当前动态框架图与保留的原版 Methods 见 [框架图与 Methods](docs/design/README.md)。旧实验脚本、旧配置及重复入口已从当前源码删除，可通过 Git 历史追溯。旧挑战集结果不作为本版精度证据。

## 贡献与许可

开发约定见 [CONTRIBUTING.md](CONTRIBUTING.md)，版本变更见 [CHANGELOG.md](CHANGELOG.md)。软件采用 [Apache License 2.0](LICENSE)。原始观测、外部板块模型和运行产物不随源码包分发，其许可与来源由各数据提供者决定。
