# 运行与发布验证

以下 500 步训练与原安装包记录对应 0.2.0；0.2.1 的 uv 环境复核和 CI 修复记录位于文末。

验证日期：2026-10-08。设备：**NVIDIA A40**。Python 3.9、PyTorch `2.5.1+cu121`、CUDA 12.1、torch-harmonics 0.8.0、pygplates 1.0.0。依赖版本见 [环境管理](environment.md)。本文记录真实执行结果，不是预计指标。

## 真实数据训练

使用 [configs/a40.json](../configs/a40.json)：180×360 Gauss 网格、宽度 32、6 个 SFNO 模块、61 个年代锚点、503,809 个可训练参数。准备后的 31,908 条有效记录中，15,745 条用于训练，1,855 条用于验证，14,308 条留作测试。每个年龄节点均重建古位置；19,253 条原始记录包含负经度，周期处理已修正。

测试侧按完整地理组隔离，不能只按原挑战清单逐条移除。此次运行没有计算测试集分数。现代端点参与训练；划分中的地理组只是缺少研究 ID 时的代理。

实际先训练 2 步保存 checkpoint，再从 `last.pt` 恢复至 **500/500 步**。恢复后的 498 步耗时 736.72 秒。结果如下：

| 指标 | 实测值 |
|---|---:|
| 初始联合目标 | 13.520790 |
| 最终联合目标 | 5.010956 |
| 最终古观测 NLL | 3.877399 |
| 最终现代端点 NLL | 2.257296 |
| 现代端点面积加权 RMSE | 2.195519 km |
| 验证集年龄边缘均值 MAE | 8.727839 km |
| 验证集 NLL | 3.873664 |
| 最终加权物理损失 | 0.004908 |
| 峰值 CUDA 分配显存 | 1,549.19 MiB，约 1.51 GiB |

`best.pt` 在第 500 步被选中。选择分数是验证古观测 NLL 加上配置权重下的现代 NLL。各分量的单位和权重不同，联合目标不能直接作为 km 误差解释；500 步也不构成收敛证明。

真实数据物理项启用了 18 条材料轨迹和 16 个弱预算因子，按板块给轨迹和预算分配互斥证据角色。这些是**模型导出的刚性板内无净源先验假设**，不是独立观测证实的无变形、无岩浆作用区。真实数据未启用定量变形、出生生产率或可学习净源项。观测仍标为 `unverified_proxy`，历史置信区间未经校准。

## 推理和数值核对

使用同一 `best.pt` 在 A40 上导出 0–60 Ma 的全部 61 张地图，每张 180×360。输出共 3,952,800 个厚度值，均有限且为正，范围 **1.884834–78.945618 km**。

- 逐一读取了全部 61 个 DAT，检查每个文件的 64,800 行坐标、纬度顺序和厚度。与 NPZ 的最大舍入差为 `5.0e-7 km`。
- NetCDF 与 NPZ 的厚度及坐标逐元素完全一致。
- 重新调用 checkpoint 的同一地图算子，结果与导出 NPZ 完全一致。
- 验证 13.4 Ma 小数年龄查询、`−120° = 240°` 周期经度，以及训练与推理取样的逐元素一致性。
- 将真实 checkpoint 单独复制到临时目录，使用 CPU 推理；所核对点与 A40 的最大差异为 `4.8828125e-4 km`，通过 `rtol=2e-5, atol=2e-5` 的浮点容差。CPU 与 GPU 不保证位级一致。

运行产物保存在工作区 `outputs/release_a40/`：训练日志、运行清单、`best.pt`、`last.pt`、验证分数，以及 `inference/thickness.npz`、`thickness.nc`、61 个 DAT 和推理清单。原始数据与运行产物不随源码发行。

关键 SHA-256：

```text
dataset.npz
4bd7ec61a3d642d9f72fbd8b724258ccac80d7665886c27d754f4a9df8f22f8e
best.pt
883a7041a3a71eb7c655a839d5499c571e098a3da43d4991cb6eeed4c7f6609c
last.pt
d1aeefedf99018134a54308ebcc5141f2c5fd4a79ad404af2cc80a4684f54742
```

可在具备相同工作区数据时复现：

```bash
norn prepare --config configs/a40.json --output processed/release_dataset.npz
norn prepare-rigid-priors --config configs/a40.json \
  --dataset processed/release_dataset.npz --output processed/release_rigid_priors.npz
norn train --config configs/a40.json --stop-after 2
norn train --config configs/a40.json --resume outputs/release_a40/last.pt
norn infer --checkpoint outputs/release_a40/best.pt --ages all \
  --output outputs/release_a40/inference --device cuda --formats npz dat netcdf
norn evaluate --checkpoint outputs/release_a40/best.pt \
  --dataset processed/release_dataset.npz --split validation \
  --output outputs/release_a40/validation.json --device cuda
uv run --no-sync python scripts/validate_exports.py --run outputs/release_a40 --device cuda
```

新训练需要空输出目录；已有运行请使用 `--resume` 或指定新的 `--output`。

## 已知真值的完整物理示例

合成例子采用静止材料、非均匀初始厚度和已知净源 `q=0.02 km/Myr`，同时启用轨迹、弱预算、出生厚度与有界净源联合优化。A40 上 100 步耗时 5.09 秒，峰值显存 38.18 MiB。

最终联合目标从 17.609287 降至 −0.276919，现代面积 RMSE 为 0.056835 km；验证选择的第 90 步 checkpoint 的验证 MAE 为 0.130177 km。负 NLL 来自概率密度及其归一化常数，不是负厚度。第 100 步学得净源系数为 0.014158 km/Myr；因此这次短运行证明联合优化路径可以运行，并不证明有限观测下源项已被精确识别。

该例也完成全部 13 个年代的 NPZ、DAT 和 NetCDF 导出；工作区结果在 `outputs/release_demo/run/`。无需真实数据即可从 README 的 `norn demo` 命令复现。

## 回归测试

0.2.0 本地完整测试：**93 passed**。包括：

- 球面几何、年龄积分、似然、数据划分和保守重网格；
- 完整联合目标的直接反传与两遍反传，在 CPU 和 A40 上的相对梯度误差均小于 `1e-5`，包含源系数梯度；
- 已知真值场的正向时间、源项和体积预算符号；未知通量屏蔽及证据重复计数拒绝；
- 同一计划断点恢复与连续训练的模型和物理参数一致；0.2.0 的该次环境中达到位级相同。跨 CPU 构建与并行归约不保证末位浮点数相同，当前回归使用 `rtol=1e-6, atol=1e-8`；
- 冻结数据的误差尺度和恢复训练的目标权重不允许静默改变；
- 删除原始观测与物理文件后，仅凭复制的 checkpoint 完成推理和导出。

该次验证的 `ruff check` 和 `ruff format --check` 通过，尚未执行远端 CI；本地 A40 证据不替代其他平台的兼容性验证。0.2.1 改用 uv 锁定环境，补充记录见下文。

## 安装包验证

`python -m build` 成功生成 `dist/norn_earth-0.2.0-py3-none-any.whl` 与 `dist/norn_earth-0.2.0.tar.gz`。源码包包含配置、方法资料、文档、发布脚本、测试、CI 和许可证；检查确认未打入原始数据、外部板块模型、checkpoint、运行目录或虚拟环境。wheel 仅包含运行库、入口与包元数据。

在全新、**未安装 PyTorch** 的 Python 3.9 环境安装 wheel 后，从 `/tmp` 成功运行版本、CLI 帮助、示例生成、数据准备和配置 dry-run；基础测试结果为 **80 passed, 1 skipped**，跳过的是需要 PyTorch 的发布集成测试模块。

随后在 A40 环境用构建出的 wheel 替换 editable 安装，从 `/tmp` 验证实际导入来自 `site-packages`，成功生成示例、训练 2 步、恢复至 4 步，以及导出合成与真实 checkpoint 的 0、13.4、60 Ma 地图。三个输出格式均成功。安装验证后将开发工作区恢复为 editable 安装，便于继续开发。

上述验证没有向 PyPI 或其他外部平台发布；这里完成的是可发布产物与本地运行验证。

## 0.2.1 uv 环境与 CI 修复

2026-10-08 改为 `pyproject.toml`、`uv.lock` 和 `.python-version` 管理环境。默认 Python 3.11，CPU 与 CUDA 12.1 互斥；精确安装命令见 [环境管理](environment.md)。工作区 `.venv` 已实际同步为 Python 3.11.14、PyTorch 2.5.1+cu121，含 pygplates、NetCDF 导出、绘图及开发工具。

本地按锁文件创建独立环境，执行结果如下：

| 环境 | 测试结果 |
|---|---|
| Python 3.9，基础依赖，无 PyTorch | 82 passed，1 个需要 PyTorch 的模块跳过 |
| Python 3.12，基础依赖，无 PyTorch | 82 passed，1 个需要 PyTorch 的模块跳过 |
| Python 3.9，CPU SFNO，干净源码副本 | 94 passed，1 个 CUDA 测试跳过 |
| Python 3.12，CPU SFNO | 94 passed，1 个 CUDA 测试跳过 |
| Python 3.11，CUDA SFNO，NVIDIA A40 | 95 passed |

干净副本仅含待提交的源码文件，没有原始数据、`processed/`、checkpoint 或运行结果。CI 失败有两处已复现原因：球面多边形首尾重复点形成零长度边，造成极区判断随浮点舍入变化；测试还曾硬编码读取本地 `processed/test_window_registry.json`；现在面积计算先去除连续重复点和重复闭合点，并正确处理反向环；新增闭合、重复和反向极区回归。注册表和现代厚度 DAT 均在测试临时目录生成。恢复训练的比较使用严格 float32 容差，同时核对步数与随机状态，避免并行 CPU 归约末位差异造成误报。

新 CUDA 环境重新完成 100 步完整物理合成训练，初始／最终目标为 17.609287／−0.276919，现代面积 RMSE 为 0.056835 km，峰值分配显存 38.18 MiB。全部 13 个年代通过 NPZ、DAT、NetCDF、周期经度、小数年龄和独立 CPU checkpoint 推理核对。原 500 步真实 checkpoint 的全部 61 张地图也再次通过导出核对；新环境直接调用算子与原 NPZ 的最大差异为 0。

本地证据在 `outputs/uv_a40/`，不提交 GitHub。CI 使用同一锁文件，运行格式检查、Python 3.9／3.12 的四种测试组合、构建和源码目录外的 wheel 安装检查；每次提交的远端状态见 [GitHub Actions](https://github.com/zhaorui-bi/NORN/actions/workflows/ci.yml)。

0.2.1 删除历史脚本、旧配置、重复 CLI 包装和旧依赖清单。`scripts/` 只保留 `demo.sh`、`a40.sh`、`validate_exports.py`；统一使用 `norn` CLI。0.2.1 wheel 和源码包已重新构建并检查不含训练数据、模型权重或历史目录。
