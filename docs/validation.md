# 运行与发布验证

## 0.5 双 GMT／标签版本（2026-10-10）

当前默认配置和受控开关见 [GMT／标签协议](gmt_tags.md)。本地 A40／Python 3.11 完整回归 **146 passed**；Python 3.12 CPU／kinematics／viz 环境 **145 passed, 1 skipped**（只跳过 CUDA 专属测试）；不含 PyTorch／pygplates 的 Python 3.9 基础环境 **96 passed, 4 skipped**（跳过可选依赖模块）。Ruff、format、shell 语法、`git diff --check`、离线锁文件检查及 wheel/sdist 构建通过。新增 CI 的 CPU／kinematics 任务用于真实执行制造数据的 GMT 和标签回归；本地通过不替代远端 CI。

两个源只使用 0–60 Ma 的 **122 文件**，逐龄核对完整集合、**5,467 要素**及其板块 ID、顶点、边中点；顶点注册最大误差 **1.42029e-8 km**（容差 0.001 km）。原框架为 Zahirovic2022／ANCHOR=0；较新的旋转版本不能凭名称替代。2 个零面积初生网络保留边界、区域覆盖为零，并记录在逐龄清单中。

真实冻结输入为 `61×8×180×360`。0/60 Ma 的大陆掩膜、板界距离、网络覆盖、板块 ID 分别有 **15,933／64,668／13,057／27,640** 个内部网格点改变。关闭缺失板块 ID 的默认恒等旋转回退后，11 条记录因年龄节点缺少线路整条剔除；地理划分规则不改。新有效记录 **31,897**：训练 **15,741**、验证 **1,848**、原地理组测试 **14,308**。这是定位支持筛查，不是按误差选择样本。

```text
processed/gmt_dataset.npz SHA256
fedc54bc813a25e17bf852fda8625874157a8836d71d0df86620b26e7fa855b4
```

A40 上两标签各执行 **10 步 smoke**（完整计划仍为 500 步、`completed=false`），确认共享数据、503,809 个模型参数、验证 MAE 选择及独立来源冻结；并非正式性能对比。`no_physics` 无物理状态／loss 或时间平滑计算；`physics` 实际计算 17 个轨迹、7 个预算区域的明确刚性无源假设。两版均导出 61 张同龄地图及 0→60 Ma 视频，独立导出校验通过。10 步运行不证明收敛、古厚度精度或物理项增益，测试集没有重新评分。

证据在工作区 `outputs/gmt_audit/` 的 `dataset_audit.json`、`frame_registration.json`、测试／构建日志及两个标签子目录；原资料、历史产物不覆盖，也不提交原始数据／权重。以下 0.4/0.3 数字是历史 GPML 结果，不能当作新 GMT 训练成绩。

## 0.4 纯 ML baseline（历史 Müller2019 GPML）

默认训练协议见[纯 ML baseline](ml_baseline.md)，主图为[无物理一致性分支的 ML 框架](design/figures/norn_ml_baseline.svg)。全套测试 **117 passed**，覆盖物理项／时间平滑配置拒绝、完全不实例化物理模块、Huber 每记录只贡献一次、测试标签不能改变训练 loss、新 backbone 两遍梯度等价、纯数据 checkpoint／resume／独立推理和验证 MAE 选择。

B0 / control 已在 A40 完成 500 步：验证 MAE **9.15851 km**，验证 NLL **3.89956**，最终现代面积 RMSE **2.03010 km**；503,809 参数，耗时 697.96 秒。它与历史 0.3 相同数据／旧 backbone／原数据 loss，但不含物理项，并按验证 MAE 选模。B3 新 backbone + 辅助 loss 同样完成 500 步，best 验证 MAE **9.33536 km**、NLL **3.90783**，348,993 参数，耗时 863.62 秒，峰值显存 11976.47 MiB。其最后一步 MAE 为 10.00009 km，不能用 last 冒充 best；按验证侧选择 B0 为正式默认基线。B3 的参数更少，但耗时／显存更大，且验证侧未改善。

B0/B3 对比放在 `outputs/ml_baselines/comparison.json`，只读取验证集，不重复打开已评分测试集。训练集中位数常数基线验证 MAE 9.24388 km，B0 也仅小幅优于该简单基线。单因素 B1/B2 尚未执行，不声称已完成完整消融。

选定的纯 ML B0 已导出 61 张 NPZ／DAT／NetCDF 地图和 0→60 Ma 视频；数值／同龄几何／独立推理核对通过，视频为 1200×600、6 fps、61 帧。推理元数据只包含数据目标模式，没有物理来源字段；证据在 `outputs/ml_control_a40/export_validation.json`、`video_validation.json` 及 `evolution_0_60Ma.mp4`。

以下 0.3 为历史受约束实验，不是当前默认基线，也不能与换了 backbone/loss 的 0.4 直接比较来归因物理项增益。

## 0.3 动态几何版本

新版实现和运行说明见 [动态重建与视频](dynamic.md)。当前架构已通过旋转板块、动态图层两遍梯度、独立几何／GIF 导出和年龄边界回归；本地完整测试 **103 passed**，格式／静态检查和 wheel／源码构建通过。真实动态图层为 `61×8×180×360`，0–60 Ma 全部解析完成；0 与 60 Ma 相比，15,722 个大陆掩膜格点和 25,783 个板块 ID 格点改变。数据划分保持不变，训练和选模期间没有评分测试集；此后按用户要求，对冻结的 `best.pt` 打开了地理组测试集，详见下文。

真实数据已在 A40 完成 **500 步**动态版本重训，耗时 729.21 秒；联合目标 `13.52038 → 4.89782`，现代面积 RMSE `1.98404 km`，验证集年龄边缘均值 MAE `9.01694 km`，峰值分配显存 `5937.41 MiB`。这些是当前点估计运行指标，不是独立地学精度证明。

同一 checkpoint 已导出 61 张地图和 **1200×600、6 fps、61 帧、10.17 秒**的 0→60 Ma 视频，使用固定 0–80 km 色标和同龄大陆／板块边缘。系统 ffmpeg 没有可用的 x264/OpenH264 编码器，视频模块经实际帧探测后选用了 MPEG-4 编码；MP4 和 GIF 路径均有回归测试。

证据在工作区 `outputs/dynamic_a40/`：`training_summary.json`、`validation.json`、`inference/`、`evolution_0_60Ma.mp4`、三时刻预览 PNG、`video_validation.json` 和 `export_validation.json`。NPZ／DAT／NetCDF、同龄几何及同一 GPU 算子核对通过；默认 CUDA TF32 卷积与 CPU 的所查点最大差为 `0.002209 km`，禁用 TF32 的严格 float32 对照差为 `3.8147e-6 km`，按原容差单独核验而非放宽容差。下面的 0.2 分数来自历史静态输入版，不能作为动态版结果或精度证据。

### 冻结模型的地理组测试（用户授权后打开）

`outputs/dynamic_a40/test.json` 记录当前模型在 14,308 条有效测试记录上的年龄边缘均值 MAE **9.93516 km**、NLL **3.98932**。逐记录核对的 RMSE 为 **12.52520 km**，中位绝对误差 **8.08678 km**；约 **42.0%** 的记录绝对误差超过 10 km。对照“沿材料历史保持现代厚度不变”的基线 MAE **12.36605 km**，训练集中位数常数基线 **10.27618 km**。模型相对现代基线有改善，但相对常数基线仅小幅改善，不能据此宣称强时空泛化。

按**域内年龄分布均值**分箱（不是原始代表年龄，也不是精确逐年代观测），0–10／10–20／20–30／30–40／40–50／50–60 Ma 的测试 MAE 分别为 **9.59／10.91／9.69／8.98／9.16／16.83 km**；50–60 Ma 仅 157 条记录，明显较弱。原始 6,008 条地质挑战并集 MAE 为 **10.31902 km**，挑战区域存在重叠，不能将各区条数直接相加。

详细区域／点年龄分箱／基线和逐记录预测在 `test_breakdown.json`、`test_predictions.csv`；评分脚本保存在同一运行目录。没有修改权重或据此调参。本次是允许现代端点的 P-B 地理代理组留出，不是禁止现代信息的 P-A 或整个年代窗口留出的 P-C；记录仍是 `unverified_proxy`。之后如果利用这些测试结果选择模型，就不能再将同一集合称为未触碰的最终测试。

## 历史 0.2 运行记录

以下 500 步训练与原安装包记录对应 0.2.0；0.2.1 的 uv 环境复核和 CI 修复记录位于文末。

验证日期：2026-10-08。设备：**NVIDIA A40**。Python 3.9、PyTorch `2.5.1+cu121`、CUDA 12.1、torch-harmonics 0.8.0、pygplates 1.0.0。依赖版本见 [环境管理](environment.md)。本文记录真实执行结果，不是预计指标。

## 真实数据训练

使用当时的 A40 配置（当前 `configs/a40.json` 已切换 GMT，不是该历史配置）：180×360 Gauss 网格、宽度 32、6 个 SFNO 模块、61 个年代锚点、503,809 个可训练参数。准备后的 31,908 条有效记录中，15,745 条用于训练，1,855 条用于验证，14,308 条留作测试。每个年龄节点均重建古位置；19,253 条原始记录包含负经度，周期处理已修正。

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

以下仅记录当时的 0.2 命令；需检出对应历史版本才能复现，不能用当前 GMT 配置覆盖历史产物：

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
