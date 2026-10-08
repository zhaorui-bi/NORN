# NORN｜诺恩

**NORN：年代不确定性感知、运动学约束的全球古地壳厚度重建软件。**

设计规范：[`../最终研究方案_NORN.md`](../最终研究方案_NORN.md)（v1.2）＋
[`../研究执行协议_NORN.json`](../研究执行协议_NORN.json)。

## 当前状态（第四轮：真实基线 B1/B2 + C1 合成消融 + ensemble 区间诊断完成）

**真实数据 P-C 40–50 冻结留出（一次性评分，n_test=1211）：**

| 模型 | 点 MAE | 混合 NLL |
|---|---|---|
| **B2 CatBoost（点年龄特征）** | **9.36 km** | **3.895** |
| B0 滚动中位 | 10.22 | 3.999 |
| B7 年代感知变分（lmax=12） | 10.62 | 4.049 |
| B1 全局 RBF（长度尺度训练侧选择 1600 km） | 12.79 | 4.322 |

发现：代理目录的**现代位置空间结构**是主导信号（B2 胜出；B1/B7 全局参数化进不去）；B2 两种年龄特征变体结果几乎相同→树几乎不用年龄特征。这直接支持设计判断：在此信号结构上，方法贡献需在可控合成场景证明。

**C1 合成消融（宽年代区间，真值已知，6 实例）：** mixture 2.653 vs copied 2.613 vs point 2.652 km——差异 ~1.5%（噪声量级）。当前基准真值时变强度不足以支撑 C1 主张；论文级需要更强时变场景+更多实例（如实记录，不虚构优势）。

**B7 ensemble 区间诊断（3 成员，训练侧自助）：** MAE 10.67 km（与单拟合一致）；**覆盖失败：cov50=1.0%、cov90=1.8%**（区间宽仅 0.6/1.0 km）——成员几乎相同。实证印证设计 §13.1：seed/自助 ensemble **远不是**校准区间；必须加年龄/源尺度/过程分支。作为方法证据写入报告。

产物：`outputs/pilot_stat/{baselines_b1_b2.json, b7_ensemble_intervals.json}`、`outputs/benchmark_runs/{c1_ablation_synthetic.json, b7_synthetic_smoke.json}`。

## 第三轮补记（G1 关闭 + 环境就绪 + Stat 试点）

- **G1 PASS**（pygplates 1.0.0）：t=0 严格恒等；XLSX vs Müller2019 中位 12.5→104 km（5→60 Ma）；纯 Python 解析器 vs pygplates 中位 0.000 km。
- **训练环境**（torch 2.8.0+cpu / torch-harmonics 0.8.0）：SFNO 前向 + 两遍梯度等价 3.9e−8 ≤ 1e−5。
- **Stat 试点**（B7, P-C 40–50）：B7 MAE 10.62 vs B0 滚动 10.22 km；train corr 0.47 → test 0.27；场几乎不随时间变化（现代端点主导）——与 G0 语义阻断自洽。
- 物理-主模型训练与 GPU/SLURM 仍被门禁拦截；G0 仍 PARTIAL（待提供方确认）。

| 组件 | 状态 | 结果 |
|---|---|---|
| 全部软件基础件（78 项单测） | ✅ | 见前两轮 |
| **G1 拓扑级验证**（pygplates 1.0.0） | ✅ **PASS** | t=0 严格恒等；XLSX vs Müller2019：中位 12.5→104 km（5→60 Ma）；纯 Python 解析器 vs pygplates 中位 0.000 km |
| **训练环境**（torch 2.8.0+cpu / torch-harmonics 0.8.0） | ✅ | SFNO 前向 + **两遍梯度等价 3.9e−8 ≤ 1e−5 门槛通过** |
| **真实数据 Stat 试点**（B7, P-C 40–50 留出，n_test=1211） | ✅ 已运行 | **B7 未胜强 B0**：MAE 10.62 vs 10.22 km（滚动中位）；NLL 4.049 vs 3.999；train corr 0.47 → test 0.27 |
| 物理-主模型训练 | 🚫 仍门禁 | G0 仍 PARTIAL（代理语义未确认），按设计不得开物理损失 |

**试点结论（诚实）**：管线端到端可用；但在“代理语义未验证 + 有效噪声 ~8.9 km”的声明假设下，年代感知全局低阶场在时间留出上不优于稳健基线。场几乎不随时间变化（现代端点主导）——与方案预警一致：G0 语义确认前不应地学解读。这为向数据提供方索要来源提供了实证动机。

产物：`outputs/pilot_stat/{report.json,fields.npz}`、`processed/g1_topological.json`、`processed/gates.json`；环境隔离在 `norn/.venv`（原始数据未动）。

**实测发现（G1 证据）**：本地 GMT 导出的全部 38 个“板块多边形”环均非有效简单环（球面超额 −180～−1663 sr，整球仅 4π≈12.57 sr；路径存在往复/双描）。软件按 §1.4 设计**拒绝栅格化并标记**，等待 G1 原始 GPML/ROT 替代——不是用坏几何硬算。

**G0/G1 第一轮推进（2026-10-06）**：详见 `../research_audit/G0_G1_推进报告.md`。
- G0：DAT 符号/单位/定义经 10 地标确认；XLSX 非复制源（corr≈−0.08）；paleo(0)≡现代坐标（14595/14595）；XLSX 为大陆代理目录。**阻断项**：薄壳点 +26 km 偏差 → 代理→物理厚度算子 Gate 升级为硬前提。
- G1：新增 `.rot`/`.gpmlz` 解析器（无 pygplates 依赖）；下载 Müller2019/muller2022 公开旋转+静态多边形（~8.5 MB，哈希入册）；相容性检验：t=0 严格重合、5 Ma 中位 87 km、60 Ma 745 km（单帧对齐不消除→非同源版本；疑似含变形网络拓扑生成）。`scripts/g1_compatibility.py` 可重跑。拓扑级复现需 pygplates（待授权）或提供方确认版本。

## 快速开始

```bash
cd norn
python3 scripts/run_tests.py               # 78 项单测（无 torch、无训练）
python3 scripts/prepare.py                 # 只读生成观测/轨迹表
python3 scripts/freeze_splits.py           # G2 预冻结划分（PRELIMINARY）
python3 scripts/synthetic_benchmark.py     # 12 实例合成基准
python3 scripts/run_benchmark_b7.py        # B7 端到端合成实测
python3 scripts/train.py --dry-run         # 两遍梯度契约/成本；正式训练仍拒绝
```

## 目录

```
src/norn_earth/           核心库（data/geometry/physics/models/losses/evaluation/utils）
scripts/                  prepare / synthetic / train / evaluate / export / run_tests
configs/                  main/stat/baselines/splits（JSON，锁定参数）
tests/                    unittest 单测
processed/                prepare.py 生成的表与 manifest（不含原始数据）
outputs/                  合成报告等产物
```

## 硬性约束（写进代码）

1. 原始数据只读；产物带 SHA256 manifest。
2. 一条观测记录一次证据：年龄积分权重和为 1，加密/复制不增加权重。
3. 观测真实年龄不取整到 TIME 锚点；缺失轨迹不重指派。
4. 无自由逐点噪声头；来源尺度受校准/有界先验限制。
5. 物理预算单位 km³，残差按 H_ref×A_ref 归一；未知通量 mask 而非置零。
6. 训练、统计拟合、环境安装、批量下载均需另行授权；Gate 未过则拒绝。
