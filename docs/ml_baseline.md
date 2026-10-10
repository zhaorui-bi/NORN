# 纯 ML baseline：数据目标与历史消融

0.5 的默认 `no_physics` 与可选 `physics` 标签均绑定同一份 0–60 Ma 的双 GMT 动态输入，操作及参考框架校验见 [GMT／标签协议](gmt_tags.md)。下文 B0–B3 数字与配置是 **0.4 Müller2019 GPML 历史实验**，不能当作新 GMT 的精度结论。

本轮先建立不含物理一致性项的基线，只优化神经网络和数据拟合损失。后续若研究物理项的增益，必须在**同一 backbone、数据 loss、数据划分、初始化种子和训练计划**上只改变物理项，不能同时换网络后将改善归因于物理约束。

## 明确的执行边界

`Config()`、公共模板与默认 A40 配置都使用 `training.objective="data_only"`；物理扩展需显式使用 `--tag physics` 和先验文件（底层目标为 `reconstruction`）。旧 schema-2 权重在加载时补齐原目标／选模语义，避免把旧训练误称为纯 ML。

`data_only` 执行路径：

- 不构造、读取、调用或优化 `PhysicsObjective`，没有轨迹、预算、出生或源项。
- 不计算时间差分平滑项；非零 `temporal_smooth_weight` 直接报错。
- 物理文件路径或任何非零物理权重直接报错，避免隐含继承旧配置。
- checkpoint 无 `physics_state`，训练日志／来源清单无物理损失或源系数。
- 旧受约束配置另存为 `configs/a40_physics_muller2019_legacy.json`；当前 `a40_physics.json` 则共享 GMT baseline 的模型、数据 loss 和验证 MAE 选择，只增加显式物理先验。

给定的动态板块几何仍是**输入数据**，不是损失约束。现代参考仍是条件特征；0 Ma 厚度图仍是监督数据。这与在材料轨迹上强制相同厚度、守恒预算或时间平滑不同。

## Backbone package

新 `gated_sfno` 使用预归一化、球谐谱分支、depthwise 周期局部分支、SwiGLU 通道 MLP 与可学习 LayerScale。年龄通过 Fourier 特征、MLP 和 FiLM 注入。没有 dropout 或 batch statistics，确保无梯度缓存与有梯度重计算完全对应。

`head_mode="input_residual"` 在可用的现代参考输入上回归残差；参考缺失时使用配置的初始化值。输出仍用 softplus 保持正值，但不施加任何轨迹／体积关系。残差头不是把参考当古标签，也不是要求预测等于参考。

本轮的 backbone 对照是一个**组合包**：6 个旧 SFNO block 改为 4 个 gated block，32 通道保持不变，年龄频率带从 2 增至 8，并更换残差头。若要区分这些子组件各自的增益，还需要额外消融。

## 数据拟合目标

保持原年龄节点、古位置和数据误差尺度不变。令

\[
\bar\mu_i=\sum_k w_{ik}H_\theta(x_i(a_{ik}),a_{ik}).
\]

纯数据目标为

\[
L=\lambda_o\left[L_{\rm age\text{-}NLL}
+\lambda_H\frac{1}{N}\sum_i\frac{\rho_\beta(\bar\mu_i-y_i)}{\beta}\right]
+\lambda_m L_{\rm modern\text{-}NLL}.
\]

- 年龄边缘 NLL 仍是固定 `nu=4` 的 Student-t 混合似然；每条记录贡献一次。
- `rho_beta` 是 PyTorch `smooth_l1_loss`，`beta=5 km`；辅助项按年龄边缘均值拟合，除以 beta 统一量级。
- 该辅助项是**经验优化目标**，不是原混合似然的等价重写，也不是每个年龄节点都必须等于同一标签。
- 现代监督仍为原空间块平均高斯 NLL；不改数据语义或混入古标签。

`selection_metric="validation_mae"` 仅按验证集年龄边缘均值 MAE 保存 best checkpoint；训练／现代 loss 不参与该选择分数。NLL、现代拟合误差另行报告，不能把 MAE 改善说成所有统计指标都改善。

## 预设对照

| 对照 | 配置 | Backbone | Huber 辅助权重 |
|---|---|---|---:|
| B0 / control | `configs/a40_ml_control.json` | 旧 SFNO | 0 |
| B1 / backbone | `configs/a40_ml_backbone.json` | gated package | 0 |
| B2 / loss | `configs/a40_ml_loss.json` | 旧 SFNO | 1 |
| B3 / joint | `configs/a40_ml_joint.json` | gated package | 1 |

均使用同一动态数据、0–60 Ma / 61 锚点、训练 15,745／验证 1,855／测试 14,308 条、seed=42、AdamW、500 optimizer steps、相同学习率／warmup／weight decay／clip／现代监督权重。不同架构的参数量、FLOPs 和耗时不相等，另外报告效率。

历史首轮 B0 与 B3 均完成 500 步。验证 MAE 分别为 **9.15851 km** 与 **9.33536 km**，B3 比 B0 差约 1.93%，不声称新组合改善；因此保留 B0 网络／数据 loss，0.5 的 `configs/a40.json` 改用双 GMT 输入，不再是旧 B0 配置的逐项别名。B3 参数从 503,809 降至 348,993，但耗时从 697.96 增至 863.62 秒、峰值显存从 5937.35 增至 11976.47 MiB，不能将参数减少等同于速度或内存改善。

训练集中位数常数／现代不变化基线的验证 MAE 为 9.24388／11.16982 km；B0 对常数基线仅有小幅改善，尚不是强泛化结论。B1/B2 提供后续单因素归因所需配置，**未执行时不声称已完成完整消融**。一次 seed 的结果只是探索，不是统计显著性结论。

```bash
# 使用已有冻结数据；不准备物理先验，不评分测试集
bash scripts/ml_baselines.sh control joint

# 后续补充单因素消融
bash scripts/ml_baselines.sh backbone loss
```

每个运行保存自己的 `best.pt`、`last.pt`、日志、summary 和验证报告。`scripts/compare_ml_runs.py` 检查数据哈希、纯数据模式及共用预算，写 `outputs/ml_baselines/comparison.json`；比较只能读取验证集报告。

历史 `outputs/dynamic_a40/` 不覆盖。动态数据 schema 和旧 schema-2 权重推理保持兼容，但不能用受约束权重 resume 一个纯数据训练计划。

## 测试纪律与科学边界

0.3 的地理组测试集已经按用户要求打开过。0.4 的调参与选择不重复评分该测试集；若今后用它的分数做选择，就不能再称为未触碰的最终测试。需要新的未使用留出才能给出更强的最终泛化声明。

地理代理组缺少研究来源独立性和空间缓冲；全局现代端点允许参与训练。宽年龄区间不是逐年代独立证据，资料仍是 `unverified_proxy`。删除物理项并不验证标签真实性；模型结果与视频仍是研究级点估计。
