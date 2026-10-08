# NORN 0.2 方法与实现

完整方法学采用原项目 `methods` 目录的 [中文 Methods PDF](design/NORN_Methods_CN.pdf) 与 [LaTeX 源文件](design/NORN_Methods_CN.tex)，框架图直接采用 `fig` 目录的 [原图](design/figures/norn_framework.png)。本文补充发布代码的实际计算路径与源码对应关系。原文中的“待接入”记录文档编写时的状态，本页和运行 manifest 描述当前版本。

## 反演变量与网络

估计正厚度 $H_\theta(x,a)$，其中 $x$ 是所查询年龄的球面位置，$a\in[0,60]$ Ma。默认 61 个整数年龄锚点共享同一网络参数，连续年龄使用相邻锚点线性插值。它是针对给定观测集合的实例型求解器，不是将任意观测集一次前向转换为历史图的观测编码器。

网络使用卷积编码器、6 个 SFNO 模块和正值输出头。每个模块包含 Gauss 网格上的球谐变换、复数谱滤波和通道混合、逆变换，以及局部卷积分支。年龄先除以最大年龄，形成低频特征，经 MLP 与 FiLM 调节隐藏通道。使用 GroupNorm；没有随机 dropout 或跨批次更新的归一化状态。局部卷积经度周期填充、极区边缘复制；局部极区处理是近似，不应据此声称严格旋转等变性。

$$H_\theta(x,a_m)=\mathrm{softplus}(F_\theta(X,a_m)(x))+0.5\ \mathrm{km}.$$

输入是固定的 8 通道空间张量。内部严格遵循 torch-harmonics 的 **北到南 Gauss 纬度、经度从 0 开始**；原生现代网格是 **南到北、经度从 0.5° 开始**。两个网格通过显式保守映射或可微插值连接。特征的变形覆盖通道无独立资料时标为缺失，而不是宣称已经使用变形资料。

源码：[operator.py](../src/norn_earth/models/operator.py)、[regrid.py](../src/norn_earth/geometry/regrid.py)、[sampling.py](../src/norn_earth/geometry/sampling.py)。

## 古观测的年龄边缘似然

每条记录有厚度 $y_i$ 与年龄分布。默认均匀分布，求积节点分别为点年龄 1 个、宽度不超过 5 Myr 的非零区间 3 个、更宽区间 8 个；权重之和为 1。

$$p(y_i\mid\theta)\approx\sum_k w_{ik}\,
t_4\!\left(y_i;H_\theta(x_i(a_{ik}),a_{ik}),\sigma_i\right),$$

$$\mathcal L_{age}=-\frac1{N_{tr}}\sum_{i\in tr}\log p(y_i\mid\theta).$$

这是似然的混合，不是把同一厚度复制为多条标签，也不是对节点的负对数似然直接平均。有效尺度为 $\sqrt{\sigma_{record}^2+\sigma_{source}^2}$；真实数据示例使用固定声明值 4 km 与 9/26 km，不能理解为已独立校准。

`coordinate_mode=paleo` 时，每个节点都由 pygplates、指定旋转模型及静态分区重建位置。完全无法定位的记录剔除；不会根据厚度误差选择年龄节点。域外年龄支持质量被记录，推理禁止时间外推。

源码：[preparation.py](../src/norn_earth/data/preparation.py)、[kinematics_engine.py](../src/norn_earth/data/kinematics_engine.py)、[objective.py](../src/norn_earth/losses/objective.py)。

## 现代约束

现代预测在原生现代观测位置查询，使用指定误差尺度的高斯 NLL。先计算每个空间块内的平均，再对块平均；这样不会因某个块具有更多格点而自动提高其证据权重。现代资料是带误差的观测，不是硬约束。

验证和真实样例允许现代端点信息参与训练，属于 P-B 型任务。软件不将这一结果解释为完全未见现代资料的空间外推。

## 薄层物理关系

使用向现代推进的时间 $\tau=60-a$，控制关系为：

$$\frac{\partial H}{\partial\tau}+\nabla_s\cdot(H\mathbf v)=q,
\qquad q=q_{mag}-q_{loss}.$$

### 材料轨迹

$$r_{traj}=H(x_2,\tau_2)-H(x_1,\tau_1)
-\int_{\tau_1}^{\tau_2}\left(q-H\nabla_s\cdot\mathbf v\right)d\tau.$$

软件按递减地质年龄给出的路径积分，使用相邻节点的梯形求积。残差除以指定厚度误差尺度，再形成平方损失。无源变形关系 $H_2=H_1/J_A$ 是同一机制的特例；不把其派生厚度再次作为独立标签。

### 移动控制域的弱形式预算

$$V(\tau)=\int_{\Omega(\tau)}H\,dA,$$

$$\frac{dV}{d\tau}=-\oint_{\partial\Omega}
H(\mathbf v-\mathbf v_b)\cdot\mathbf n\,dl+\int_\Omega q\,dA.$$

软件用内部面积求积点计算体积和源积分，用边界点、边长权重和**相对法向速度**计算通量：

$$r_V=\frac{\Delta V+I_{flux}-I_{source}}{H_{ref}A_{ref}}.$$

参考体积由物理文件提供并固定，不随预测变化。源或边界通量未知时，整个预算因子不启用。不会假定全球活动地壳总体积恒定。跨拓扑事件的积分需要外部制备程序分段，并提供各段有效掩膜。

### 出生与受限源项

洋壳出生先验使用 $H_{birth}=Q_{retained}/u_{full}$，对应 km²/Myr 与 km/Myr；只在外部资料支持时启用。

净源采用固定少量基函数：$q=\sum_j c_j\psi_j$。各系数通过有界变换约束在用户提供的上下界内，使用指定均值及完整协方差先验，可另外对指定系数对施加平滑。没有自由逐像素源项。厚度资料难以分别识别总生产、保留比例和损失，本版接口要求使用有意义的净源或保留产率组合。

每个物理因子声明证据 ID。同一证据不能同时用于轨迹、预算或由相同源资料派生的出生厚度先验；同一角色内重复求积因子按证据组归一，不因复制而增加独立证据量。

源码：[differentiable.py](../src/norn_earth/physics/differentiable.py)，数据接口见 [physics.md](physics.md)。

## 联合优化与两遍梯度

目标由年龄、现代、轨迹、预算、过程先验和可选正则组成。各项在统一配置中设权；没有物理文件或相应权重为零时，该模块不进入训练。时间平滑是可选正则，不能代替守恒关系。

第一遍在固定网络参数下生成锚点场缓存 $U_m$，将它们视为叶变量，计算完整目标和 $G_m=\partial\mathcal L/\partial U_m$。受限源参数在这一步获得联合目标的直接梯度。

第二遍按小批量重新生成锚点，对 $\langle\mathrm{stopgrad}(G_m),H_\theta^{(m)}\rangle$ **立即反向传播**。各批次之间不更新参数，所有梯度完成后才执行 AdamW。日志记录第一遍的真实目标；不会报告代理内积为损失。这样保留场缓存，但不同时保留 61 个年代的全部激活图。

源码：[trainer.py](../src/norn_earth/training/trainer.py)。完整观测和物理目标的梯度等价性在 [test_release.py](../tests/test_release.py) 中验证。

## 当前资料边界

真实 A40 样例启用古位置、年龄边缘似然、现代约束，以及明确标注的模型导出刚性无源先验。后者是敏感性假设，不能替代独立的变形、生产率、保留率或消亡通量资料。完整源项和出生约束在可控合成例子中联合训练和验证。

软件计算正确、训练收敛、独立推理成功，分别回答不同问题；这些证据不能自动推出地质厚度历史具有唯一解或全球准确。输出间隔不是可辨识分辨率，年龄边缘似然也不自动产生厚度置信区间。
