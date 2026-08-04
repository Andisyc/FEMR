# One-Shot Intent-Execution Adaptation

> 状态：早期概念笔记  
> 日期：2026-07-14

## 1. FADA 已经验证了什么

FADA 的核心概念不是 Planner-Tracker 双策略，也不是 Transformer，而是：

> 当运动意图保持纯洁时，失败的真机轨迹可以作为执行校准信号。

对于走直线等任务，Planner 给出的运动意图可以视为正确目标；真实运动相对目标的偏移主要属于执行问题，因此可以固定 Planner，只校准 Tracker 或 IDM。

## 2. FADA 尚未触及的下一层

真实部署中的失败不一定只意味着“没有执行准确”。环境或任务条件变化后，原运动意图本身也可能需要改变。

这里应区分三个层级：

- **任务意图**：机器人最终要完成什么，例如保持方向和速度向前行走。
- **运动意图**：为了完成任务，应采用什么姿态、步态和全身协调方式。
- **执行映射**：机器人如何把运动意图准确实现为真实运动。

FADA 固定前两个层级，只校准执行映射。下一层研究应在保持任务意图不变时，同时处理运动意图变化与执行偏差。

## 3. 为什么轻质大件搬运是干净切入点

负重同时引入几何约束、惯性变化、外力补偿和接触动力学，难以判断失败来自运动意图还是执行映射。

更干净的第一步是搬运**质量可忽略、但体积较大的物体**：

- 任务意图仍是按指定速度和方向移动；
- 物体几何迫使手臂、躯干、步宽和整体姿态发生变化；
- 质量影响被尽量排除，主要保留运动意图变化。

因此，该任务可以隔离并验证：

> 真实执行失败能否揭示当前 Command 缺失的姿态条件，并反向修正运动意图？

## 4. 统一概念

FADA 的失败轨迹用于修正当前意图的执行。更进一步，失败轨迹可以被理解为对当前**部署条件**的一次观测。

部署条件可能同时包含：

- **Intent shift**：当前任务条件要求怎样改变运动意图；
- **Execution shift**：给定运动意图在真实机器人上为何产生系统性偏移。

由此形成核心假设：

> 一条失败轨迹不应只用于修正该轨迹，而应识别一个可跨 Command 复用的部署条件。

在同一物体或同一机器人条件下，只使用一条真机动作序列完成适应，随后泛化到未见过的速度、方向、姿态组合和动作序列。

暂定名称：**One-Shot Intent-Execution Adaptation**。

## 5. 对 FADA 工程流程的概念压缩

FADA 使用教师模型、IDM、Planner、真机数据收集和微调等多阶段流程，并通过 Planner-Tracker 架构解耦保护意图纯洁。其不可缺少的因果约束其实只有三个：

1. 原始任务语义必须保持稳定；
2. 适应变量必须具有明确且受限的修改权限；
3. 真实失败轨迹必须提供可诊断的适应证据。

因此，下一步可验证：

> 意图纯洁能否由适应权限保证，而不依赖显式的 Planner-Tracker 双策略？

这意味着把**结构解耦**压缩为**功能或权限解耦**：保留一个已训练的单策略，冻结其主体，只适应一个受限变量。概念流程可缩短为：

> 训练单策略 -> 执行并观察一条真机轨迹 -> 识别部署条件并适应

真机数据收集与适应还可以形成同一个在线循环。MLP 或 Transformer、独立 IDM 或 residual adapter 都只是候选实现，不属于当前概念本身。

## 6. 最核心的科学命题

> 能否从单条真机轨迹中识别一个可跨 Command 泛化的部署条件，同时表达必要的运动意图变化与执行校准？

该命题比 FADA 多推进了一层：

- FADA：固定运动意图，校准“怎样执行”。
- 本概念：固定任务意图，同时识别“应该怎样运动”与“怎样准确执行”。

## 7. 必须避免的退化

最危险的退化是把适应变成轨迹记忆：在采集 Command 上表现改善，但无法迁移到新的 Command。

因此，适应对象必须表达**条件级变化**，而不是轨迹级动作修正。真正的验证标准不是原轨迹重放精度，而是：

> 在一个 Command 上观察一条真机轨迹后，能否改善同一部署条件下未见过的 Command？

另一个关键歧义是：同一失败既可以被解释为“运动意图错误”，也可以被解释为“执行不准确”。未来方法必须为两类修改定义不同权限，否则二者会相互解释，破坏泛化。

## 8. 当前研究边界

当前第一阶段只研究轻质大件带来的几何与姿态条件，不处理显著负重、惯性变化或复杂力补偿。

当前也不预设具体网络结构。首先需要验证的是：单条轨迹能否识别可泛化的条件变量，以及意图变化与执行变化是否必须显式分解。

## 9. 下一个关键问题

> 为了获得跨 Command 泛化，部署条件应当被表示为一个共享低维变量，还是必须显式分解为 intent context 与 execution context？

这是下一步需要闭合的单一概念变量。

---

## Research Log

### 2026-07-15：从“泛化”退回到更锋利的概念

#### 原始讨论

FADA 最令人惊讶之处，不是其复杂工程流程，而是它验证了一个大胆猜想：当运动意图保持纯洁时，失败的真机轨迹本身就能用于校准执行。当前研究希望找到同样具有反直觉性、但可由最小工程闭环验证的下一层命题。

讨论曾提出：一条失败轨迹可以识别部署条件，并跨 Command 泛化。进一步考虑后，该表述被降级，原因是“泛化”是连续、模糊且高度依赖分布距离的问题。若将泛化作为论文主命题，就必须建立强 baseline、定义 Command 距离，并展示非常显著的跨分布效果；否则概念容易退化为普通性能提升。

#### 当前概念修正

泛化不再作为核心贡献，只作为辅助证据。更接近 FADA 式演进的候选猜想是：

> 失败轨迹不仅能校准“如何执行”，还能校准“应该执行什么”。

FADA 将适应权限限制在 Tracker/IDM，即执行空间。下一步候选概念将 residual 提升到 Command 或运动意图空间：第一次执行因姿态意图不适合当前条件而失败，系统仅依据该失败轨迹产生一个保持任务语义的 intent residual，使第二次执行成功。

轻质大件搬运仍是当前最干净的验证场景：它主要引入几何和姿态约束，同时尽量排除负重、惯性变化和力补偿。理想展示是第一次因身体姿态或物体几何关系失败，第二次在 Tracker 不变的情况下，通过修正运动意图直接成功。

#### 已接受的研究边界

- **主命题候选**：真实失败能否自主修正运动意图，而不仅是执行映射。
- **主实验要求**：一次失败、一次修正、第二次成功，形成强视觉对比。
- **任务语义不变**：移动方向、目标位置或速度等高层任务保持不变。
- **允许变化**：姿态、手臂构型、躯干配置、步宽等运动意图。
- **执行层边界**：必须区分 intent residual 与 execution residual，避免两者相互解释。
- **泛化地位**：只验证修正是否在邻近速度或方向仍有效，不作为标题级贡献。
- **当前非范围**：显著负重、复杂外力补偿、全局泛化，以及具体网络结构。

#### 第一轮创新性检索结果

当前宽泛表述“从失败修正意图”不能直接声称新颖。第一轮检索发现以下高风险邻近工作：

- [FADA](https://arxiv.org/abs/2606.28476)：用少量目标域 rollout 校准 IDM，但冻结 Planner，解决执行映射偏差。
- [RMA](https://arxiv.org/abs/2107.04034)：从近期状态动作历史识别环境条件并快速适应，威胁“轨迹识别部署条件”的表述。
- [A Learning-based Iterative Control Framework](https://www.roboticsproceedings.org/rss18/p029.html)：用执行误差学习前馈控制，并跨参考轨迹泛化，威胁“单条执行误差向邻近 Command 迁移”的表述。
- [FAST](https://arxiv.org/abs/2602.11929)：使用轻量 delta action policy 完成 humanoid 快速适应，说明 residual 微调本身不是创新。
- [RuN](https://arxiv.org/abs/2509.20696)：将条件运动生成与动力学 residual 解耦，说明“生成意图 + residual 执行”已有直接邻居。
- [ReactiveBFM](https://arxiv.org/abs/2606.30362)：让 Planner 从不完美物理状态学习错误恢复，直接威胁“执行反馈修正 Planner”的宽泛表述。
- [REFLECT](https://arxiv.org/abs/2306.15724)：从失败执行中解释错误并指导语言 Planner 修正任务计划。
- [Dream2Fix](https://arxiv.org/abs/2603.13528)：从失败表征预测可执行恢复轨迹，说明“失败到纠正轨迹”已被研究。
- [AdaMimic](https://arxiv.org/abs/2510.14454)：从单个参考动作实现 adaptable humanoid control，威胁“单动作快速适应”的表述。
- [ResVLA](https://arxiv.org/abs/2604.21391)：显式区分 global intent 与 local dynamics，说明意图/动力学分解本身也不是新概念。

#### 当前创新性状态

**尚未证明新颖。** 第一轮检索只说明各个组成部分均有先例。仍可能成立的创新边界必须是以下组合，而不是其中任一单点：

> 无需人工纠正、奖励或最优示范，仅使用一次真实 humanoid 失败，区分运动意图错误与执行偏差，产生保持任务语义的 intent residual，并使第二次真实执行成功。

该组合目前未在第一轮检索中发现被完整闭合，但不能据此宣称创新。开始设计方法前，必须对上述最相近论文进行全文级 claim matrix，逐项核对：输入证据、失败来源、被修改变量、人工监督、适应次数、Planner/Tracker 权限、真实机器人验证和跨 Command 证据。

#### 下一步

> 先证明“失败能够自主修改运动意图”这一命题是否仍有独立创新空间，再决定 residual 的表示和工程架构。

### 2026-07-15：从任务选择退回到单次失败的可辨识性

进一步讨论发现，轻质大件搬运虽然能够制造清晰的运动意图变化，但它会引入完整的 loco-manipulation 流程、视觉感知、手物交互数据、抓取稳定性和新的任务 reward。此时研究工作量已经从验证一个 question 扩张为建设一个 task stack。除非存在可以直接复用且已完成闭环验证的代码，这些上层能力不应进入第一阶段的方法闭包。搬运因此被降级为未来展示场景，而不是当前核心问题。

真正被任务选择遮住的基础问题是：

> 一条失败轨迹究竟能够辨识什么？

设机器人接收任务目标 \(g\)，产生运动意图 \(m\)，再通过执行映射 \(f\) 得到真实轨迹 \(x\)。当真实轨迹偏离目标时，失败至少存在两种解释：运动意图本身不适合当前条件，或正确运动意图没有被准确执行。概念上可写为：

\[
\text{failure} = \Delta m_{\mathrm{intent}} + \Delta f_{\mathrm{execution}}.
\]

单条轨迹只提供一次观测，却包含两个未知原因。在没有额外结构假设时，intent error 与 execution error 通常不可辨识。FADA 能够闭合，正是因为它预先固定 Planner，将运动意图视为正确，从而把目标域偏移限制在 Tracker/IDM 的执行映射中。若下一步工作允许修改运动意图，就必须重新回答：凭什么知道失败应归因于 intent，而不是 controller？

另一个关键结论是：失败本身主要提供负面证据。它能够说明“当前方案不对”，但不天然给出“应该怎样修改”。因此，“使用一次失败轨迹直接微调”还不是完整机制；梯度或修正方向至少需要来自以下一种信息：

1. **可计算的目标误差**：正确目标或参考仍然已知，失败与目标之间的偏差直接提供方向；
2. **预训练的失败先验**：在仿真或既有数据中学习失败模式与修正之间的关系，真机失败只负责识别或更新条件变量；
3. **额外试探形成的反事实证据**：比较修改前后的结果，判断候选修正是否优于原策略，但这不再是严格意义上的单轨迹适应。

由此，当前最基础且最锋利的科学问题不再是“选择什么任务来修改运动意图”，而是：

> 在没有正确示范的情况下，使单次失败从负面证据变成定向修正信号，最少需要什么先验？

这一定义暂时独立于搬运、视觉和具体网络结构。下一步必须先决定允许哪一种修正方向来源：可计算任务误差、仿真预训练先验，或额外一次试探执行。只有该变量闭合后，实验场景、适应对象与工程架构才有确定依据。

### 2026-07-15：为什么失败 Rollout 能实现执行对齐

重新细读 FADA 后需要修正一个重要表述：目标域适应并不是直接更新整个 Tracker 或 IDM。论文在目标域冻结 Planner 参数 \(\phi\) 和预训练 IDM 参数 \(\psi\)，仅优化插入 IDM 的 LoRA 参数 \(\Delta\psi\)。Appendix B.3 给出的配置为 rank \(r=8\)、scaling \(\alpha=16\) 和 dropout 0.05。部署时实际使用的是 \(I_{\psi+\Delta\psi}\)。因此，基础 IDM 权重没有更新，但有效的 plan-to-action 映射发生了受限变化；不能把“参数冻结”误解为“执行函数没有适应”。

目标 rollout 被转换为与源域 IDM 训练相同的窗口：

\[
W_{\mathrm{tgt}} = (O_t^H, A_t^H, Y_{t,\mathrm{exec}}^K, U_{t,\mathrm{exec}}^K).
\]

其中，\(Y_{t,\mathrm{exec}}^K\) 是真实执行后观察到的未来本体状态，\(U_{t,\mathrm{exec}}^K\) 是同一物理 rollout 中实际执行的动作序列。目标域没有 reward、oracle label、simulator calibration 或 source replay；监督完全来自这组 realized-future/action 配对。

由此，使用“状态分布修补”解释 FADA 是有根据的，但仍不完整。失败 rollout 至少同时提供两类信息：

1. **Target occupancy**：暴露源域未覆盖的部署历史、姿态、接触后状态和累积偏移；
2. **Target inverse-dynamics correspondence**：给出目标动力学下“什么动作实际产生了什么未来”的因果配对。

LoRA 则不负责提供数据方向，而是限制适应权限和参数容量。FADA 的 full-IDM finetuning ablation 在三个任务上均劣于 zero-shot，说明少样本条件下完整更新容易过拟合；LoRA 的作用更接近受限执行残差和正则化。冻结 Planner 还提供了不变的任务意图锚点，使目标域更新不会重新解释任务语义。

因此，目前更准确的机制假设是：

> 失败 rollout 通过覆盖关键目标域状态，并提供这些状态下的 realized-future/action 对，使低秩 IDM residual 能够重新校准 Planner-to-action 接口。

这一解释包含四个尚未被论文完全拆开的因素：target occupancy、future-action pairing、frozen Planner anchor 和 low-rank update constraint。FADA 的现有实验分别证明了 LoRA 比 full finetuning 更稳定、数据收益在约 6000 steps 后趋于饱和，以及 IDM consistency gap 在固定基座负载实验中下降；但这些证据还没有回答四个因素中哪一个是主要因果来源。

由此形成新的基础研究问题：

> Why Do Failed Rollouts Enable Execution Alignment?

第一项最小判别实验应保持 target rollout 的状态边缘分布和样本数量不变，只打乱 \(Y_{t,\mathrm{exec}}^K\) 与 \(U_{t,\mathrm{exec}}^K\) 的时序或窗口配对。如果适应收益消失，说明关键证据是目标域逆动力学对应关系，而不是单纯的状态覆盖；如果仍然有效，才支持 occupancy repair 占主导的解释。后续再分别控制 rollout 质量、off-support 程度、LoRA 容量和跨 Command 迁移，系统拆解“哪些失败是可校准数据，哪些失败只是坏数据”。

### 2026-07-15：LoRA 可能在辨识跨状态共享的低维动力学修正

进一步追问仍然存在一个 gap：目标域 rollout 数量有限，也没有覆盖完整的未来状态分布，为什么 LoRA-IDM 能在尚未于目标域真正实现过的 Planner intent 上输出正确动作？这里首先需要区分：适应过程没有重新学习期望未来。Planner 已经能够输出 \(\hat Y_{\mathrm{plan}}\)，并在目标适应过程中保持冻结；LoRA-IDM 新学的只有给定历史和未来状态时应当输出什么动作。

目前最有解释力、但尚未被 FADA 原文直接证明的机制推断是：源域 IDM 已经学习了大部分通用逆动力学，目标域数据无需重新学习完整分布，只需辨识一个由持续部署条件引起、能够跨状态共享的低维修正。可写为：

\[
G_{\mathrm{tgt}}(h,Y)
=
G_{\mathrm{src}}(h,Y)
+
\Delta G_{\theta}(h,Y).
\]

其中，\(G_{\mathrm{src}}\) 由大规模源域训练提供，目标 rollout 只负责估计小维度的 \(\theta\)。如果目标域变化主要来自电机增益、持续负载、摩擦系数或稳定接触条件，那么大量不同状态中的执行误差可能共享同一个隐藏原因。例如，当执行器响应整体缩小为源域的 0.8 倍时，需要辨识的主要是一个近似缩放量，而不是重新学习所有 \(Y\rightarrow U\) 配对。有限状态上的 realized-future/action 对便可能足以估计该修正，并将它应用到邻近但未在目标域实现过的 Planner intent。

LoRA 在这里可能同时承担两层作用：第一，它限制目标域更新的自由度，避免少样本重写整个 IDM；第二，它用预训练特征规定了修正如何从已观测状态延伸到未观测状态。FADA 的主适应预算约为 6000 control steps，通过滑动窗口形成大量但高度相关的监督样本。这些数据不足以重新学习目标域完整分布，却可能足以估计一个低维、持续的动力学差异。

从这个角度看，FADA 不是传统的 zero-shot transfer，而更接近：

> few-shot implicit system identification + pretrained inverse-dynamics prior + local out-of-sample intent query.

真正尚未闭合的科学假设是：

> 目标域动力学差异是否真的可以由一个跨状态共享的低秩参数更新表达？

如果该假设成立，LoRA 参数近似承担了隐式 dynamics latent 的角色，窄分布目标数据也能改善未见过的 Command；如果不成立，FADA 的收益可能主要来自 rollout support 附近的局部插值，而不是可复用的动力学辨识。

对应的最小实验是：只在狭窄 Command 或姿态范围收集适应数据，然后逐渐增加测试 intent 与适应轨迹之间的距离，测量

\[
\mathrm{adaptation\ gain}
\quad\mathrm{vs.}\quad
d(\hat Y_{\mathrm{test}},Y_{\mathrm{adapt}}).
\]

若远离适应轨迹的 Command 仍获得稳定收益，说明 LoRA 更可能辨识了跨状态共享的动力学修正；若收益随距离迅速消失，则其主要作用是局部状态分布修补。这一实验可以直接区分“隐式低维系统辨识”和“局部行为插值”两种解释。

### 2026-07-28：失败轨迹补充的不是“正确真实分布”，而是真实物理对应关系

进一步讨论发现，“失败轨迹弥补了缺失的真实分布”仍然跳过了一个关键因果环节。失败 rollout 只直接说明：在目标机器人上，从某段历史出发，实际执行这些动作后产生了哪些未来状态。它补充的是 **realized future-action correspondence**，而不是“正确意图—正确动作”的目标分布。失败数据能够说明现实怎样响应，却不天然说明本来应该怎样运动。

FADA 能够把失败 rollout 变成校准信号，依赖三个同时成立的边界：

1. Planner 冻结，Planner future 被当作正确运动意图锚点；
2. 目标 rollout 提供目标动力学下的 realized-future/action 配对；
3. 预训练 IDM 保留源域逆动力学先验，LoRA 只允许小容量执行层修正。

因此，当前更准确的概念表述是：

> 当正确运动意图已有外部锚点时，失败轨迹能够把缺失的真实物理对应关系补入执行模型。

这也明确了 FADA 的责任边界。对于一个固定 Intent，例如走直线，目标 rollout 覆盖该 Intent 下的一片历史、姿态、接触和未来状态分布；适配后的 IDM 可以改善这片分布附近的 plan-to-action 映射。但这不等于“失败轨迹同时校准了 Intent 与 Execution”，因为 FADA 没有从失败中学习新的正确意图。

同样，即使论文报告多个任务，只要每个任务分别收集自己的目标 rollout 并分别适配，就只能证明 **within-Intent execution calibration**。它不能证明：使用任务 A 的失败轨迹适配后，无需任务 B 的目标数据即可执行任务 B。当前文档未把跨任务迁移视为 FADA 已验证事实；该 claim 仍需通过论文级实验协议核对。

由此，下一层问题被进一步收缩为：

> 一个 Intent 的失败轨迹所校准的执行模型，能否在不收集新目标数据的情况下处理另一个 Intent？

若只能改善适配 rollout 覆盖的状态区域，其机制更接近局部分布修补；若能改善未覆盖的新 Intent 查询，才支持跨状态共享的目标动力学修正。

#### LoRA 数学对象的修正

此前使用

\[
G_{\mathrm{tgt}}(h,Y)
=
G_{\mathrm{src}}(h,Y)
+
\Delta G_{\theta}(h,Y)
\]

描述低维修正，容易被误解为最终 action 或完整网络函数上的 residual。该式只适合作为非严格的函数空间直觉，不应作为 LoRA 的结构公式。

LoRA 的相加发生在被适配网络层内部。对第 \(l\) 层：

\[
z_l
=
\left(W_l+B_lA_l\right)x_l
=
W_lx_l+B_lA_lx_l.
\]

其中 \(W_l\) 是冻结的预训练权重，\(A_l,B_l\) 是目标域训练的低秩分支。它修正中间特征和有效网络参数，而不是在 Tracker 输出后添加 \(\Delta a\)。整个适配后 IDM 的动作条件分布应写为：

\[
p_{W+\Delta W}(a\mid h,Y),
\qquad
\Delta W_l=B_lA_l.
\]

这里 \(p\) 表示动作条件分布，\(W+\Delta W\) 表示内部若干层已由 LoRA 修正。由于网络含有非线性，一般不能把完整网络写成 \(G_{W+\Delta W}=G_W+G_{\Delta W}\)。

#### 当前未闭合边界

- **已接受**：失败 rollout 提供真实物理对应关系，而不是正确意图标签。
- **已接受**：FADA 通过冻结 Planner 消除 Intent/Execution 的主要归因歧义。
- **已接受**：LoRA 是逐层参数/特征修正，不是 action residual。
- **未确认**：FADA 的收益主要来自 target occupancy、future-action pairing，还是跨状态共享的低维修正。
- **未闭合**：若同时允许 Intent 与 Execution 改变，什么额外证据能够提供意图修正方向并区分两类误差。

### 2026-07-29：从目标域微调到轨迹条件推理

进一步回看 ASAP、HDMI、OmniXtreme 与 FADA，可以看到一条比“跨任务泛化”更稳定的演进主轴：**持续压缩真实世界必须承担的学习负担**。

- ASAP 需要从真实执行中获得 residual 信息，并继续优化策略；
- HDMI、OmniXtreme 保留主体能力，但仍需额外训练 residual policy；
- FADA 不再训练独立修正器，而是用少量真实 rollout 对预训练 IDM 进行低秩监督适配。

这一演进背后的共同观察是：大部分运动能力已经在源域中获得，目标域不应重新学习完整技能，只需提供源域无法包含的真实物理证据。FADA 已经把目标域学习压缩得很轻，但其部署过程仍然属于学习：失败 rollout 被构造成监督样本，梯度更新把目标域信息写入 LoRA 参数。

这里必须区分两种压缩：

1. **流程压缩**：减少数据整理、梯度步数、模块数量或运行时间。如果真实轨迹仍以相同方式监督参数更新，这主要是工程改进。
2. **证据角色改变**：真实轨迹不再负责教会模型新的执行映射，只负责消除模型对当前部署条件的不确定性。这才可能形成新的科学命题。

由此产生一个候选的下一层概念：

> 能否把 test-time learning 变成 trajectory-conditioned inference，使真实世界只提供判别当前执行条件的证据，而不再承担策略训练？

其概念流程是：离线阶段预先学习可适配的执行规律；部署时读取一条真实失败轨迹，将其压缩为临时 deployment context；冻结全部模型参数，只用该 context 条件化 Tracker 对后续 Intent 的动作输出。轨迹在这里更像物理提示，而不是微调数据集。学习没有消失，而是从部署阶段前移到离线阶段。

该命题与 FADA 的关键差别不是“更新更快”，而是目标域证据的因果角色发生改变：

- **FADA**：真实轨迹提供监督，改变可部署模型参数；
- **候选概念**：真实轨迹提供证据，只改变临时推理状态，模型参数保持冻结。

这一概念只有在以下条件下才被验证：相同的单条目标 rollout 输入后，完全禁止目标域梯度更新和永久参数修改，轨迹条件模型仍能产生与 FADA 相当的 execution alignment。若只是用另一个网络预测 LoRA 权重，或者把多步优化蒸馏为一次参数写入，却没有证明真实轨迹从“教学”变成“消歧”，则仍可能只是流程工程压缩。

当前不可回避的边界是：轨迹条件推理无法凭空处理离线训练从未表示的物理变化。目标 rollout 只能在预训练适配先验中识别或组合修正；它是否包含足以区分当前部署条件的证据，仍受轨迹激励范围与可辨识性约束。

#### 当前概念状态

- **候选主命题**：真实世界适配能否从参数学习退化为基于单条轨迹的条件推理。
- **不可缺少的变量**：目标 rollout 在部署阶段承担的是“教学”还是“消歧”。
- **冻结边界**：部署阶段不得通过梯度或其他永久写入改变策略、Tracker 或 adapter 参数。
- **允许变化**：由轨迹产生、只服务于当前部署条件的临时 context 或内部推理状态。
- **尚未选择**：context 表示、序列编码器、训练目标与具体网络结构。
- **尚未证明**：现有 in-context adaptation 技术是否能够在 humanoid execution alignment 中闭合这一命题。

治理状态：本节是早期研究提案，不是已接受的 FrontRES 活动合同，也未映射到当前 FrontRES Concept Figure。独立候选图见 `note/architecture/concept/08_trajectory_conditioned_execution_alignment.data.json`，仅用于讨论仿真跨 Intent 训练与冻结参数真机推理的概念闭环。在核心证据、训练权限和验证边界被确认前，不进入实现计划。

### 2026-08-03：科研目标应从现象与证据边界中生成

进一步反思发现，当前候选思路之所以逐渐依赖大量不确定条件，不只是因为方法尚未完善，也因为讨论采用了目标先行的顺序：先设定“一条轨迹应当校准其他 Intent”，再引入仿真覆盖、context、元学习和可辨识性等机制去支撑这个预设目标。每增加一个机制，也同时增加一个尚未被现象证明的前提。

对于早期科学研究，更稳健的顺序应当反过来：

> 可重复现象 → 可观察证据 → 因果分析 → 最小可证伪变量 → 该变量自然支持的任务与命题 → 最小方法。

这里并不是取消研究目标，而是改变目标的来源。目标不再是预先指定的能力愿望，而是从现象所暴露的信息和因果边界中派生出来的科学命题。工程可以从“想实现什么”出发寻找手段；科学概念形成则应先回答“已经观察到什么、它必然说明什么、它不能说明什么”。当这些边界足够清楚时，真正值得研究的目标会自然浮现。

对当前问题而言，不再预设直线失败轨迹必须能够校准转弯或八字。首先只研究一条真实 rollout 中哪些执行信息必然存在、哪些只在特定激励条件下存在、哪些原则上没有被观察到。随后，由这些信息能够可靠约束的后续行为范围，决定跨 Intent claim 的边界。Context Encoder、离线仿真族和具体适配机制都应晚于这一现象分析，而不能替代它。

真机上的可见效果是具身智能工作的必要证据，但不是 insight 正确的充分证据。若不输入失败轨迹也能得到相同提升，真实效果仍然存在，却不能支持“轨迹提供了执行条件信息”这一因果解释。只有当改变或移除所识别的关键证据，会按照分析所预测的方向改变后续执行，真机结果才同时支持方法效果与科学 insight。

当前状态：轨迹条件推理仍保留为候选假设，但不再把“单轨迹跨 Intent 成功”当作必须实现的预设目标。下一层讨论应先刻画单条 rollout 的信息边界，再由该边界决定研究命题和最小可执行方法。

### 2026-08-03：从真实物理对应关系到 Context-Conditioned Action Residual

在接受“科研目标应从现象与证据边界中生成”之后，讨论进一步收缩了当前任务。第一阶段不再要求一条直线轨迹校准转弯、圆圈或八字，而是接受 **within-Intent calibration**：直线任务使用真实直线轨迹，转身任务使用真实转身轨迹。需要检验的新问题变为：

> 能否把同一任务中的真实失败 rollout 从微调数据改造成推理条件，使冻结模型无需目标域梯度更新即可产生执行修正？

#### FADA 公开实现的证据边界

截至 2026-08-03，对两位共同一作和 LeCAR-Lab 公开 GitHub 仓库的检索得到以下边界：

- [Angchen Xie](https://github.com/AngchenXie) 名下未发现 FADA 方法实现；
- [Nike353/fada-corl](https://github.com/Nike353/fada-corl) 是只有简短 README 的空占位仓库；
- [Nike353/few_shot_adaptation](https://github.com/Nike353/few_shot_adaptation) 是 online adaptation 研究综述笔记，不是 FADA 代码；
- [LeCAR-Lab/FADA-humanoid](https://github.com/LeCAR-Lab/FADA-humanoid) 是 Vite + React + TypeScript 项目网页，源码明确保留了“代码公开后再启用 Code 按钮”的状态；
- 网页公开的 `public/models/reach` ONNX 文件与 reach/payload 演示对应，不能据此认定为完整 humanoid FADA checkpoint；
- 公开仓库中没有发现 Oracle RL、Planner-IDM DAgger 蒸馏、真实 rollout 记录、IDM LoRA 适配或 G1 部署实现。

因此，后续机制分析必须区分论文确认与合理推断。演示 ONNX 不能作为完整实现证据；Real Action 的底层控制语义、日志字段、窗口时间对齐和 LoRA 样本构造目前仍没有代码确认。

#### Real Action、失败轨迹与 IDM 校准

论文中的目标域窗口写为：

\[
W_{\mathrm{tgt}}
=
(O_t^H,A_t^H,Y_{t,\mathrm{exec}}^K,U_{t,\mathrm{exec}}^K).
\]

其中，\(U_{t,\mathrm{exec}}^K\) 应理解为同一真实 rollout 中实际发送并执行的动作序列，而不是正确动作标签；\(Y_{t,\mathrm{exec}}^K\) 是这些动作在真实物理中实际产生的未来状态。其直接证据是：

> 在当前机器人和当前历史下，这些 Action 实际产生了这些 Future State。

这类正向执行数据可以构成逆动力学样本，因为它提供了 realized-future/action correspondence；但它只能说明如何复现已经发生的 future，不能单独说明如何实现一个尚未发生的 Planner future。FADA 能把它用于执行校准，依赖三个边界共同成立：

1. Planner 冻结并提供正确运动 Intent；
2. 预训练 IDM 已经包含主要源域逆动力学先验；
3. LoRA 只允许小容量、低秩的执行映射修正。

因此，不能把 FADA 简化为“输入 Real Observation、以 Real Action 为正确标签进行普通行为克隆”。训练窗口的历史、future/action 配对和时序关系都是机制的一部分。当前仍未被论文完全拆解的是：LoRA 的收益主要来自 target occupancy、future/action correspondence，还是跨状态共享的低维执行修正。

#### 冻结 Tracker 暴露出的结构缺口

轨迹条件推理最初被描述为“Context Encoder 读取真实轨迹，冻结 Tracker 根据 Context 输出新动作”。进一步检查后发现，该表述缺少 Context 到 Action 的作用路径：

> 单独外挂一个 Context Encoder，不能使一个从未接收 Context 的冻结 Tracker 改变动作。

这里需要区分两种冻结边界：

1. **只在部署时冻结**：离线阶段可以共同训练 Context Encoder 与 context-conditioned Tracker，部署时冻结全部权重；
2. **原 Tracker 始终冻结**：Context 必须进入一个具有动作修改权限的独立 Corrector。

若希望最大限度保留原 Tracker，并把当前概念压缩为最小可执行形式，第二种边界自然导出一个 context-conditioned residual policy。

#### 候选数学形态：Context-Conditioned Action Residual

令冻结 Tracker 为 \(\pi_0\)，当前历史或本体状态为 \(h_t\)，Planner Intent 为 \(Y_t\)，则 Nominal Action 为：

\[
a_t^0=\pi_0(h_t,Y_t).
\]

真实校准轨迹只用于产生临时 Context：

\[
z=E_\phi(\tau_{\mathrm{cal}}).
\]

候选残差网络输出：

\[
\Delta a_t
=
\pi_{\mathrm{res},\theta}
(h_t,Y_t,a_t^0,z),
\qquad
a_t=a_t^0+\Delta a_t.
\]

这一形式给出了明确的权限划分：

- \(\pi_0\) 保留已有的 Intent-to-Action 主映射；
- \(E_\phi\) 只从真实轨迹识别当前执行条件；
- \(\pi_{\mathrm{res},\theta}\) 根据当前 Intent、状态、Nominal Action 和 Context 修改最终 Action；
- 部署时 \(\pi_0\)、\(E_\phi\) 与 \(\pi_{\mathrm{res},\theta}\) 均保持冻结，变化的只有临时 Context 与网络激活。

Context \(z\) 不是一个固定“加力值”，而是选择一张残差规律。同一个“左腿响应偏弱”Context，在直线与转身中可以输出不同修正，因为当前 Intent、状态和 Nominal Action 不同。如果 residual network 只读取 Context，它无法形成 Intent-dependent correction。

这一候选方法与 FADA 的区别是适配发生的位置不同：

```text
FADA:
real trajectory -> supervised gradient -> low-rank Delta W -> adapted IDM

Candidate:
real trajectory -> temporary context z -> residual policy -> Delta Action
```

必须保留此前对 LoRA 数学对象的修正：FADA 的 LoRA 是网络内部的逐层低秩参数更新，不是 action residual；这里的 \(\pi_{\mathrm{res}}\) 才是显式输出层 action residual。

Additive residual 本身也是一个可证伪的物理假设：目标机器人所需的逆动力学位于源 Tracker 附近，Sim2Real gap 可以由局部动作修正表达。如果真实变化会改变接触模式、动作方向或正确运动 Intent，简单的 \(a_t^0+\Delta a_t\) 可能不再充分。

#### 候选离线训练闭环

残差网络不能通过模仿失败 rollout 中的 Action 获得正确修正，因为这些 Action 正是失败执行的一部分。更完整的候选训练单位是同一个隐藏仿真物理条件下的一对 rollout：

```text
sample Intent I and randomized dynamics M

Support rollout:
frozen Tracker -> M -> tau_support -> Context Encoder -> z

Query rollout under the same M:
frozen Tracker + pi_res(..., z) -> task return / successful execution
```

Support rollout 负责暴露“该机器人怎样响应动作”；Query rollout 的任务回报、成功监督或 privileged teacher 才负责提供修正方向。训练阶段只更新 \(E_\phi\) 与 \(\pi_{\mathrm{res},\theta}\)，不更新基础 Tracker。若未来要求一个网络服务多个 Intent，训练数据还需要交叉覆盖：同一个物理条件对应多个 Intent，同时同一个 Intent 对应多个物理条件，避免把任务特征错误编码为物理 Context。当前第一阶段不要求跨 Intent 闭合。

#### 应当制造什么仿真轨迹

域随机化不应直接追求“轨迹越多越好”。训练 Support 必须由与部署一致的因果链产生：

```text
Intent -> frozen Tracker -> Action -> randomized Robot Dynamics -> Observation
```

因此，需要覆盖的是不同的 **Action-to-Observation response**，而不是任意运动学轨迹或独立动作噪声。真实物理证据至少存在于逐时刻的：

```text
previous Observation -> executed Action -> next Observation
```

在尚未证明充分统计量之前，这些逐时刻、逐关节、带符号的响应不应被压缩成单个误差范数。质量、阻尼、摩擦、延迟、电机增益或接触参数即使不同，只要它们在当前任务中产生相同的可观察执行效果并需要相同修正，就可以属于同一个功能等价类。

轨迹条件残差可学习的核心要求是：

\[
\tau_1\approx\tau_2
\quad\Longrightarrow\quad
\Delta a_1^*\approx\Delta a_2^*.
\]

如果两段不可区分的 Support trajectory 需要方向相反的最优 residual，那么一条轨迹在信息上不足，增加网络容量也无法解决。合理的仿真轨迹族至少需要满足：

1. **功能覆盖**：包含任务相关的多种可观察执行偏差；
2. **桥接连续性**：相邻物理效果之间存在可以插值的轨迹；
3. **可恢复性**：冻结 Tracker 加有限 residual 后仍然能够完成任务；
4. **访问相关性**：覆盖该 Intent 实际访问的状态、关节、接触与步态相位。

#### 仿真覆盖真机与性能下界

纯仿真不能无条件证明自己覆盖了真机。数学不等式只能把覆盖假设传递到性能结论，不能凭空创造覆盖关系。更精确地说，域随机化先采样物理条件：

\[
M\sim\mu(M),
\qquad
\tau\sim p(\tau\mid M,I,\pi_0).
\]

真实轨迹不必在仿真中逐点出现；它只需落在一个“邻近轨迹需要邻近 residual”的功能区域。可以用真实轨迹定义一个与观测相容的候选物理集合：

\[
\mathcal U_\alpha(\tau_{\mathrm{real}})
=
\{M:\ M\text{ 与 }\tau_{\mathrm{real}}\text{ 的执行响应相容}\}.
\]

随后在该集合内定义候选性能下界：

\[
J_m(\pi\mid\tau_{\mathrm{real}})
=
\inf_{M\in\mathcal U_\alpha(\tau_{\mathrm{real}})}
J(\pi;M,I).
\]

只有当真实机器人以所声明的置信度位于 \(\mathcal U_\alpha\) 中时，\(J_m\) 才能解释为真实性能下界。一条真实任务轨迹可以用于识别 Context，并检查自己是否接近离线学习的响应流形；它不能证明所有真实任务或所有未激励动力学均已被覆盖。

#### 当前概念状态

- **已收缩的范围**：第一阶段研究同一 Intent 的真实轨迹条件执行校准，不再把跨 Intent 成功设为必要前提；
- **候选主变量**：trajectory-conditioned, Intent-dependent action correction；
- **候选最小方法**：冻结 Tracker、轨迹 Context Encoder 与 context-conditioned action residual；
- **候选训练闭环**：同一随机物理条件下的 Support rollout 与 Query rollout；
- **不可缺少的可证伪条件**：相近执行轨迹必须对应相近最优 residual；
- **未确认**：Additive action residual 是否足以表达真实 humanoid execution gap；
- **未确认**：应如何构造或校准 \(\mathcal U_\alpha(\tau)\)，以及如何证明真实轨迹处于其支持范围；
- **未确认**：FADA 真实 Action 的底层控制语义、完整 rollout 字段和 IDM LoRA 样本时间对齐；
- **治理边界**：本节记录的是已审阅的讨论与候选方法，不是活动实现合同，不触发代码、训练或 Concept Figure 更新。
