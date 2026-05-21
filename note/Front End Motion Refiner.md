# On-line Motion Refiner


- 架构：
    - FrontRES 输出ΔSE3(task-space correction)+Conf_pos+Conf_rpy
    - 执行约束：jump gate、Δz upward block (屏蔽向上输出修正)
- 扰动项：
    - xy/yaw
    - z/rp
    - sink/float/root tilt/IID/OU/local artifact
    - 扰动是否与动作空间和 reward 对齐
- 更新机制 & 奖励项：
    - executability margin
    - repair gain / repair ratio
    - stability margin
    - task preservation
    - action cost
    - double-sigmoid repairability window `mu`
    - harmful repair penalty
    - behavior fitting diagnostics
- 训练 schedule：
    - warmup
    - DR scale
    - actor takeover
</aside>

<aside>
💡

**Split Reference Branch**

`R_perturbed = noisy-GMT reference 的 executable score`

`R_frontres  = FrontRES 修正后 reference 的 executable score`

`R_feasible  = active cone 下 oracle 最优修正后的 executable score`

</aside>

<aside>
💡

**Gap & Gain & Ratio**

`damage_gap  = max(R_feasible - R_perturbed, 0)`

`repair_gain = R_frontres - R_perturbed`

`repair_ratio = repair_gain / max(damage_gap, gap_floor)`

</aside>

<aside>
💡

**Sample Regions**

当前代码不再使用旧的 `safe_gate / fragile_gate / broken_gate / bad_repair_gate` 参与 reward。

现在只保留一个连续修复窗口权重：

`mu = repairability window weight`

它回答一个问题：

`当前样本有多适合作为“学习修复”的样本？`

硬分区只作为诊断项：

`safe   : damage_gap < safe_gap`

`repair : safe_gap <= damage_gap <= broken_gap`

`broken : damage_gap > broken_gap`

其中：

`safe` 表示几乎没坏，期望 FrontRES 少动。

`repair` 表示损伤在可修范围内，期望 FrontRES 提高 executability。

`broken` 表示损伤太深，期望 FrontRES 不要硬修。

</aside>

<aside>
💡

**Executable Score**

`planar   -> xy_score`

`yaw      -> yaw_score`

`global_z -> z_score`

`local_rp -> rp_score`

`xy_score:`

`anchor xy tracking`

`anchor xy velocity`

`foot phase xy consistency`

`yaw_score:`

`anchor yaw tracking`

`yaw rate tracking`

`z_score:`

`anchor z tracking`

`end-effector z consistency`

`rp_score:`

`command anchor roll/pitch 与 robot anchor roll/pitch 的误差`

</aside>

<aside>
💡

**Double-Sigmoid Repairability Window**

`enter = sigmoid((damage_gap - safe_gap) / temp)`

`exit  = sigmoid((broken_gap - damage_gap) / temp)`

`mu_raw = enter * exit`

为了避免两个 sigmoid 相乘后峰值小于 1，代码会用窗口中心处的峰值做归一化：

`mu = clamp(mu_raw / peak, 0, 1)`

直观含义：

`damage_gap < safe_gap  -> mu ≈ 0`

`safe_gap < damage_gap < broken_gap -> mu ≈ 1`

`damage_gap > broken_gap -> mu ≈ 0`

因此 `mu` 是唯一的主 reward gate：

`exec_gate = mu`

`cost_gate = 1 - mu`

</aside>

<aside>
💡

**Harmful Repair Penalty**

FrontRES 不能只追求平均 gain，它还必须避免“修正后比不修更差”。

`repair_gain = R_frontres - R_perturbed`

如果：

`repair_gain < 0`

说明 FrontRES 输出了 harmful correction。

当前代码用连续幅度惩罚，而不是用不可导/不稳定的二值 rate：

`raw_harm = relu(-repair_gain - eps_harm)`

其中：

`eps_harm` 是容忍区，默认 `0.001`，用于忽略 executable score 的微小噪声。

但是，`repair_gain < 0` 本身还不能证明 FrontRES 做了“有害修正”。

如果 FrontRES 几乎 no-op，`R_frontres - R_perturbed` 的轻微负值可能只是分支评估噪声。

因此当前代码额外使用 action-conditioned harm gate：

`action_harm_gate = clamp((action_cost - harm_action_cost_floor) / (harm_action_cost_ref - harm_action_cost_floor), 0, 1)`

`harm = raw_harm * action_harm_gate`

含义：

只有当 FrontRES 真的输出了足够大的修正时，负 gain 才会被当成 harmful repair。

harm 惩罚权重：

`harm_weight = mu + side_harm_weight * (1 - mu)`

含义：

`mu` 区域修坏：强惩罚。

`safe/broken` 区域修坏：也惩罚，但可用 `side_harm_weight` 控制强度。

默认：

`frontres_harm_epsilon = 0.001`

`frontres_harm_penalty_weight = 0.25`

`frontres_side_harm_weight = 0.0`

`frontres_harm_action_cost_floor = 0.001`

`frontres_harm_action_cost_ref = 0.01`

</aside>

<aside>
💡

**Final Reward**

`r_delta =`

`mu * exec_signal`

`- harm_penalty`

`- (1 - mu) * intervention_cost`

`+ optional geometry/rescue terms`

当前主训练配置中：

`exec_signal = repair_gain`

`harm_penalty = harm_penalty_weight * harm_weight * relu(-repair_gain - eps_harm) * action_harm_gate`

`intervention_cost = normalized weighted action magnitude`

`geometry/rescue terms` 默认关闭，用于保持主信号干净。

注意：

旧公式中的 `fragile_gate`、`bad_repair_gate`、`safe_cost / broken_cost` 已经不是主 reward 机制。

</aside>

<aside>
💡

**Actor Gate / PPO Sample Weight**

注意：这里的 `actor_gate` 不是 reward gate。

我们取消的是旧的复杂 reward gating：

`safe_gate / damage_gate / broken_gate / fragile_gate / bad_repair_gate`

当前保留的 `actor_gate` 只是 PPO actor loss 的样本权重。

`mu` 控制 reward 中哪些样本提供修复信号。

`actor_gate` 控制 PPO 更新时每个样本对 actor 参数更新的相对贡献。

当前：

`actor_gate = mu + side_actor_weight * (1 - mu)`

默认：

`frontres_side_actor_gate_weight = 0.05`

含义：

`repair` 样本：正常更新 actor。

`safe/broken` 样本：只用很小权重更新 actor，避免无效样本主导 PPO。

PPO actor loss 使用按权重总和归一化：

`surrogate_loss = sum(actor_gate * surrogate_i) / sum(actor_gate)`

因此 `actor_gate` 改变样本相对权重，而不是简单缩小整个 batch 的学习率。

</aside>

<aside>
💡

**Console Diagnostics**

下面是当前控制台 `FRONTRES` 区域会输出的完整诊断项。

我建议把这些指标分成四类：

`Performance`

运行性能、episode 长度、基线分支和当前扰动强度。它们回答：

`训练跑得是否正常？当前 batch 的环境难度是多少？GMT baseline 是什么水平？`

---

`Main Reward`

主 reward 和主行为质量指标。它们回答：

`FrontRES 是否真的把扰动 reference 修得更可执行？是否产生了有害修正？`

---

`Detail Reward`

reward 的细分来源。它们回答：

`收益或损失来自 z、xy、roll/pitch、yaw，还是来自 action cost / rescue / geometry 项？`

---

`Optimization / Update`

优化与更新诊断项。它们不直接属于 reward，而是描述 PPO / supervised / curriculum / sample weighting 的训练过程。它们回答：

`当前 actor 是怎么被更新的？PPO 和 supervised 是否冲突？哪些样本在主导更新？`

</aside>

<aside>
💡

**Console Diagnostics: Performance**

`Learning iteration`

当前训练迭代数。

格式：

`current_iteration / max_iterations`

用于确认训练进度以及当前是否处于 warmup、actor takeover 或 PPO fine-tuning 阶段。

---

`PHASE`

当前训练阶段。

常见值：

`SUPERVISED WARMUP`

`CRITIC WARMUP`

`ACTOR TAKEOVER`

`PPO + SUPERVISED ANCHOR`

`PPO + WEAK SUPERVISION`

`PPO FINE-TUNING`

括号中的说明会提示当前是否固定 DR、是否冻结 PPO actor、是否正在 ramp PPO actor weight。

---

`Computation`

格式：

`steps/s (collection time, learning time)`

含义：

训练吞吐量、采样耗时、学习耗时。主要用于发现仿真或学习是否异常变慢。

---

`Mean action noise std`

当前 policy action distribution 的平均噪声标准差。

值越大，探索越强；值越小，动作越确定。

---

`Mean episode length`

FrontRES 当前训练分支的平均 episode 长度。

这不是 GMT baseline，而是带 FrontRES 分支的 episode 长度。

---

`ep_len_GMT (baseline)`

冻结 GMT baseline 分支的平均 episode 长度。

如果 `Mean episode length` 明显低于 `ep_len_GMT`，说明 FrontRES 可能让执行更差。

如果明显高于 `ep_len_GMT`，说明 FrontRES 可能提高了可执行性。

---

`perturb curriculum`

当前扰动课程。

示例：

`single [global_z,local_rp,planar,yaw]`

前半部分是扰动复杂度，后半部分是当前启用的扰动家族。

---

`DR scale`

当前扰动强度缩放。

它是判断 batch 难度的关键指标。`broken_frac` 很高时，先看 `DR scale` 是否过大。

---

`survival rate`

当前 batch 中 episode 没有提前失败的比例。

越接近 1，说明整体还可执行；明显下降时，可能扰动太强或 FrontRES 输出有害修正。

</aside>

<aside>
💡

**Console Diagnostics: Main Reward**

`r_delta (FrontRES)`

FrontRES 当前训练分支的平均 episode reward。

这是训练侧最直接的总体 reward 观测，但它包含环境 episode 的整体结果，不如 `gap/gain/fit/harm` 精确。

---

`reward_GMT (baseline)`

GMT baseline 的平均 episode reward。

和 `r_delta` 配合看：

`r_delta > reward_GMT`：FrontRES 分支整体更好。

`r_delta < reward_GMT`：FrontRES 分支整体更差。

---

`supervised_cos_sim`

FrontRES 输出方向和 supervised anti-perturbation target 的余弦相似度。

它看的是“方向是否像人工标签”，不是“执行后是否真的更好”。

---

`|Δpos|`

FrontRES 输出的位置修正平均幅度，单位 `m`。

过大通常意味着修正激进；过小可能说明没有学会有效修复，或当前 batch 大多是 safe。

---

`|Δrpy|`

FrontRES 输出的姿态修正平均幅度，单位 `rad`。

用于判断 roll/pitch/yaw 修正是否异常放大。

---

`gap/gain/ratio`

对应：

`damage_gap / repair_gain / repair_ratio`

`damage_gap`：扰动造成了多少 executable damage。

`repair_gain`：FrontRES 相对 perturbed baseline 提高了多少 executable score。

`repair_ratio`：`repair_gain / damage_gap` 的归一化诊断值，不再参与 gate。

阅读方式：

`gap > 0 且 gain > 0`：有损伤，FrontRES 正在修。

`gap > 0 但 gain < 0`：有损伤，但 FrontRES 修坏了。

`gap 很小`：样本本来就 safe，不能期待大 gain。

---

`signal/w_signal/train_r`

对应：

`exec_signal / weighted_exec_signal / train_reward`

`exec_signal`：未乘窗口权重前的执行收益信号，当前主要等于 `repair_gain`。

`weighted_exec_signal`：乘以 `mu` 后的执行收益信号。

`train_reward`：最终写入 FrontRES 训练的单步 reward，已经包含 harm penalty 和 action cost。

---

`fit total/rate/gain`

对应：

`behavior_fit / repair_fit_rate / repair_fit_gain`

`behavior_fit`：总拟合度，衡量当前 batch 中 FrontRES 的实际行为是否符合期望。

公式概念：

`修复收益 - 有害修正 - 窗口外乱动成本`

再除以当前 batch 中可解释的损伤与成本规模。

`repair_fit_rate`：可修窗口内，FrontRES 修掉了多少比例的 executable damage。

这是最像“训练集拟合度 / accuracy”的指标。

`repair_fit_gain`：可修窗口内平均 executable gain。

它不会被 repair 样本比例稀释，比 `weighted_exec_signal` 更适合看修复能力本身。

---

`harm rate/mag`

对应：

`harm_rate / harm_mag`

`harm_rate`：可修窗口内，有多少样本被 FrontRES 修坏。

`harm_mag`：可修窗口内，经过 action gate 后的平均有害修正幅度。

理想状态：

`harm_rate` 接近 0，`harm_mag` 接近 0。

---

`safe/broken harm`

对应：

`safe_harm_rate / broken_harm_rate`

`safe_harm_rate`：本来几乎没坏的 safe 样本，有多少被 FrontRES 修坏。

`broken_harm_rate`：深度损坏的 broken 样本，有多少被 FrontRES 硬修得更差。

这两个指标用于检查 FrontRES 是否在不该修的区域乱动。

---

`safe/broken abstain`

对应：

`safe_abstain_cost / broken_abstain_cost`

衡量 FrontRES 在 safe / broken 样本上是否保持克制。

值越低，说明窗口外越接近 no-op。

---

`positive_gain_frac`

`repair_gain > 0` 的样本比例。

它只看正负，不看样本是否处于可修窗口，也不看有害幅度。

现在它是辅助指标，优先级低于 `fit total/rate/gain` 和 `harm rate/mag`。

---

`safe/repair/broken frac`

当前 batch 的硬分区比例。

`safe`：损伤太小，不需要明显修复。

`repair`：损伤处于可修窗口，是最重要的学习样本。

`broken`：损伤太深，不应强行修复。

阅读方式：

`repair_frac` 太低：真正可学习样本不足。

`broken_frac` 太高：扰动太难，优先降低 DR 或简化 curriculum。

`safe_frac` 太高：样本太容易，FrontRES 主要学 no-op。

</aside>

<aside>
💡

**Console Diagnostics: Detail Reward**

`r_z/r_xy/r_rp/r_yaw`

对应四个 active cone 方向的 executable reward 分量：

`r_z`：竖直方向修复质量。

`r_xy`：平面位置修复质量。

`r_rp`：roll / pitch 姿态修复质量。

`r_yaw`：yaw 方向修复质量。

用途：

判断当前 reward 是否和扰动家族、动作空间对齐。

---

`repair/geom/rescue/action_cost`

对应：

`r_exec / r_geom / r_rescue / intervention_cost`

`repair`：核心执行修复 reward。

`geom`：几何一致性辅助项。

`rescue`：从坏状态拉回可执行区域的辅助项。

`action_cost`：FrontRES 修正幅度成本。

当前主配置里，`repair` 是主信号，`geom/rescue` 通常是弱项或关闭。

---

`exec reward FR/pert`

对应：

`R_frontres / R_perturbed`

`R_frontres`：FrontRES 修正后的 executable score。

`R_perturbed`：不经过 FrontRES 的 perturbed baseline executable score。

它们的差就是：

`repair_gain = R_frontres - R_perturbed`

---

`exec planar/vertical/task`

执行性 reward 的三类聚合：

`planar`：平面运动可执行性。

`vertical`：高度、接触、roll/pitch 相关可执行性。

`task`：弱任务一致性项，用于避免 trivial no-motion。

用途：

判断 reward 改善来自真正的运动可执行性，还是来自任务项偏置。

---

`exec/cost gate`

对应：

`exec_gate / cost_gate`

当前基本等于：

`exec_gate = mu`

`cost_gate = 1 - mu`

它是调试项，用于确认 reward window 是否按预期工作。

</aside>

<aside>
💡

**Console Diagnostics: Optimization / Update**

这些指标不是 reward 项，建议统一叫：

`Optimization / Update Diagnostics`

中文可以叫：

`优化与更新诊断项`

它们描述的是“训练如何更新参数”，而不是“当前样本得到多少 reward”。

---

`mu (reward window)`

对应：

`window_mu`

`mu`：平均 repairability window 权重，也是主 reward window。

---

`actor sample weight`

对应：

`actor_gate`

`actor_gate`：PPO actor loss 的样本权重，不是 reward gate。

当前：

`actor_gate = mu + side_actor_weight * (1 - mu)`

PPO actor loss 使用权重总和归一化：

`surrogate_loss = sum(actor_gate * surrogate_i) / sum(actor_gate)`

因此它改变样本相对贡献，不是简单缩小整个 batch 的步长。

---

`grad cos PPO/Sup`

对应：

`grad_cos_ppo_supervised`

括号中的：

`norm ratio`

对应：

`grad_norm_ratio_ppo_to_supervised`

含义：

PPO actor 梯度和 supervised 梯度的方向关系。

`grad cos > 0`：两者大致同向。

`grad cos < 0`：两者冲突。

`norm ratio` 很大：PPO 梯度幅度远大于 supervised，可能冲掉 warmup 学到的修正方向。

---

`r_delta EMA`

`r_delta` 的指数滑动平均。

比单个 iteration 的 `r_delta` 更平滑，用于观察整体趋势。

---

`λ_supervised`

supervised loss 的当前权重。

越大，actor 越受 supervised anti-perturbation target 约束。

越小，actor 越主要由 PPO reward 驱动。

---

`PPO actor weight`

PPO actor surrogate loss 的当前权重。

在 actor takeover 阶段会逐渐 ramp up。

如果该值升得太快，而 `norm ratio` 又很大，容易出现 PPO 把 supervised 方向冲掉的问题。

</aside>

<aside>
💡

**Console Diagnostics: Compact Cheat Sheet**

这一节是快速查阅版，只保留“控制台行名 -> 含义 -> 理想阅读方式”。

---

## Performance

`Learning iteration`

训练进度：`当前迭代 / 总迭代`。

---

`PHASE`

当前训练阶段。重点看是否处于 `SUPERVISED WARMUP`、`CRITIC WARMUP`、`ACTOR TAKEOVER` 或 PPO fine-tuning。

---

`Computation`

仿真和学习速度。异常变慢时先查环境、采样或 learner。

---

`Mean action noise std`

policy 探索强度。太大可能动作抖，太小可能探索不足。

---

`Mean episode length`

FrontRES 分支平均 episode 长度。

---

`ep_len_GMT (baseline)`

GMT baseline 平均 episode 长度。和 `Mean episode length` 对比判断 FrontRES 是否让执行更稳定。

---

`reward_GMT (baseline)`

GMT baseline 平均环境 reward。和 `r_delta (FrontRES)` 对比判断 FrontRES 是否带来整体收益。

---

`perturb curriculum`

当前扰动复杂度和扰动家族。用于判断训练信号是否和 active action cone 对齐。

---

`DR scale`

当前扰动强度。`broken_frac` 高时优先检查它是否过大。

---

`survival rate`

存活率。下降说明扰动、修正或训练更新可能正在破坏可执行性。

---

## Main Reward

`r_delta (FrontRES)`

FrontRES 分支平均 episode reward。总体结果指标，但不够细。

---

`supervised_cos_sim`

FrontRES 输出方向和 supervised target 的相似度。看“像不像标签”，不等于“执行是否更好”。

---

`|Δpos|`

位置修正幅度。过大说明修正激进，过小可能没有有效修复。

---

`|Δrpy|`

姿态修正幅度。用于观察 roll / pitch / yaw 输出是否异常。

---

`gap/gain/ratio`

`damage_gap / repair_gain / repair_ratio`。

核心判断：

`gap > 0, gain > 0`：FrontRES 正在修。

`gap > 0, gain < 0`：FrontRES 修坏了。

---

`signal/w_signal/train_r`

`exec_signal / weighted_exec_signal / train_reward`。

看 reward 从原始执行收益到窗口加权，再到最终训练 reward 的变化。

---

`fit total/rate/gain`

`behavior_fit / repair_fit_rate / repair_fit_gain`。

最重要的拟合度指标：

`behavior_fit` 看总体行为是否正确。

`repair_fit_rate` 看可修窗口内修掉了多少 damage。

`repair_fit_gain` 看可修窗口内平均 gain。

---

`harm rate/mag`

可修窗口内的有害修正比例和有害幅度。越低越好。

---

`safe/broken harm`

safe / broken 区域的有害修正比例。用于发现不该动时乱动、坏样本上硬修。

注意：

当前它只统计 action-conditioned harm。

如果 `safe/broken abstain` 很低，那么 no-op 附近的 score 噪声不会再被算成 harmful repair。

---

`safe/broken abstain`

safe / broken 区域的动作成本。越低越说明窗口外接近 no-op。

---

`positive_gain_frac`

`repair_gain > 0` 的比例。辅助指标，不如 `fit` 和 `harm` 可靠。

---

`safe/repair/broken frac`

batch 难度分布。

`repair_frac` 太低：有效学习样本不足。

`broken_frac` 太高：扰动太难。

`safe_frac` 太高：样本太容易。

---

## Detail Reward

`r_z/r_xy/r_rp/r_yaw`

四个方向的 executable reward 分量。用于定位哪个修正方向出了问题。

---

`repair/geom/rescue/action_cost`

核心修复收益、几何辅助、救援辅助、动作成本。

主配置优先看 `repair` 和 `action_cost`。

---

`exec reward FR/pert`

`R_frontres / R_perturbed`。

两者差值就是 `repair_gain`。

---

`exec planar/vertical/task`

平面、竖直/接触、任务一致性三类 executable score。

用于判断 reward 是否被某一类项主导。

---

`exec/cost gate`

`exec_gate / cost_gate`，当前基本是 `mu / (1 - mu)`。

这是调试窗口权重用的指标。

---

## Optimization / Update

`mu (reward window)`

`mu` 是 reward window。

---

`actor sample weight`

`actor_gate` 是 PPO actor loss 的样本更新权重，不是 reward gate。

---

`actor/exec/cost`

`actor_gate / exec_gate / cost_gate`。

这是紧凑调试行，用于同时查看 PPO 样本权重、执行 reward 窗口、动作成本窗口。

---

`grad cos PPO/Sup`

PPO 梯度和 supervised 梯度的方向相似度。

括号里的 `norm ratio` 表示 PPO 梯度幅度相对 supervised 的倍数。

`cos < 0` 或 `norm ratio` 极大时，PPO 可能冲掉 supervised 方向。

---

`r_delta EMA`

`r_delta` 的滑动平均。比单次 iteration 更适合看趋势。

---

`λ_supervised`

supervised loss 权重。越大，越约束 FrontRES 保持 supervised 修正方向。

---

`PPO actor weight`

PPO actor loss 权重。actor takeover 阶段会 ramp up。

如果它升太快，同时 `norm ratio` 很大，训练容易不稳。

</aside>

<aside>
💡

**Console Diagnostics: Reading Priority**

如果只想快速判断训练状态，优先看：

`gap/gain/ratio`

`fit total/rate/gain`

`harm rate/mag`

`safe/repair/broken frac`

`mu (reward window)`

`actor sample weight`

`grad cos PPO/Sup`

`reward_GMT` vs `r_delta`

如果要定位 reward 为什么异常，再看：

`r_z/r_xy/r_rp/r_yaw`

`repair/geom/rescue/action_cost`

`exec planar/vertical/task`

`exec reward FR/pert`

如果要定位训练是否被 PPO 冲坏，再看：

`λ_supervised`

`PPO actor weight`

`grad cos PPO/Sup`

`norm ratio`

</aside>

<aside>
💡

**哪些指标保留，哪些可以降级**

## 必须保留

`fit total/rate/gain`

`harm rate/mag`

`safe/broken harm`

`safe/broken abstain`

`gap/gain/ratio`

`safe/repair/broken frac`

`mu (reward window)`

`actor sample weight`

`grad cos PPO/Sup`

`supervised_cos_sim`

`|Δpos| / |Δrpy|`

## 调试期保留，稳定后可删除

`exec/cost gate`

因为它们现在基本等于 `mu / (1 - mu)`。

`signal/w_signal/train_r`

它能看 reward 组成，但不如 `fit total/rate/gain` 适合评估拟合度。

`positive_gain_frac`

它只看正负比例，不看窗口权重，也不看有害幅度。现在可由 `harm_rate` 和 `repair_fit_rate` 部分替代。

## 不再使用

`safe_gate`

`fragile_gate`

`damage_gate`

`broken_gate`

`bad_repair_gate`

`safe_cost / fragile_cost / broken_cost`

这些属于旧 gating 设计，不应再用来解释当前训练日志。

</aside>


<aside>
💡

**Final Loss**

`loss =`

`ppo_actor_weight * surrogate_loss`

`+ value_loss_coef * value_loss`

`+ entropy_coef * entropy` 

`+ lambda_supervised * supervised_loss`

</aside>
