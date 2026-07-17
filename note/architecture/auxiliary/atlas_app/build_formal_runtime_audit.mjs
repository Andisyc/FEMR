import fs from "node:fs";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "../..");
const output = path.join(root, "runtime/04_stage3_formal_runtime_audit.data.json");

const specs = [
  ["AUDIT-ROUTE-01", "正式 Stage 3 路由", "M-03, M-05, SR-01", "正式 train 身份进入 live update loop", "source/rsl_rl/rsl_rl/runners/frontres_segment_live_training.py", "run_frontres_segment_live_training_loop()", ["stage/task/mode/checkpoint", "live_train/alternate_modes", "iteration/update loop"]],
  ["AUDIT-PERTURB-01", "扰动配置", "M-02", "specialist family 与 DR scale 进入正式训练", "scripts/rsl_rl/train.py", "_configure_frontres_motion_perturbations()", ["specialist_mode", "family config", "DR scale/schedule"]],
  ["AUDIT-PERTURB-02", "实际扰动", "M-02, Q-PAIR", "实际 rollout 行收到预期 local_rp 扰动", "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/motion_perturbations.py", "MotionPerturber.apply_perturbations()", ["family row mask", "applied root-position delta", "DR scale/finite stats"]],
  ["AUDIT-SEGDATA-01", "Segment 数据身份", "SR-01", "cache source 与 sampled segment 身份可追溯", "source/rsl_rl/rsl_rl/frontres/frontres_segment_dataset.py", "FrontRESSegmentDataset.get_segments()", ["sampled segment ids", "state/reference/cache horizon/family batch", "sampler batch consumer"]],
  ["AUDIT-SAMPLER-01", "Segment Replay 事务", "SR-01", "sample、rollout evidence 与 priority update 同源", "source/rsl_rl/rsl_rl/frontres/frontres_segment_sampler.py", "sample()/update_with_probe()", ["sample state", "rollout evidence", "priority before/after"]],
  ["AUDIT-KPLAN-01", "K-step 计划", "M-06, SR-01", "curriculum K 被展开为 per-row rollout budget", "source/rsl_rl/rsl_rl/frontres/frontres_segment_sampler.py", "plan_rollout_budget()/expand_rollout_trials()", ["curriculum max K", "per-row horizon_k", "expanded trial rows"]],
  ["AUDIT-KROLLOUT-01", "K-step 执行", "M-06, Q-PAIR", "expanded trial rows 将同一 K 交给 reset 和 rollout", "source/rsl_rl/rsl_rl/frontres/frontres_segment_sampler.py", "expand_rollout_trials()", ["base segment budget", "expanded source/trial/K rows", "reset/rollout consumer rows"]],
  ["AUDIT-RESET-LIFECYCLE-01", "Reset Lifecycle", "M-06, Q-PAIR, SR-01", "index reset 后四类 role 共享可比较的 episode 与 dynamic-state 起点", "source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py", "_run_live_rollout_capture()", ["reset 前/随机化后/reset 后 episode_length_buf", "quartet origin-relative root/joint pair error", "逐步 role done/timeout/termination/survival 与 active term masks"]],
  ["AUDIT-ANCHOR-Z-01", "Anchor Z Termination", "M-06, Q-PAIR, M-10", "逐 role 定位 reference anchor z 与 robot torso z 的首个错位对象", "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py", "bad_anchor_pos_z_only()", ["world-frame reference_z/robot_z 与 frame identity", "clean/raw/correction z 分解和 signed/abs error", "0.5m threshold 前的原始 termination mask"]],
  ["AUDIT-OBS-01", "870D Observation", "M-04, M-10", "100D balance prefix 与 770D GMT suffix 保持布局", "source/rsl_rl/rsl_rl/runners/frontres_runtime.py", "apply_obs_normalizer()", ["raw obs[*,870]", "prefix100/suffix770", "normalized finite obs"]],
  ["AUDIT-ACTION-01", "Full-6D Actor", "M-04", "mean、sigma 与 sampled action 保持 full-6D 同源", "source/rsl_rl/rsl_rl/modules/front_residual_actor_critic.py", "update_distribution()/act()", ["policy obs", "mean/sigma[*,6]", "sampled action[*,6]"]],
  ["AUDIT-APPLY-01", "Delta SE(3) 应用", "M-04, M-10", "完整 6D repair 写入 repaired reference", "source/rsl_rl/rsl_rl/frontres/task_space_correction.py", "apply_frontres_task_corrections()", ["raw Delta SE(3)", "task correction", "repaired reference"]],
  ["AUDIT-GMT-01", "Frozen GMT", "M-10", "GMT 执行 repaired reference 且参数保持冻结", "source/rsl_rl/rsl_rl/modules/front_residual_actor_critic.py", "get_env_action()", ["GMT observation", "requires_grad/optimizer exclusion", "GMT execution/checksum"]],
  ["AUDIT-PAIR-01", "Quartet Role 布局", "Q-PAIR, SR-01", "Clean/Noisy/Repaired/Train 行身份与 reset 对齐", "source/rsl_rl/rsl_rl/runners/frontres_training_setup.py", "configure_frontres_pair_layout()", ["pair layout", "trial role rows", "reset/valid role counts"]],
  ["AUDIT-PAIR-EVIDENCE-01", "Paired Execution Evidence", "Q-PAIR, Q-01", "同 segment/K 且同 audit transaction 的 Noisy 与 Repaired 证据可比较", "source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py", "_capture_motion_quality_frame()/_capture_physics_frame()", ["shared segment/K", "Noisy/Repaired execution", "paired style/physics evidence", "audit transaction/batch signature"]],
  ["AUDIT-GAIN-01", "Canonical Repair Gain v002", "Q-01", "同一 capture identity 下 raw survival、effective K、survival quality 与 Style/Physics/Repair 合成为 Total Gain", "source/rsl_rl/rsl_rl/frontres/frontres_gain.py", "compute_segment_gain()", ["raw survival_steps + effective_horizon_K", "survival_quality repaired/noisy + physics_survival_gain", "survival_gain_step_sum + gain_total", "same capture transaction/batch signature"]],
  ["AUDIT-RETURN-01", "Gain 到 PPO Return v002", "Q-01, M-05", "同一 capture identity 的 canonical Gain 与 K-normalized survival Gain 成为 storage reward、return 和 advantage", "source/rsl_rl/rsl_rl/frontres/frontres_segment_storage.py", "compute_returns_and_advantages()/to_ppo_batch()", ["survival_gain_steps + effective K", "survival_gain_step_sum + returns/advantages", "PPO batch", "same capture transaction/batch signature"]],
  ["AUDIT-HSL-LOAD-01", "HSL 到 Stage 3", "M-03, M-04", "Stage 2 actor 与 observation normalizer 正确进入 Stage 3", "source/rsl_rl/rsl_rl/runners/frontres_checkpointing.py", "load_runner()", ["checkpoint identity", "actor/normalizer state", "Stage 3 live policy"]],
  ["AUDIT-WARMUP-01", "Actor/Critic Warmup", "M-05", "critic-only、actor ramp 与 joint phase 按 iteration 生效", "source/rsl_rl/rsl_rl/frontres/frontres_segment_warmup.py", "frontres_segment_warmup_phase()", ["persisted iteration", "phase/actor weight", "loss gradient boundary"]],
  ["AUDIT-PPO-01", "Segment PPO 更新", "M-05, M-04, Q-01", "old tuple 到 accepted post-update policy 形成闭环", "source/rsl_rl/rsl_rl/algorithms/frontres_segment_ppo.py", "compute_frontres_segment_ppo_loss()", ["old stats/action/advantage", "loss/backward/optimizer step", "post-KL/trust decision"]],
  ["AUDIT-PERSIST-01", "Checkpoint 写盘", "M-03, SR-01, M-05, Q-01", "完整 Stage 3 训练语义写入 model_N.pt", "source/rsl_rl/rsl_rl/runners/frontres_checkpointing.py", "save_runner()", ["base policy payload", "normalizer/sampler/Gain/warmup", "required-key assert/torch.save"]],
  ["AUDIT-DIAG-01", "正式 Diagnostics", "Q-01, M-05", "日志读取 canonical summary, 明确 single transaction 或 update-loop aggregate, 且缺失值不伪装为零", "source/rsl_rl/rsl_rl/frontres/frontres_segment_diagnostics.py", "repair_effect_summary_to_scalars()", ["canonical live summary", "single/aggregate transaction identity", "diagnostic scalars", "terminal/logger consumer"]],
];

const rerun3Reached = new Set([
  "AUDIT-ROUTE-01", "AUDIT-PERTURB-01", "AUDIT-PERTURB-02", "AUDIT-SEGDATA-01",
  "AUDIT-SAMPLER-01", "AUDIT-KPLAN-01", "AUDIT-KROLLOUT-01", "AUDIT-OBS-01",
  "AUDIT-ACTION-01", "AUDIT-APPLY-01", "AUDIT-GMT-01", "AUDIT-PAIR-01",
  "AUDIT-PAIR-EVIDENCE-01", "AUDIT-GAIN-01", "AUDIT-RETURN-01", "AUDIT-HSL-LOAD-01",
  "AUDIT-WARMUP-01",
]);
const runtimeStatus = Object.fromEntries(specs.map(([id]) => [
  id,
  rerun3Reached.has(id)
    ? "runtime-observed: 32-env formal run reached this owner and all quartet rows survived K=8 (E37)"
    : "unconfirmed: rerun3 did not reach this owner",
]));
runtimeStatus["AUDIT-PPO-01"] = "runtime-observed: valid=8, critic-only update observed, trust accepted, KL=6.008e-05 (E37)";
runtimeStatus["AUDIT-RESET-LIFECYCLE-01"] = "runtime-observed: all quartet roles start aligned and survive every K=8 step without termination (E37)";
runtimeStatus["AUDIT-ANCHOR-Z-01"] = "runtime-observed: first-call raw/clean/robot z align, max abs error=0.0201m, anchor_pos=0 for all roles (E37)";
runtimeStatus["AUDIT-DIAG-01"] = "runtime-observed: canonical Gain, valid fraction, warmup, PPO and trust diagnostics are populated (E37)";
runtimeStatus["AUDIT-ROUTE-01"] = "runtime-observed: official route reached absolute iter 700 joint phase at actor_weight=1.0 and completed 4/4 accepted updates (E70)";
runtimeStatus["AUDIT-HSL-LOAD-01"] = "runtime-observed: model_220 full-resume restored actor/normalizer identity at absolute iter 220 (E69)";
runtimeStatus["AUDIT-PERSIST-01"] = "runtime-observed: model_701 saved complete model/optimizer/normalizer/sampler/Gain/warmup payload at iter 701 (E70)";
runtimeStatus["AUDIT-WARMUP-01"] = "runtime-observed: critic and actor warmup progressed into joint phase_iter=0 with actor_weight=1.0 (E68-E70)";
runtimeStatus["AUDIT-PPO-01"] = "runtime-observed: joint batches valid=13/14/16/16 had nonzero parameter delta, accepted trust, and frozen GMT (E70)";
runtimeStatus["AUDIT-KROLLOUT-01"] = "runtime-observed: reset_success_frac=1.0 and valid=7/8 are populated (E39)";
runtimeStatus["AUDIT-APPLY-01"] = "runtime-observed: finite full-6D action and delta_norm=0.016680 (E39)";
runtimeStatus["AUDIT-PAIR-01"] = "runtime-observed: roles policy=8/baseline=24 and valid=7 (E39)";
runtimeStatus["AUDIT-RETURN-01"] = "runtime-observed: reward/return/advantage tensors are finite and populated (E39)";
runtimeStatus["AUDIT-PERTURB-01"] = "runtime-observed: rp, dr_scale=1.25 and max_horizon_k=64 are populated (E41)";
runtimeStatus["AUDIT-PERTURB-02"] = "runtime-observed: local_rp=8 with finite strength min/mean/max (E41)";
runtimeStatus["AUDIT-GAIN-01"] = "runtime-observed: FRS-GAIN-v002 raw/K/quality/step-sum fields are finite with zero sum error (E58)";
runtimeStatus["AUDIT-PAIR-EVIDENCE-01"] = "runtime-observed: paired evidence shares one complete transaction/batch identity with Gain and returns (E67); E68 confirms mixed-K captures";
runtimeStatus["AUDIT-GAIN-01"] = "runtime-observed: FRS-GAIN-v002 components and total share the capture transaction/batch identity (E67/E68)";
runtimeStatus["AUDIT-RETURN-01"] = "runtime-observed: policy-row Gain steps, returns and advantages are finite and transaction-local for K=8..64 (E67/E68)";
runtimeStatus["AUDIT-DIAG-01"] = "runtime-observed: resumed four-capture loop reports aggregate identity with four complete transaction/batch pairs (E69)";

const probeRationales = {
  "AUDIT-ROUTE-01": [
    ["命令刚完成 Stage 3 preset, 这里能确认本次运行身份而不受 runner 后续状态影响", "失败归属: CLI 参数或 Stage 3 preset"],
    ["formal guard 在这里首次排除 sentinel/eval/update_loop 等替代路径", "失败归属: runner route guard 或 boundary config"],
    ["进入 live update loop 前是正式训练承诺点, 再晚只能看到训练结果", "失败归属: train dispatch 到 live loop 的连接"],
  ],
  "AUDIT-PERTURB-01": [
    ["specialist mode 与 DR 参数尚未写入 alg config, 可隔离 CLI 来源错误", "失败归属: CLI/preset 输入"],
    ["这里首次形成正式训练使用的 canonical perturbation config", "失败归属: Stage 3 perturbation configuration"],
    ["sampler/rollout 读取前检查可发现配置存在但未透传的旁路", "失败归属: alg config 到 runtime owner 的转发"],
  ],
  "AUDIT-PERTURB-02": [
    ["干净参考与 per-row family/DR mask 同时可见, 可先证明扰动选择正确", "失败归属: family mask 或 DR scale 来源"],
    ["OU、artifact、IID 合成后 applied delta 第一次完整成立", "失败归属: MotionPerturber.apply_perturbations"],
    ["paired execution 消费前检查可发现正确扰动被覆盖或错分配到 role", "失败归属: perturbation 到 command/paired rollout 的连接"],
  ],
  "AUDIT-SEGDATA-01": [
    ["sampled ids 刚进入 dataset, 可确认 sampler 请求的 segment 身份", "失败归属: sampler 产生的 segment ids"],
    ["state、reference、cache horizon、family 在 batch.validate 后首次构成完整 Segment 数据对象", "失败归属: FrontRESSegmentDataset.get_segments"],
    ["sampler batch builder 读取前检查可发现 batch 身份被重新索引", "失败归属: dataset batch 到 live sampler connector"],
  ],
  "AUDIT-SAMPLER-01": [
    ["采样前的 priority/state 是 source 选择的唯一依据", "失败归属: sampler persistent state"],
    ["sample 返回时 segment/source/K/trial role 首次绑定为同一事务", "失败归属: sample_rollout_rows"],
    ["priority update 后检查可证明更新使用同一次 rollout evidence", "失败归属: update_with_probe 或 evidence wiring"],
  ],
  "AUDIT-KPLAN-01": [
    ["persistent segment state 与 max K 同时可见, 能验证 curriculum 输入", "失败归属: segment state 或 max_horizon config"],
    ["trial_count、horizon_k、reason 在 budget 返回前首次完整对应", "失败归属: plan_rollout_budget"],
    ["trial expansion 前检查可发现 budget 正确但未被采用", "失败归属: budget 到 expansion 的连接"],
  ],
  "AUDIT-KROLLOUT-01": [
    ["base budget 进入 expansion 时可验证每个 source segment 的 K", "失败归属: K plan 输出"],
    ["source_index/trial_index/role/K 在 expanded plan 中首次逐行对齐", "失败归属: expand_rollout_trials"],
    ["reset/rollout 消费前检查可发现 expanded rows 被截断或重排", "失败归属: trial plan 到 reset/rollout connector"],
  ],
  "AUDIT-RESET-LIFECYCLE-01": [
    ["index reset 刚结束且 env.step 尚未发生, 可直接比较每个 role 的 episode 生命周期是否同源", "失败归属: episode_length_buf randomization 或 index-reset lifecycle wiring"],
    ["policy/candidate/noisy/clean 的机器人 dynamic state 在此首次应构成可比较 quartet", "失败归属: index reset robot-state write 或 quartet state synchronization"],
    ["env.step 返回时 done 与 time_out 同时可见, 能区分时限结束和物理 termination", "失败归属: environment termination owner 或 rollout survival accounting"],
  ],
  "AUDIT-ANCHOR-Z-01": [
    ["termination owner 刚读取最终 reference/robot anchor z, 可排除 runner 二次计算差异", "失败归属: command anchor 或 robot anchor source"],
    ["clean/raw/correction 与 frame identity 同时可见, 能定位 cache、correction 或 time-step 错位", "失败归属: command cached reference lifecycle"],
    ["原始 mask 返回前检查可证明数值分解与实际 anchor_pos done 完全同源", "失败归属: bad_anchor_pos_z_only threshold comparison"],
  ],
  "AUDIT-OBS-01": [
    ["raw 870D obs 尚未归一化, 可先验证 100D+770D 布局来源", "失败归属: environment observation construction"],
    ["prefix/suffix 分别归一化后首次形成 actor 实际输入", "失败归属: apply_obs_normalizer 或 persisted stats"],
    ["actor 调用前检查可发现 normalized obs 被替换或再次处理", "失败归属: normalizer 到 policy connector"],
  ],
  "AUDIT-ACTION-01": [
    ["policy obs 刚进入 actor, 可把 observation 错误与 actor 错误分开", "失败归属: observation/normalizer upstream"],
    ["Normal(mean,sigma) 建立后 full-6D distribution 语义第一次完整成立", "失败归属: actor head、std 或 distribution construction"],
    ["act 返回前首次完成 raw sample 到 bounded Delta SE(3) action 的转换", "失败归属: sampling、bounding 或 action representation"],
  ],
  "AUDIT-APPLY-01": [
    ["full-6D task action 尚未写入 command, 可验证执行输入与 actor 输出同源", "失败归属: actor 到 task-correction connector"],
    ["position/rotation correction 写入 command buffer 后执行语义首次成立", "失败归属: apply_frontres_task_corrections"],
    ["GMT 刷新参考前检查可发现 buffer 被清零、截断或 role 错位", "失败归属: command correction 到 GMT execution"],
  ],
  "AUDIT-GMT-01": [
    ["GMT policy obs 与 FrontRES correction 同时可见, 可确认 repaired path 输入", "失败归属: repaired reference 或 GMT observation"],
    ["no_grad GMT 调用后可直接检查冻结执行与 robot action", "失败归属: frozen GMT invocation"],
    ["env.step 前检查可发现 GMT 输出被旧 action 或其他分支覆盖", "失败归属: GMT output 到 environment action connector"],
  ],
  "AUDIT-PAIR-01": [
    ["env 总行数与 quartet 开关刚进入布局 owner, 可验证分区依据", "失败归属: pair-layout config"],
    ["n_train/n_candidate/n_base/n_clean 首次成为 motion-command baseline", "失败归属: configure_frontres_pair_layout"],
    ["reset/capture 前检查可发现 role counts 未同步到 command owner", "失败归属: pair layout 到 reset/capture wiring"],
  ],
  "AUDIT-PAIR-EVIDENCE-01": [
    ["同一 quartet frame 的 Clean/Repaired/Noisy 行尚未聚合, 可验证配对身份", "失败归属: role slicing 或 motion/frame alignment"],
    ["style 与 physics frame 完成后首次具备 canonical Gain 所需配对语义", "失败归属: motion/physics capture owners"],
    ["Gain 调用前检查可发现 transaction/batch signature 不一致或 evidence 被跨 motion、跨 K 或跨 role 混合", "失败归属: paired capture 到 Gain connector"],
  ],
  "AUDIT-GAIN-01": [
    ["raw survival_steps 与 effective_horizon_K 首次同时进入 Gain owner, 可确认原始单位和每行 K 没有错位", "失败归属: paired capture 或 horizon forwarding"],
    ["survival_quality repaired/noisy 与 physics_survival_gain 在 owner 产物边界同时成立, 可确认没有回退到 raw step difference", "失败归属: compute_paired_physics_gain 或 survival unit conversion"],
    ["同一 transaction 的逐步 survival Gain 累计与最终 gain_total 交给正式 consumer 前同时可见, 可发现旧 score、旧单位或跨 batch 旁路", "失败归属: Gain 到 storage/sampler evidence wiring"],
  ],
  "AUDIT-RETURN-01": [
    ["per-step survival_gain_steps 与 per-row K 同时可见, 可验证 PPO reward 输入的单位", "失败归属: Gain reward construction 或 horizon forwarding"],
    ["survival_gain_step_sum 与 returns/advantages 同时成立, 可验证逐步 Gain 没有被累计或折扣路径重复放大", "失败归属: compute_returns_and_advantages"],
    ["PPO batch 构造时检查可发现 transaction/batch signature、return、advantage、valid rows 错位", "失败归属: storage to_ppo_batch conversion"],
  ],
  "AUDIT-HSL-LOAD-01": [
    ["checkpoint path 与 payload identity 刚读取, 可确认实际加载的 Stage 2 artifact", "失败归属: launch checkpoint path 或 torch.load payload"],
    ["actor 与 normalizer state 映射完成后首次形成 Stage 3 初始化状态", "失败归属: load_runner state mapping"],
    ["首次 live policy 使用前检查可发现加载后又被重置或漏传", "失败归属: checkpoint load 到 Stage 3 runner initialization"],
  ],
  "AUDIT-WARMUP-01": [
    ["persisted iteration 与两个 warmup boundary 同时可见, 可验证 phase 输入", "失败归属: resume iteration 或 warmup config"],
    ["phase 与 actor_loss_weight 形成后首次决定允许哪些梯度", "失败归属: frontres_segment_warmup_phase"],
    ["PPO loss 读取前检查可发现 phase 正确但权重未生效", "失败归属: warmup phase 到 PPO config wiring"],
  ],
  "AUDIT-PPO-01": [
    ["old action/logprob/mean/sigma/advantage 同时可见, 可验证 rollout tuple 同源性", "失败归属: storage/PPO batch source"],
    ["loss、ratio、KL 在同一次 forward 后首次具备 PPO 更新语义", "失败归属: compute_frontres_segment_ppo_loss"],
    ["optimizer/rollback 后检查可区分计算正确但提交状态错误", "失败归属: optimizer step、post-KL 或 trust rollback"],
  ],
  "AUDIT-PERSIST-01": [
    ["policy/optimizer/iteration state 刚进入 save owner, 可确认写盘来源", "失败归属: runner training state"],
    ["normalizer/sampler/Gain/warmup 合并后 payload 语义首次完整", "失败归属: save_runner payload construction"],
    ["torch.save 前检查能证明真实写盘对象而非 logger 推断", "失败归属: required-key validation 或 filesystem save boundary"],
  ],
  "AUDIT-DIAG-01": [
    ["canonical live summary 尚未格式化, 可确认日志字段真实来源", "失败归属: live summary aggregation"],
    ["diagnostic scalars 形成后可检查缺失值是否仍为 UNCONFIRMED/NaN", "失败归属: repair_effect_summary_to_scalars"],
    ["terminal/logger 消费前检查可发现 aggregate 是否误报为 single transaction, 或正确 scalar 被旧字段覆盖", "失败归属: diagnostics formatting 或 logger wiring"],
  ],
};

for (const [id] of specs) {
  const steps = probeRationales[id];
  if (!Array.isArray(steps) || steps.length !== 3 || steps.some((step) => step.length !== 2)) {
    throw new Error(`${id} requires three explicit [whyHere, failureOwner] decisions`);
  }
}
const whyHereTexts = specs.flatMap(([id]) => probeRationales[id].map((step) => step[0]));
if (new Set(whyHereTexts).size !== whyHereTexts.length) {
  throw new Error("whyHere must be written per boundary; duplicated template rationale is forbidden");
}

const sourceLinksFor = (ownerPath, auditId) => {
  const absolutePath = path.join(path.resolve(root, "../.."), ownerPath);
  const lines = fs.readFileSync(absolutePath, "utf8").split("\n");
  const auditIndex = lines.findIndex((line) => line.includes(auditId));
  if (auditIndex < 0) throw new Error(`${auditId} is missing from ${ownerPath}`);
  const nearestBefore = (marker) => {
    for (let index = auditIndex; index >= 0; index -= 1) {
      if (lines[index].includes(marker)) return index;
    }
    throw new Error(`${auditId} has no ${marker} block in ${ownerPath}`);
  };
  const indexes = [nearestBefore("# B1:"), nearestBefore("# B2:"), nearestBefore("# B3:")];
  return indexes.map((index) => ({
    sourceLine: index + 1,
    sourceHref: `/open-source?path=${encodeURIComponent(ownerPath)}&line=${index + 1}`,
  }));
};

const card = ([id, title, design, summary, ownerPath, ownerFunction, captures], index) => {
  const sourceLinks = sourceLinksFor(ownerPath, id);
  return ({
  id,
  title: `Probe ${String(index + 1).padStart(2, "0")} | ${title}`,
  summary,
  cardKind: "runtime_probe",
  owns: `验证 ${summary}`,
  mustNot: "不得改变训练分支、张量语义、更新顺序或缺失值状态",
  objects: [`Design: ${design}`, ...captures],
  files: [{ id: `${id}-SRC`, path: ownerPath, role: "被检查的正式 owner 边界", functions: [ownerFunction] }],
  mainRoute: [
    `B1 ${ownerFunction}: upstream state -> owner input`,
    `B2 ${ownerFunction}: owner input -> semantic object`,
    `B3 downstream consumer: semantic object -> formal consumer`,
  ],
  mainRouteTitles: ["进入 Owner", "检查 Owner 产物", "确认下游消费"],
  probeSteps: [
    { location: `${ownerPath}:${sourceLinks[0].sourceLine} :: ${ownerFunction} 入口/调用前`, capture: captures[0], whyHere: probeRationales[id][0][0], failureOwner: probeRationales[id][0][1], ...sourceLinks[0] },
    { location: `${ownerPath}:${sourceLinks[1].sourceLine} :: ${ownerFunction} 产物边界`, capture: captures[1], whyHere: probeRationales[id][1][0], failureOwner: probeRationales[id][1][1], ...sourceLinks[1] },
    { location: `${ownerPath}:${sourceLinks[2].sourceLine} :: 正式 consumer 接收前`, capture: captures[2], whyHere: probeRationales[id][2][0], failureOwner: probeRationales[id][2][1], ...sourceLinks[2] },
  ],
  probe: {
    owner: `${ownerPath}::${ownerFunction}`,
    insertion: "对应 B1/B2/B3 子框, 由同名 AUDIT-* snapshot 输出",
    capture: captures,
    failIf: ["关键对象缺失、非有限或 shape/role 不符", "owner 产物未到达正式 consumer"],
  },
  review: ["按 B1/B2/B3 阅读选择理由、插桩位置、截获对象和失败归属", "E41 已闭合全部 compact audit 字段; 方法质量仍单独待验"],
  tests: ["S2 official-route connectivity", "S4 formal live snapshot"],
  gap: runtimeStatus[id],
  });
};

const data = {
  title: "04 Stage 3 Formal Runtime Audit",
  subtitle: "22 个重要 Owner 边界的正式训练插桩阅读图. 每个 B 子框解释为什么测这里、截获什么以及失败归属.",
  layout: "repository_reading_atlas",
  canvasWidth: 16900,
  defaultZoom: 0.18,
  runtimeOrder: specs.map(([id]) => id),
  supportOrder: [],
  systems: [{ id: "SYS-FORMAL-AUDIT", title: "Stage 3 Formal Runtime Audit", summary: "Concept Figure -> owner -> probe -> formal consumer", color: "#0f766e", modules: specs.map(card) }],
};

fs.writeFileSync(output, `${JSON.stringify(data, null, 2)}\n`);
console.log(`wrote ${output} with ${specs.length} owner cards`);
