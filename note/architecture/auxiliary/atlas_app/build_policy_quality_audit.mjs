import fs from "node:fs";
import path from "node:path";

const root = path.resolve(process.cwd());
const repoRoot = path.resolve(root, "../..");
const output = path.join(root, "runtime/05_policy_quality_audit.data.json");

const specs = [
  ["QUALITY-ID-01", "Matched Comparison Identity", "FRS-DP-02/03/06 | SR-01/M-06/Q-PAIR", "正式 policy_quality_eval 是否安装真实 owner 并让同一 manifest item 的 zero/HSL/policy 从同一动态起点开始", "source/rsl_rl/rsl_rl/runners/frontres_policy_quality_eval.py", "run_frontres_policy_quality_eval()", ["validated request + real-owner installation", "matched manifest execution", "comparison/state/checkpoint/owner identity"], "manifest/reset/evaluator", "integrated-offline: Q-E6; real simulator equality pending Q1-F"],
  ["QUALITY-DATA-01", "Gradient-Bearing Distribution", "FRS-DP-02/03 | SR-01/M-06", "真正进入 PPO 的 rows 是否对应预期 replay/K/difficulty 分布", "source/rsl_rl/rsl_rl/runners/frontres_segment_live_sampler.py", "_attach_frontres_segment_trial_plan()", ["sample source and unique segment", "policy/search role + K/difficulty", "repeat/staleness/valid policy rows"], "sampler/replay/curriculum", "pending: Q4 distribution evidence"],
  ["QUALITY-ACTION-01", "Counterfactual 6D Actions", "FRS-DP-04 | M-04", "zero/HSL/policy 的 full-6D action 是否保持来源、边界与实际写入身份", "source/rsl_rl/rsl_rl/frontres/task_space_correction.py", "apply_frontres_task_corrections()", ["requested bounded Delta SE(3)", "contact-consistent applied correction", "command/GMT action identity"], "actor/checkpoint/projection/application", "offline source/shape observed: Q-E3; real application pending Q1-F"],
  ["QUALITY-GAIN-01", "Canonical Quality Label", "FRS-DP-06/07 | Q-PAIR/Q-01", "相同 execution evidence 是否由唯一 FRS-GAIN-v002 owner 产生可学习标签", "source/rsl_rl/rsl_rl/frontres/frontres_gain.py", "compute_segment_gain()", ["matched Style/Physics/Repair inputs", "FRS-GAIN-v002 composition", "total/per-step/K/finite result"], "pairing/Gain units/repair cost", "pending: Q2/Q5 matched Gain evidence"],
  ["QUALITY-CREDIT-01", "Return And Advantage Credit", "FRS-DP-07/09 | Q-01/M-05", "canonical Gain 的符号和质量差异是否正确进入 returns/advantages", "source/rsl_rl/rsl_rl/frontres/frontres_segment_storage.py", "compute_returns_and_advantages()", ["policy-row Gain steps + K", "returns/advantages + valid mask", "sign and bucket dominance"], "storage/return/advantage scaling", "pending: Q4/Q6 credit evidence"],
  ["QUALITY-UPDATE-01", "Local PPO Direction", "FRS-DP-09 | M-05", "正负 advantage 是否把 action log-prob 沿正确方向更新且通过 trust", "source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py", "run_frontres_segment_single_update()", ["old stats + advantage sign", "backward/step/post-KL", "accepted delta + held-out direction"], "PPO/detach/optimizer/trust", "pending: Q6 local-update evidence"],
  ["QUALITY-EXEC-01", "Frozen-GMT Execution", "FRS-DP-04/05/06/07 | M-04/M-10/Q-PAIR/Q-01", "action 改变后 frozen GMT 的物理执行是否真的改善", "source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py", "_capture_motion_quality_frame()", ["success/fall/survival + action identity", "ZMP/contact/MPJPE/dynamics", "short-K versus long-sequence boundary"], "application/GMT/metric authority", "callback connectivity observed: Q-E3; physical evidence pending Q1-F/Q2"],
  ["QUALITY-TRAJECTORY-01", "Checkpoint Learning Curve", "FRS-DP-02/07/09 | SR-01/Q-01/M-05", "固定 manifest 上 checkpoint 序列何时改善、遗忘或收敛到 no-op", "source/rsl_rl/rsl_rl/runners/frontres_policy_quality_eval.py", "install_frontres_policy_quality_manifest_executor()", ["ordered manifest/checkpoint axis", "matched route result per item", "atomic trajectory/result artifact"], "trajectory/replay/curriculum/generalization", "executor schema observed: Q-E5; checkpoint trajectory pending Q3"],
];

const sourceLines = (file, id) => {
  const lines = fs.readFileSync(path.join(repoRoot, file), "utf8").split(/\r?\n/);
  const marker = lines.findIndex((line, markerIndex) => {
    if (!line.includes(`# ${id}:`)) return false;
    return ["B1", "B2", "B3"].every((block) =>
      lines.some(
        (candidate, candidateIndex) =>
          candidateIndex >= markerIndex &&
          candidateIndex <= markerIndex + 12 &&
          candidate.includes(`# ${block}:`),
      ),
    );
  });
  if (marker < 0) throw new Error(`missing ${id} source marker in ${file}`);
  return ["B1", "B2", "B3"].map((block) => {
    const index = lines.findIndex((line, i) => i >= marker && i <= marker + 12 && line.includes(`# ${block}:`));
    if (index < 0) throw new Error(`missing ${id} ${block} source marker in ${file}`);
    return index + 1;
  });
};

const href = (file, line) => `/open-source?path=${encodeURIComponent(file)}&line=${line}`;
const modules = specs.map(([id, title, parent, question, file, fn, captures, failureOwner, status], index) => {
  const lines = sourceLines(file, id);
  const files = lines.map((line, step) => ({
    id: `${id}-B${step + 1}`,
    path: file,
    role: `B${step + 1}: ${captures[step]}`,
    sourceLine: line,
    sourceHref: href(file, line),
  }));
  const mainRoute = captures.map((capture, step) => `B${step + 1} ${capture}`);
  const probeSteps = captures.map((capture, step) => ({
    location: `${file}:${lines[step]} :: ${fn} B${step + 1}`,
    capture,
    whyHere: [
      "比较控制变量在任何 route 分叉前首次完整可见",
      "owner 已形成当前层产物且下游尚未聚合或覆盖",
      "正式 consumer 接收前最后一次可判定失败归属",
    ][step],
    failureOwner: `失败归属: ${failureOwner}`,
    sourceLine: lines[step],
    sourceHref: href(file, lines[step]),
  }));
  return {
    id,
    title: `Quality ${String(index + 1).padStart(2, "0")} | ${title}`,
    summary: question,
    cardKind: "quality_probe",
    parentDesignPoint: parent,
    question,
    failureOwner,
    owns: question,
    mustNot: "不得修改方法语义、旧 evaluator、训练状态或用 unmatched aggregate 宣称质量",
    objects: [`Parent: ${parent}`, ...captures],
    files,
    mainRoute,
    probeSteps,
    probe: {
      owner: `${file}::${fn}`,
      insertion: "B1/B2/B3 source-linked quality boundaries",
      capture: captures,
      failIf: ["comparison identity 不一致", "owner 产物缺失、非有限或未到达指定 consumer"],
    },
    review: ["先读问题与可证伪关系", "再按 B1/B2/B3 打开源码并确认 failure owner"],
    tests: ["offline deterministic quality contract", "matched live evidence only when gate permits"],
    gap: status,
  };
});

const data = {
  title: "05 FrontRES Policy Quality Audit",
  subtitle: "从训练分布到物理执行的八个质量 owner. Q1 只关闭 matched identity scaffolding, 不宣称 policy useful.",
  layout: "repository_reading_atlas",
  canvasWidth: 7600,
  defaultZoom: 0.28,
  runtimeOrder: specs.map(([id]) => id),
  supportOrder: [],
  systems: [{
    id: "SYS-POLICY-QUALITY",
    title: "Policy Quality Causal Chain",
    summary: "distribution -> Gain -> credit -> update -> action -> frozen-GMT execution -> trajectory",
    color: "#0f766e",
    modules,
  }],
};

fs.writeFileSync(output, `${JSON.stringify(data, null, 2)}\n`);
console.log(`wrote ${output} with ${modules.length} quality cards`);
