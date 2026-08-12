---
contract_id: FRS-METHOD-v023
status: superseded
effective_date: 2026-08-11
updated_date: 2026-08-11
supersedes: FRS-METHOD-v022
scope: Recovery-Aware B8/M4 training with bounded curriculum-driven outer sealed-Scenario replay
superseded_by: FRS-METHOD-v024
---

# Bounded Outer Scenario Replay Curriculum

## Design delta

FRS-METHOD-v022 supplied stable Scenario identity, fresh current-policy M4
recollection, DR-compatible selection and the detached `E_V`/`E_A` scores. It
did not bound the active working set and used a fixed 40/50/10 source mixture.
On AMASS this admitted new Scenarios faster than the same Critic could revisit
them. FRS-METHOD-v023 replaces that source mixture with a bounded replay
curriculum. Gain, utility, PPO mass and Scenario identity are unchanged.

## Active pool and archive

Each K owns one archive and one bounded active set. The archive may grow; only
the active set may be sampled by replay. Its production capacity ladder is:

```text
64 -> 128 -> 256 Scenarios per K
```

Every active Scenario must receive at least four committed, fresh-current-policy
M4 visits before the capacity may expand. Capacity expansion is allowed only in
`joint`, after the current K has reached full DR, so Scenario breadth and DR do
not expand together. A K transition starts the new K at capacity 64 while
preserving the shared Actor, Critic and optimizer.

Active membership retains Easy/Medium/Hard/Broken quotas 20/30/40/10 using
deterministic largest-remainder integer counts at each capacity. Admission uses
an underfilled class. At full capacity, a new committed candidate replaces one
same-class, at-least-four-visit, non-anchor record with the lowest joint
`E_V/E_A` learning value; the displaced record remains in the archive. Per
class, the two highest `E_V` and two highest `E_A` records are protected anchors.

## Exact B8 selection schedule

One formal transaction selects eight distinct Scenarios under one frozen
`pi_old`, with M=4 fresh Repair attempts per Scenario:

```text
low_dr_joint_init / coupled_ramp:
  1 admission opportunity + 6 E_V replay + 1 stale review

joint:
  1 admission opportunity + 4 E_A replay + 2 E_V replay + 1 stale review
```

An admission opportunity creates a new Scenario only when active capacity or a
visit-qualified replacement exists. Otherwise that slot replays a compatible
active Scenario, preventing breadth from outrunning repeat visits. Other
unavailable replay slots fail over to compatible active records of the same
score kind, then to controlled new admission; they never change the already
drawn DR class. `E_V` therefore remains represented after warmup instead of
disappearing in joint. Each selected record is rematerialized from its stable
key and recollected with fresh M4 actions.

## Transaction and persistence

Selection, admission, replacement, capacity advance, score EMA, staleness and
RNG are preview state. They commit only after the matching exact-one Adam
receipt. Failed, partial, duplicate or mixed transactions change none of them.

Checkpoint-v17 stores replay schema v3, archive records, active membership and
capacity per K, both score maps, visits/staleness, RNG and the last commit.
Checkpoint-v16 and earlier are incompatible with active training.

## Preserved boundaries

- Actor remains the deployable 158D full-6D direct Delta SE(3) policy.
- Critic remains the 449D action-pre state value `V(s)`.
- Raw FRS-GAIN-v008, per-attempt symlog and exact-M mean target remain.
- K8 -> K16 -> K32, M4, one grouped Adam step and frozen GMT remain.
- Replay never reuses policy rows or changes optimizer mass.

## Falsifiers

- A transaction exposes fewer or more than eight distinct Scenario states.
- Active capacity grows before every active record has four committed visits.
- Active capacity grows while DR is still ramping.
- Joint selection contains no `E_V` calibration replay.
- Active class quotas drift, or replacement deletes an archive record.
- Failed commit changes membership, capacity, score, staleness or RNG.
