# FRS Gain Relational Interface Proposal v001

Status: implementation candidate; Contract migration remains gated before training.

## Design change

The current active route publishes scalar `G_total` and trains PPO on its
symlog utility. The proposed route publishes a relational result:

```text
BETTER | WORSE | SAME | INCOMPARABLE
```

The relation is produced by the confirmed hierarchical Physics ordering and
must preserve non-compensable violations and explicit incomparable cases.

## Preserved behavior

- `FRS-GAIN-v008`, `FRS-PPO-v012`, `FRS-TRAIN-v024`, and `FRS-EVAL-v006`
  remain active and unchanged.
- The current training route, checkpoint identity, Replay, and scalar Critic
  target remain authoritative until a Contract migration is confirmed.
- The relational route is selected only by an explicit Actor-only owner flag;
  it does not silently change the legacy scalar route.

## Resolved training interface

The scalar state-value Critic is retired on the relational route. The new
consumer receives directed same-Scenario preference edges and uses the clipped
Actor-only loss. The compatibility Critic is frozen and excluded from the
optimizer; no scalar target is constructed.

## Remaining activation gates

1. Connect the formal transaction executor to the relational edge batch with
   no scalar Replay or utility commit.
2. Keep masked pairwise `c_i` as the accepted Actor credit, with `SAME` and
   `INCOMPARABLE` contributing no update mass rather than a tie or penalty?
3. What is the exact current-transaction aggregation and checkpoint schema?
4. Which old scalar artifacts are historical, and what is the migration/reject
   rule?

## Candidate pre-training adapter

The independent candidate adapter constructs pairwise relations inside one
current transaction. For attempt `i`, let `W_i` be the number of confirmed
preference edges it wins, `L_i` the number it loses to, and `N_i=W_i+L_i`. Its
edge credit is:

```text
c_i = W_i - L_i,           if N_i > 0
c_i = undefined,           if N_i = 0
```

`SAME` and `INCOMPARABLE` pairs do not create a direction. An invalid evidence
pair invalidates the candidate batch; it is never converted to zero credit.
The candidate Actor-credit mask is `N_i > 0`, so a batch with no comparable
pair has status `NO_COMPARABLE_PAIRS` rather than fabricated credit. The PPO
owner divides the sum of edge losses by the number of valid preference edges;
the two ends of each edge therefore receive equal and opposite mass. This
adapter deliberately defines no Critic target. The relational route uses no
Critic; a fully ordered M4's zero-sum edge credit is already the within-Scenario
baseline. The evidence adapter now maps each sealed Scenario's exact-M Outcome
carriers to global row edges; missing Outcome evidence remains a transaction
error.

## Pre-training execution plan

This engineering unit stops before any training run. The user retains control
of code synchronization, checkpoint creation, and training execution.

1. Freeze the relational ordering card and Design Inspector semantics. Done.
2. Implement the independent masked Actor-credit candidate. Done.
3. Run exact-M4 pseudo-data Module Alignment, permutation, invalid-evidence and
   controlled-counterexample tests. Done.
4. Activate the coordinated Actor-only relational Contract migration, with the
   current scalar Critic explicitly retired from the new route. In progress.
5. Implement the official composition root, checkpoint schema and strict
   fresh-start/legacy-reject rules. Interface and checkpoint identity are done;
   formal executor and Replay owner remain.
6. Exercise one bounded offline pseudo-transaction and complete construction,
   final code review and formal pre-training audit.
7. Stop and hand back the synchronized code, tests, manifests, and exact
   user-run command. No training, checkpoint generation, live transaction, or
   policy-quality claim is performed in this unit.

## Stop condition

Until these decisions have a confirmed Design Inspector receipt and an active
Contract migration, do not modify the training Gain owner or launch live
training. The current candidate may continue only through offline semantic
tests and bounded telemetry diagnostics. Even after the pre-training audit is
complete, this unit stops before training and returns control to the user.
