---
contract_id: FRS-GAIN-v009
status: active-pre-training
effective_date: 2026-08-17
updated_date: 2026-08-20
supersedes: FRS-GAIN-v008-for-relational-route
scope: Hierarchical Physics/Intent evidence to a fail-closed partial order; no scalar Gain
---
# Hierarchical Relational Gain

FRS-GAIN-v009 publishes evidence and a relation between Repair outcomes. It
does not publish `G_total`, symlog utility, a Critic target, or a hidden scalar
tie-break. Clean remains the evaluator anchor and Noisy remains a diagnostic
baseline; preference edges compare Repairs from the same sealed Scenario.

## Classification

Each Repair is classified in exactly one Physics level:

\[
L_0=\text{survival failed},\quad
L_1=\text{severe contact/support violation},\quad
L_2=\text{admissible but unsettled},\quad
L_3=\text{admissible and stable for the final hold window}.
\]

Survival failure, expected-support no-load, unplanned support switch, and
illegal contact are hard Physics evidence. They cannot be compensated by
Intent, drift, ZMP, or cost.

## Relation

For two outcomes \(A,B\) from one Scenario:

\[
A\succ B \iff
\begin{cases}
L(A)>L(B), &\text{or}\\
L(A)=L(B)=L_1\text{ and }A\text{ Pareto-dominates }B, &\text{or}\\
L(A)=L(B)=L_2\text{ and }A\text{ Pareto-dominates }B, &\text{or}\\
L(A)=L(B)=L_3\text{ and }A\text{ Pareto-dominates }B.
\end{cases}
\]

The L0 internal order is intentionally unconfirmed. In L1 the vector is the
three severe violations; in L2 it is capture margin/trend, applicable phase
ZMP margin, linear/angular momentum error, support drift, and stable-hold
evidence; in L3 it is the confirmed Intent vector followed by repair cost only
when Intent is tied. Quantized exact Pareto comparison is used so the relation
is transitive. Tradeoffs are `INCOMPARABLE`, not forced into a score.

Missing, invalid, inconsistent, or mismatched ZMP applicability is `INVALID`
and closes the transaction. A valid non-applicable ZMP is represented by
`zmp_applicable=false, zmp_margin=None` and is not zero-filled.

## Training boundary

The relation adapter emits directed edges `(winner, loser)` only for
`BETTER/WORSE`. `SAME` and `INCOMPARABLE` emit no edge. Edge incidence
(`wins - losses`) is a diagnostic only. This Gain owner does not define an
Actor Loss, policy ratio, gradient scale, learning rate, or optimizer step;
those semantics belong exclusively to FRS-PPO-v014 and FRS-TRAIN-v025.

The scalar state-value Critic and scalar target are inert on this route. If a
transaction has no valid edge, it performs no optimizer or Replay commit and
reports `NO_COMPARABLE_PAIRS`.
