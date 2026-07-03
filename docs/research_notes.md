# Research Notes: Hodgkin–Huxley PINN — Forward Problem

Status: forward problem complete. Inverse problem not yet started (see project
README roadmap).

---

## 1. Executive Summary

This project set out to answer a specific question: can a physics-informed
neural network reconstruct the full Hodgkin–Huxley (HH) state — including
gating variables that are never directly measurable in a real experiment —
from sparse, noisy **voltage-only** observations?

The answer is yes, but getting there required discovering and working around
a fundamental failure mode of standard PINNs on stiff, oscillatory systems.
Physics-only training reliably collapses to a smooth, non-spiking equilibrium
rather than the true spiking solution — a failure independently confirmed in
concurrent literature (Kainth et al., 2025) on the same equations. The working
solution is a four-phase hybrid data+physics curriculum, combined with two
non-obvious design choices (a short observation window and spike-biased data
sampling) that took systematic experimentation to identify.

Final result: all four state variables (`V, m, h, n`) reconstructed to
2–5% relative L2 error against a Radau numerical reference, with `m, h, n`
receiving **zero direct data supervision** at any point after `t=0` — their
accuracy is attributable entirely to the physics residual.

---

## 2. Problem Statement & Scientific Motivation

The Hodgkin–Huxley model describes neuronal action potentials via four
coupled ODEs: membrane voltage `V`, and three gating variables `m, h, n`
governing sodium and potassium channel conductance. In real electrophysiology,
patch-clamp recordings measure `V` directly; gating variables are never
observed — they are inferred, if at all, indirectly through model fitting.

This makes HH a genuine test of what a PINN is supposed to offer over
classical numerical integration: not solving the ODE (`scipy.integrate.
solve_ivp` already does that, in milliseconds, given known parameters and
initial conditions), but inferring hidden state from partial, imperfect
observations using the governing physics as a constraint. A forward PINN that
matches a numerical solver given full information is not yet interesting;
one that reconstructs unmeasured variables from partial information is.

HH is also a deliberately hard target for this: it is stiff (widely separated
time constants between fast sodium activation and slower recovery dynamics),
and the spiking solution occupies a narrow, hard-to-reach region of the
network's output space relative to the much larger basin of smooth,
non-spiking trajectories that also approximately satisfy the ODE almost
everywhere except during the spike itself.

---

## 3. Implementation Bugs Found and Fixed

These were found by actually executing the code against real data, not by
inspection alone — several were only caught because a run produced physically
impossible output (a gating variable exceeding 1, or a loss of exactly zero
weight).

| # | Location | Symptom | Root cause | Fix |
|---|---|---|---|---|
| 1 | `model.py`, `init_params` | `AttributeError: module 'jax' has no attribute 'sqrt'` | `jax.sqrt` used instead of `jnp.sqrt` for the Glorot-uniform bound | Changed to `jnp.sqrt` |
| 2 | `hh.py`, `constraints()` | `TypeError` on `jnp.array(0, 0)` | Second `0` interpreted as a `dtype` argument, not a second value | Changed to `jnp.array(0.0)` |
| 3 | `hh.py`, `residual()` **and** `true_solution()` | `h` (and eventually all gates) diverging to unphysical values (`h` reaching `~1e21`) | Sign error: gating ODEs written as `α(1−x) + βx` instead of the correct `α(1−x) − βx`. Present in **both** the training residual and the "ground truth" Radau reference simultaneously — invisible to any check that only compares training output against this same reference | Corrected sign in both locations |
| 4 | `trainer.py`, `__init__` | `UnboundLocalError: cannot access local variable 'optimizer'` | Optimizer assigned to `self.optimizer` inside the `if/else` branch, but a later `use_grad_clip` block referenced a bare `optimizer` that was never defined | Assign to local `optimizer` in both branches; wrap with grad clipping; assign to `self.optimizer` once at the end |
| 5 | `trainer.py`, `train()` | Logged loss values reading roughly `1/log_every` too small, on the wrong step cadence | `if i % log_every == 0`, which fires on the very first step (`i=0`) with only one step's loss accumulated | Changed to `if (i + 1) % log_every == 0` |
| 6 | `hh.py`, `constraints()` | IC always evaluated at `t=0` even when training on a later time window | Hardcoded `t0 = jnp.array(0.0)` instead of using the problem's actual window start | Added `t_start` parameter to `HHProblem`; IC evaluated at `self.t_start` |
| 7 | `hh.py`, `build_supervised()` | Supervised loss dominated by voltage almost entirely regardless of `loss_weights` | `_u()` returns V in physical mV units, but stored targets (`ic_target`, `data_u`) were normalised by `_V_SCALE` — a ~100x scale mismatch | Normalise predictions by the same `_V_SCALE` before comparing to targets |
| 8 | `trainer.py`, `_loss()` | Stage-2 (partial `data_mask`) supervised gradient diluted relative to full-supervision configurations | `mean()` taken over all `(M, k)` entries including zero-weighted (unobserved) ones, so unobserved variables silently shrank the effective gradient | Changed to `sum(weights * sq_err) / sum(weights)` — a true weighted mean |

Bug #3 is worth flagging specifically as a methodological lesson: a bug
present in both the model being trained and the reference used to validate it
is invisible to any test that only checks agreement between the two. It was
only caught by an independent sanity check — gating variables are bounded in
`[0,1]` by construction, and the reference solution violated that bound.

---

## 4. The Core Failure Mode: Flat-Attractor Collapse

**Observation.** Training the PINN with the physics residual active and no
data supervision reliably converges to a smooth trajectory near `V ≈ −60` mV
with no spike, regardless of network size, learning rate, or training
duration.

**Ruling out capacity as the cause.** Pure supervised pre-training (physics
term set to zero, network fit directly to the true spiking trajectory)
reproduces the spike shape without difficulty. This proves the network is
expressive enough to represent the correct solution — the failure is specific
to what happens when the physics-residual loss term is active, not a
limitation of the architecture.

**Mechanism.** The spiking trajectory has a much higher initial physics loss
than the flat trajectory, because it requires large, sharply localised
derivatives (`dV/dt` reaching magnitudes on the order of hundreds of mV/ms
during the spike) that a randomly-initialised or smoothly-converging network
does not produce early in training. Gradient descent on the residual loss
finds the nearby low-loss flat solution and has no mechanism to discover the
much sharper, higher-effort spiking solution once settled there.

**Independent confirmation.** Kainth et al. (2025) report, on the same HH
system, using a standard ("vanilla") PINN baseline: *"The predicted voltage
quickly flattens to a subthreshold resting state with no further spiking...
Static gating variables: Predicted values of n(t), m(t), h(t) stay close to
their initialisation and do not evolve in time."* This matches our
observation closely enough to treat it as the same documented phenomenon,
not an artifact of this specific implementation.

---

## 5. Approaches That Failed, and Why

### 5.1 Sequential time-window training
**Idea:** train on short, sequential time windows (e.g. 0.1–0.5 ms), carrying
the learned state forward as the initial condition for the next window,
avoiding the need to solve the whole spike at once.

**Failure mode:** small errors in the gating variables — particularly `m`,
which has the fastest kinetics near threshold — compound across windows.
Concretely: propagating the network's own (slightly inaccurate) state to
`t=1.5` ms gave `m=0.11`, producing `dV/dt = −24.7` mV/ms (no spike possible).
The true state at that time has `m=0.19`, producing `dV/dt = +34.3` mV/ms (the
spike fires). A gating error of `0.08` is the entire difference between firing
and not firing. Sequential windowing has no mechanism to correct this kind of
error once introduced, since each window only sees a fixed initial condition,
not the true trajectory.

### 5.2 Radau warm-starting
**Idea:** pre-train the network directly on the Radau numerical solution,
then fine-tune with the PINN loss.

**Rejected on principled grounds before being tried in earnest:** if the
network is fit to a solution obtained by a classical solver that already
requires the parameters and initial conditions to be known, the PINN
contributes nothing that the classical solver didn't already provide — it
defeats the purpose of using a PINN at all. Tried anyway for comparison, and
confirmed: while the spike survives, the result is essentially the numerical
solver's answer wearing a neural network, not independent physics-based
inference.

### 5.3 Tail extrapolation
**Setup:** train on `t ∈ [0, 30]` ms, evaluate on the held-out `t ∈ [30, 50]`
ms, which contains a second full action potential (constant `I_ext` produces
periodic firing; the full 50 ms domain in fact contains four spikes, at
approximately `t = 2, 17, 32, 46` ms).

**Result:** the model predicts an essentially flat line in the held-out
region (`V` std `= 0.90`, true std `= 27.5`), completely missing the second
spike.

### 5.4 Interior gap-filling
**Setup:** withhold data from a window in the *middle* of the domain (e.g.
`[12, 22]` ms, containing the spike at `t≈17`), with data present on both
sides, to test whether bracketing data helps physics propagate a solution
into the gap.

**Result:** identical failure to tail extrapolation. Even a 2 ms gap placed
directly on the spike peak, with data immediately outside it on both sides,
produces a flat prediction (`V_max` predicted `= −24.8` mV vs. true `= 30.8`
mV). Gap width was not the limiting factor — even the narrowest gap tested
failed identically to the widest.

**Interpretation:** both failures are the same underlying phenomenon —
the physics residual weight required to avoid destroying a data-anchored
spike during training (see Section 6) is also too weak to independently
sustain or reconstruct a spike anywhere data does not directly anchor one,
regardless of temporal distance from that data. This is consistent with the
"propagation hypothesis" of Daw et al. (2023): correct solutions must
propagate outward from data/boundary points during training, and when that
propagation is hindered (as it is here, by design, to protect the spike from
being erased), the network is stuck at the trivial solution everywhere else.

**Metric correction, discovered along the way.** The originally planned
validation metric, `ratio = rel_out / rel_in`, is unreliable once `rel_in`
becomes small — it diverges from denominator collapse alone, independent of
whether `rel_out` (the number that actually matters) has changed. Retroactive
computation showed the "generalisation gap" `rel_out − rel_in` stayed roughly
constant (~0.22 at `N=30`, ~0.45–0.46 at `N=120/150`) while the ratio swung
from `1.84x` (apparently passing) to `24x` (apparently failing) across the
same range purely from this artifact. The gap is the more meaningful
quantity; no calibrated pass/fail threshold has been established for it.

**Scope conclusion.** Extrapolation and gap-filling were retired as
validation objectives for this project. HH under a fixed stimulus protocol is
not a forecasting task — none of the earlier ODE/PDE template projects tested
extrapolation either — and no architecture used here was built to support it.
A fair test of "does physics do real work" is dense reconstruction within an
observed window, which Sections 6–9 address directly.

---

## 6. The Working Strategy: Hybrid Data + Physics

The successful recipe is a four-phase curriculum, each phase initialised from
the previous phase's trained parameters, with `data_mask = (True, False,
False, False)` (voltage-only supervision) held fixed throughout:

| Phase | `residual_weight` | `loss_weights` (V, m, h, n) | Purpose |
|---|---|---|---|
| Pre-training | `0.0` | — | Fit `V` to data with physics off. Gates receive no signal beyond the `t=0` initial condition |
| Phase A | `1.0` | `(0.0, w, w, w)` | Physics switched on for the gates only. `V`'s own physics term stays zeroed, so its data fit is undisturbed, while `m, h, n` are pulled into ODE-consistency with the now-accurate `V` |
| Phase B | `1.0` | `(0.02, w, w, w)` | Joint fine-tune — a small physics weight is reintroduced for `V` on top of its continuous data supervision, for light regularisation now that the gates are consistent |

`w` (the shared gate physics weight) and the phase step counts were tuned
empirically — see Section 7.

This is best understood as curriculum training along two independent axes,
not one:

- **Physics strength**, ramped over training within a phase or across phases
  (separate from the axis below).
- **Which variable** physics is allowed to act on, controlled by
  `loss_weights` rather than a single shared scalar. `V`'s data supervision
  is never switched off — what changes is only when and how strongly physics
  is layered on top of it, and for which variables.

The reason this variable-axis curriculum succeeds where the time-axis
curriculum (Section 5.1) failed is that `V` remains globally, continuously
supervised by data across the *entire* domain at once. There is no
handoff of an imperfect boundary state between stages — only the target of
physics enforcement changes, never the domain it acts over.

Two further design choices, found through the experiments in Section 7,
turned out to matter as much as the phase structure itself:

- **Short observation window.** Training on `t ∈ [0, 30]` ms (containing two
  oscillatory cycles) rather than the full `[0, 50]` ms (four cycles),
  following the short-window regime reported in Wei et al. (2026).
- **Spike-biased data sampling.** Concentrating a portion of the sparse data
  points within the narrow (~2–3 ms) spike windows, rather than sampling
  uniformly at random across the domain.

---

## 7. Systematic Experiments and Ablations

### 7.1 Data quantity, noise, and sampling strategy (Stage 1, full supervision, `t_end = 50` ms)

| Config | N | Sampling | σ_V | `residual_weight` schedule | V | m | h | n |
|---|---|---|---|---|---|---|---|---|
| Baseline | 30 | uniform | 2.0 | 0.01 (flat, 20k steps) | 0.317 | 0.581 | 0.184 | 0.111 |
| +Steps | 30 | uniform | 2.0 | 0.01 (flat, 30k steps) | 0.307 | 0.555 | 0.177 | 0.109 |
| +Fine ramp | 30 | uniform | 2.0 | 8-rung, 0.01→0.25 (32k) | 0.282 | 0.486 | 0.150 | 0.076 |
| N=50 | 50 | uniform | 2.0 | 4-rung, 0.01→0.18 (16k) | 0.168 | 0.224 | 0.124 | 0.058 |
| N=80 | 80 | uniform | 2.0 | 4-rung (16k) | 0.140 | 0.196 | 0.068 | 0.045 |
| N=120 | 120 | uniform | 2.0 | 4-rung (16k) | 0.092 | 0.160 | 0.086 | 0.071 |
| Low-noise | 30 | uniform | 0.5 | 4-rung (16k) | 0.289 | 0.504 | 0.241 | 0.088 |
| Biased-30 | 30 | spike-biased | 2.0 | 4-rung (16k) | 0.144 | 0.263 | 0.234 | 0.090 |
| Biased-50 | 47† | spike-biased | 2.0 | 8-rung (32k) | 0.278 | 0.498 | 0.126 | 0.066 |
| Biased-30F | 30 | spike-biased | 2.0 | 8-rung (32k) | 0.186 | 0.307 | 0.208 | 0.102 |
| N=120-biased | 104† | spike-biased | 2.0 | 4-rung (16k) | 0.226 | 0.392 | 0.093 | 0.078 |

† after deduplication against the dense reference grid.

**Findings:**
- More training steps alone had almost no effect, despite the loss visibly
  still decreasing — a sign the loss was optimising something decoupled from
  full-domain trajectory accuracy.
- A finer `residual_weight` ramp gave a modest (~15–30% relative) improvement
  across all variables — real, but a second-order effect.
- Data quantity was the dominant lever: `N=30→80` roughly halved every error.
  Noise level was comparatively unimportant — a 4x noise reduction at fixed
  `N=30` did not reliably help and in one case made results worse.
- Spike-biased sampling at `N=30` nearly matched `N=80` uniform on `V`
  specifically (`0.144` vs. `0.140`), confirming the effect is about
  *coverage of fast dynamics*, not raw point count — but did not help `m` or
  `h` nearly as much, since those gates need broader temporal coverage than
  just the spike window.
- The advantage of biasing reverses at higher `N`: `N=120`-biased performed
  *worse* than `N=120`-uniform across nearly every variable. Once uniform
  sampling already puts several points inside the spike window naturally,
  further concentrating density there starves the rest of the domain.

### 7.2 Stage 2 (voltage-only) gate/physics decoupling (`t_end = 50` ms, `N=80`)

Motivation: `h` remained near 75% relative error under Stage 2 regardless of
`N`, ruling out data quantity as the cause. Diagnosis: the single shared
`residual_weight`, kept low to protect `V`'s spike, was also throttling the
*only* signal the gates receive. Fix: `loss_weights` to decouple gate physics
strength from `V` physics strength (Section 6, Phase A/B).

| Config | `gate_w` | Phase A / B steps | V | m | h | n |
|---|---|---|---|---|---|---|
| Baseline (uniform, single-phase) | — | 16k (flat 0.01) | 0.147 | 0.254 | 0.749 | 0.160 |
| Decoupled | 0.3 | 8k / 8k | 0.158 | 0.194 | 0.471 | 0.111 |
| Decoupled | 0.4 | 8k / 8k | 0.158 | 0.179 | 0.362 | 0.107 |
| Decoupled | 0.3 | 14k / 8k | 0.129 | 0.149 | 0.362 | 0.100 |
| Decoupled | 0.4 | 14k / 8k | **diverged to NaN** | | | |
| Decoupled | 0.3 | 20k / 8k | 0.128 | 0.159 | 0.322 | 0.097 |
| Decoupled (best at N=80) | 0.3 | 14k / 14k | 0.131 | 0.150 | 0.307 | 0.095 |

**Findings:**
- Decoupling gate physics from `V` physics improved every variable
  simultaneously relative to baseline — not a trade-off.
- `gate_w = 0.4` combined with a longer Phase A caused outright training
  divergence (NaN), not just diminishing returns — a real instability
  boundary close to the working `0.3` configuration, not a gentle slope.
- Extending training duration (`N=150` instead of `N=80`, same recipe)
  improved `V, m, n` substantially (`0.131→0.056`, `0.150→0.090`,
  `0.095→0.083`) but made `h` worse (`0.307→0.392`) — a mixed, not uniformly
  positive, result.

### 7.3 Shortened domain + spike-biased sampling (`t_end = 30` ms, `N≈70`)

Initial attempt carried over the `t_end=50` decoupled recipe unmodified with
uniform sampling: Stage 2 result was **worse** than the `t_end=50, N=80`
baseline (V=0.191, m=0.336, h=0.091, n=0.066) despite denser sampling in
points/ms, prompting a fresh diagnosis rather than further hyperparameter
tuning.

**Diagnosis, in order:**
1. Right after pure data-only pre-training (physics fully off), `V` error was
   already `18.8%` — the bottleneck was upstream of the physics phases
   entirely.
2. Extending pre-training to 20k steps did not help (`21.5%`), ruling out
   undertraining.
3. Splitting the error by region showed it concentrated almost entirely at
   the spikes (`63.5%` local error near the two spike windows vs. `2.8%`
   elsewhere) — a resolution problem, not a general fit-quality problem.
4. Reducing noise 10x (`σ_V: 2.0→0.2`) made `V` error *worse* (`24.2%`),
   ruling out noise as the driver.

**Fix:** explicit spike-biased sampling (15 points each in the `[1,4]` ms and
`[15,19]` ms windows, 40 uniform elsewhere, `N≈68–70` after deduplication),
with `gate_w` relaxed slightly to `0.2`.

| Stage | V | m | h | n |
|---|---|---|---|---|
| Pre-training (physics off) | 0.077 | 1.489 | 1.318 | 0.600 |
| Phase A | 0.051 | 0.059 | 0.038 | 0.019 |
| Phase B (final) | **0.027** | **0.032** | **0.051** | **0.020** |

Reproducibility check across three random seeds (data draw + initialisation):

| Seed | V | m | h | n |
|---|---|---|---|---|
| 42 | 0.029 | 0.036 | 0.042 | 0.019 |
| 7 | 0.017 | 0.026 | 0.029 | 0.011 |
| 99 | 0.020 | 0.033 | 0.060 | 0.022 |

All four variables land under 5% relative L2 error on two of three seeds —
the best result of the project, by a wide margin over any configuration
found in Sections 7.1–7.2.

**Sanity check on the "voltage-only" claim.** The data-generation function
computes noisy values for `m, h, n` at every sampled time point even though
`data_mask` excludes them from the loss. To confirm these values are truly
inert rather than leaking information some other way, the gate columns were
replaced with physically nonsensical values (uniform random in `[-1000,
1000]`) and training re-run: the result was bit-for-bit identical to the
normal-noise run (max absolute difference `= 0.0`). The final gate
reconstruction is attributable entirely to the physics residual.

---

## 8. Literature Comparison

| Source | Architecture | System | Relation to this work |
|---|---|---|---|
| Ferrante et al. (2022), arXiv:2209.11998 | Standard collocation PINN (same paradigm as this project) | HH, isolated single spikes | Closest methodological match. Validates hidden-gate reconstruction from V-only data as a legitimate, previously-published result. Never attempts extrapolation; fits single isolated spikes rather than a continuous multi-spike domain; smaller network (3 layers, 40–60 units) |
| Kainth et al. (2025), arXiv:2511.11734 | Neural ODE + explicit integrator + scale-aware residual normalisation | HH | Independently reports the identical flat-attractor collapse for a "vanilla PINN" baseline on the same system (Section 4). Their extrapolation success comes from a structurally different mechanism (learned vector field + explicit numerical integration), not transferable to a collocation PINN without changing architecture |
| Krishnapriyan et al. (2021), NeurIPS | — (theoretical/empirical analysis) | General stiff PDE/ODE PINN failure modes | Establishes that soft-constraint PINN failures are optimisation/loss-landscape pathologies, not architecture-capacity limitations — consistent with Section 4's pre-training capacity check |
| Daw et al. (2023), ICML | — (theoretical/empirical analysis) | General PINN interior-point failures | The "propagation hypothesis" explains the interior gap-filling failure (Section 5.4): correct solutions must propagate from data/boundary points during training, and this propagation was deliberately weakened here to protect the spike |
| Wei, Wang & Zhu (2026), arXiv:2603.08742 | Standard PINN + random weight factorisation, adaptive loss balancing, residual scaling, separate LR schedules for physical parameters | Morris-Lecar (fast-slow spiking model) | Directly motivated the short-window, voltage-only-observation framing used in Section 7.3. Reports `V` rel-L2 as low as `0.09%` at 1% noise — roughly 20–30x tighter than this project's best result (`2.7%`). Gap plausibly attributable to their additional architectural techniques (not yet implemented here) and/or HH's greater stiffness relative to Morris-Lecar; not yet isolated which factor dominates |

---

## 9. Final Results

Configuration: `t_end = 30` ms (two oscillatory cycles), `N ≈ 70` spike-biased
noisy voltage samples (`σ_V = 2` mV), `data_mask = (V observed; m, h, n fully
latent)`, network `(128, 128, 64)` MLP, `tanh` activation.

| Variable | Pre-training | Phase A | Phase B (final) |
|---|---|---|---|
| V | 7.7% | 5.1% | **2.7%** |
| m | 148.9% | 5.9% | **3.2%** |
| h | 131.8% | 3.8% | **5.1%** |
| n | 60.0% | 1.9% | **2.0%** |

The pre-training column is the clearest evidence that the final accuracy is a
physics result, not a data-fitting result: `m, h, n` are essentially
untrained at that point (>60–150% relative error, since no data ever touches
them beyond the single `t=0` initial condition), and Phase A — physics acting
on the gates alone — brings all three under 6% using no data at all beyond
what was already used to fix `V`.

Result confirmed reproducible across three random seeds (Section 7.3);
confirmed not attributable to hidden gate-data leakage (Section 7.3, garbage
substitution test).

---

## 10. Open Questions and Limitations

- **Spike-biased sampling requires prior knowledge of spike timing.** The
  sampling strategy that produced the best result concentrates points using
  the true Radau solution to locate the spikes in advance. On genuinely
  unknown real data, spike timing would not be known ahead of the experiment
  in the same way — this result validates the underlying resolution
  diagnosis but is not yet a deployable recipe for truly unknown data.
- **`h` is the least reliably reconstructed variable** across most
  configurations tested (Section 7.2's N=150 result, Section 7.3's seed=99
  run), though the best configuration overall gets it under 6%.
- **Unexplained interaction between biased sampling and fine `residual_weight`
  ramps.** `Biased-50` and `Biased-30F` (fine ramp on biased data) performed
  worse on `V` than `Biased-30` (coarse ramp, same biased data) — the
  opposite direction from the fine ramp's effect on uniform data. Not
  investigated further.
- **The `gate_w = 0.4` instability boundary is not characterised.** It is
  known to cause divergence when combined with a longer Phase A; the exact
  boundary (as a function of `gate_w`, phase length, and learning rate) has
  not been mapped.
- **The generalisation-gap metric (`rel_out − rel_in`) has no calibrated
  pass/fail threshold.** It corrects the ratio metric's denominator-collapse
  artifact but has only been computed across a narrow range of configurations
  (0.22–0.46), insufficient to set a meaningful cutoff.
- **The ~20–30x gap to Wei et al.'s reported accuracy is not fully
  explained.** Plausible contributors (their architectural techniques;
  HH vs. Morris-Lecar stiffness) have not been isolated individually.

---

## 11. Appendix: Full Experiment Log

*(Reserved for the complete chronological run log, including exact
hyperparameters, random seeds, and wall-clock times for every experiment
referenced in Sections 7–9. To be populated from raw run records.)*
