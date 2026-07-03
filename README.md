# Hodgkin–Huxley PINN

A Physics-Informed Neural Network (PINN) that reconstructs Hodgkin–Huxley (HH)
neuron dynamics from sparse, noisy **voltage-only** observations — inferring
the unobservable gating variables (`m`, `h`, `n`) purely from the biophysical
model, without ever seeing a single gate measurement.

This mirrors a real electrophysiology constraint: patch-clamp recordings
measure membrane voltage directly, but ion-channel gating states are never
observed. The scientific question this project answers is whether a
physics-constrained network can recover that hidden state from voltage alone.

## Status

**Forward problem: complete.** The network reconstructs the full four-variable
state (`V, m, h, n`) from voltage-only data, with all four variables reaching
under ~5% relative L2 error against a numerical (Radau) reference.

**Inverse problem: not yet started.** See [Roadmap](#roadmap) below.

## The model

Hodgkin–Huxley describes the action-potential dynamics of a single neuron as
four coupled ODEs:

```
C_m dV/dt = I_ext − g_Na m³h(V−E_Na) − g_K n⁴(V−E_K) − g_L(V−E_L)
      dm/dt = α_m(V)(1−m) − β_m(V)m
      dh/dt = α_h(V)(1−h) − β_h(V)h
      dn/dt = α_n(V)(1−n) − β_n(V)n
```

A single MLP maps time `t → [V, m, h, n]`. Voltage is output as `raw × 100`
(the network predicts `V/100`); gates are output through a sigmoid, keeping
them bounded in `(0, 1)`. Time derivatives are computed by forward-mode
automatic differentiation (`jax.jacfwd`) rather than finite differences.

The entire project is built directly on the **JAX ecosystem** — `jax` for
array operations and automatic differentiation, `optax` for optimization. The
MLP (`model.py`), training loop (`trainer.py`), and residual computation
(`hh.py`) are all hand-written against `jax`/`jax.numpy` primitives with no
higher-level framework in between. There is no TensorFlow/Keras or PyTorch
anywhere in this codebase.

## Why voltage-only is hard: the flat-attractor problem

Training this system with physics alone (no data) reliably collapses to a
smooth, non-spiking equilibrium (`V ≈ −60` mV) rather than the true spiking
solution. This is not a bug in this implementation — it's a documented failure
mode for standard collocation-based PINNs on stiff, oscillatory systems
(Kainth et al., 2025 report the identical collapse independently, on the same
equations). The spiking trajectory is a harder-to-reach basin in the loss
landscape than the flat one, and residual minimisation alone finds the easy
basin every time.

The fix used here is a **hybrid data + physics** approach: a small amount of
sparse, noisy voltage data anchors the network near the true trajectory, and
the physics residual is layered in afterward to enforce consistency and infer
the unobserved gates. This is the same approach used in the PINN literature
for this class of problem (Ferrante et al., 2022; Wei et al., 2026) — full
end-to-end forecasting or physics-only extrapolation is not the target; dense
reconstruction within an observed window is.

## Training strategy

Training proceeds through four phases, each initialised from the previous
phase's trained parameters:

| Phase | `data_mask` (V, m, h, n) | `residual_weight` | `loss_weights` (V, m, h, n) | Purpose |
|---|---|---|---|---|
| **0 — Data generation** | — | — | — | Radau reference (dense grid) + sparse noisy voltage samples, concentrated around the spike windows |
| **Pre-training** | `(T,F,F,F)` | `0.0` | — | Fit V to data with physics off. Gates receive no signal beyond the initial condition |
| **Phase A** | `(T,F,F,F)` | `1.0` | `(0.0, w, w, w)` | Physics switched on for the gates only. V's own physics term stays at zero so its data fit is undisturbed; `m,h,n` are pulled into ODE-consistency with the now-accurate V |
| **Phase B** | `(T,F,F,F)` | `1.0` | `(0.02, w, w, w)` | Joint fine-tune — a small physics weight is reintroduced for V on top of its continuous data supervision |

`data_mask` is never `(T,T,T,T)` in this project — voltage is the only
variable ever supervised by data, at any phase. The gates are reconstructed
entirely by the physics residual, using the coupling terms in the ODE system.

Two specific choices matter and are not arbitrary:

- **Domain length.** Training is done on a short window (containing 1–2
  spikes) rather than a long multi-spike trace, following the short-window
  regime used in Wei et al. (2026). A longer domain with the same data budget
  spreads point density thin and degrades gate reconstruction substantially.
- **Spike-biased sampling.** Data points are concentrated in the narrow
  (~2–3 ms) spike windows rather than sampled uniformly at random. Error
  concentrates almost entirely at the spikes under uniform sampling (measured
  at >60% local relative error vs. ~3% elsewhere) — a resolution problem, not
  a noise problem. Biasing sample density toward the fast dynamics fixes it
  directly.

## Results

Final reconstruction accuracy (relative L2 error vs. Radau reference,
voltage-only supervision, gates fully latent):

| Variable | Pre-training (physics off) | Phase A | Phase B (final) |
|---|---|---|---|
| V | 7.7% | 5.1% | **2.7%** |
| m | 148.9% | 5.9% | **3.2%** |
| h | 131.8% | 3.8% | **5.1%** |
| n | 60.0% | 1.9% | **2.0%** |

The pre-training column is the clearest evidence that this is a physics
result, not a data-fitting result: `m, h, n` are essentially untrained at that
point (over 100% relative error — no data ever touches them beyond the
initial condition), and Phase A alone, using only the ODE residual, brings all
three under 6%.

## Repository structure

```
hodgkin-huxley-pinn/
├── hh_pinn/
│   ├── __init__.py     # exports: HHProblem, init_params, apply, Trainer
│   ├── base.py         # DEProblem abstract interface (ODE/PDE agnostic)
│   ├── hh.py           # HHProblem: residual, supervised data, IC, Radau reference
│   ├── model.py        # plain MLP: init_params, apply
│   └── trainer.py      # collocation training loop, loss weighting, optimizer
├── notebook/
│   └── hh_demo.ipynb   # end-to-end walkthrough: data generation through Phase B
├── requirements.txt
├── .gitignore
└── .gitattributes
```

## Requirements

See `requirements.txt`. Core dependencies: `jax`, `optax`, `numpy`, `scipy`
(for the Radau reference solver), `matplotlib` (notebook plots). No
TensorFlow/Keras or PyTorch is used anywhere in this project — the network,
training loop, and autodiff are all pure JAX/Optax.

## Usage

```python
from hh_pinn import HHProblem, init_params, Trainer

problem = HHProblem(I_ext=10.0, t_end=30.0,
                    data_t=data_t, data_u=data_u,
                    data_mask=[True, False, False, False])

params, activation = init_params(input_dim=1, hidden=(128, 128, 64),
                                 output_dim=4, activation='tanh')

trainer = Trainer(params, activation, problem,
                  lr=1e-3, residual_weight=0.0)   # pre-training: physics off
trainer.train(steps=10_000, batch_size=64)
```

See `notebook/hh_demo.ipynb` for the full four-phase pipeline, including data
generation and evaluation against the Radau reference.

## Roadmap

- [x] Forward problem — full-state reconstruction from voltage-only data
- [ ] **Inverse problem** — joint reconstruction of hidden state *and*
      estimation of unknown biophysical parameters (e.g. `g_Na`, `I_ext`) from
      the same class of sparse, noisy voltage data. *To be filled in.*

## References

- Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of
  membrane current and its application to conduction and excitation in nerve.
- Ferrante, M. et al. (2022). Physically constrained neural networks for
  inferring physiological system models. arXiv:2209.11998.
- Kainth, K. et al. (2025). Physics-Informed Neural ODEs with Scale-Aware
  Residuals for Learning Stiff Biophysical Dynamics. arXiv:2511.11734.
- Wei, C., Wang, Y., & Zhu, X. (2026). Robust Parameter and State Estimation
  in Multiscale Neuronal Systems Using Physics-Informed Neural Networks.
  arXiv:2603.08742.

Detailed experiment logs, ablations, and debugging notes are tracked
separately in the project's research notes.
