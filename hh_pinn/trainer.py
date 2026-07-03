import time
import jax
import jax.numpy as jnp
import optax

from .model import apply


class Trainer:
    """
    Train an MLP (represented as params pytree) to satisfy a DEProblem via
    collocation loss minimisation.

    Loss = sqrt(mean([residual_weight * mse_res, supervised_weight * mse_supervised]))

    mse_supervised is computed via problem.build_supervised() when available
    (returns per-point, per-variable weights), otherwise falls back to
    problem.constraints() with uniform weighting for legacy DEProblem subclasses.

    Optional extensions (all default OFF / no-op):
        loss_weights       : per-output-variable weights on the squared residual
        residual_weight    : scalar multiplier on mse_res; set to 0.0 to disable
                             physics during pre-training
        supervised_weight  : scalar multiplier on mse_supervised
        use_causal_weights : re-weight residual by exp(-eps * t/t_end)
        causal_eps         : controls causal window width
        use_grad_clip      : clip global gradient norm before Adam update
        grad_clip_norm     : norm threshold for gradient clipping
    """

    def __init__(
            self,
            params: dict,
            activation: str,
            problem,
            lr: float = 1e-3,
            use_lr_decay : bool = False,
            loss_weights=None,
            residual_weight: float = 1.0,
            supervised_weight: float = 1.0,
            use_causal_weights: bool = False,
            causal_eps: float = 1.0,
            use_grad_clip: bool = False,
            grad_clip_norm: float = 1.0
    ):
        self.params = params
        self.activation = activation
        self.problem = problem

        self.loss_weights = None if loss_weights is None else jnp.array(loss_weights)
        self.residual_weight = residual_weight
        self.supervised_weight = supervised_weight
        self.use_causal_weights = use_causal_weights
        self.causal_eps = causal_eps

        if use_lr_decay:
            schedule = optax.exponential_decay(
                init_value=lr, transition_steps=5_000, decay_rate=0.5
            )
            optimizer = optax.adam(schedule)
        else:
            optimizer = optax.adam(lr)
        
        if use_grad_clip:
            optimizer = optax.chain(optax.clip_by_global_norm(grad_clip_norm), optimizer)
        
        self.optimizer = optimizer
        self.opt_state = self.optimizer.init(params)
    
    def _sample_collocation(self, batch_size: int, key: jax.Array):
        bounds = self.problem.domain()
        if 'x_lb' in bounds:
            key1, key2 = jax.random.split(key)
            xs = jax.random.uniform(
                key1, (batch_size, 1), minval=bounds['x_lb'], maxval=bounds['x_ub']
            )
            ts = jax.random.uniform(
                key2, (batch_size, 1), minval=bounds['t_lb'], maxval=bounds['t_ub']
            )

            return jnp.concatenate([xs, ts], axis=1)
        else:
            return jax.random.uniform(key, (batch_size, 1), minval=bounds['lb'], maxval=bounds['ub'])

    def _apply(self, params, x):
        return apply(params, x, self.activation)
    
    def _loss(self, params: dict, xt: jnp.ndarray):
        # --- Physics residual ---
        residual = self.problem.residual(params, self._apply, xt)

        if self.loss_weights is not None:
            mse_res = jnp.mean((residual ** 2) * self.loss_weights[None, :])
        else:
            mse_res = jnp.mean(residual ** 2)
        
        if self.use_causal_weights:
            t_ub = self.problem.domain()['ub']
            causal_w = jnp.exp(-self.causal_eps * (xt[:, -1] / t_ub))
            if self.loss_weights is not None:
                mse_res = jnp.mean(
                    (residual ** 2) * self.loss_weights[None, :] * causal_w[:, None]
                )
            else:
                mse_res = jnp.mean((residual ** 2) * causal_w[:, None])

        # --- Supervised term (IC + data) ---
        if hasattr(self.problem, 'build_supervised'):
            pred, tgt, weights = self.problem.build_supervised(params, self._apply)
            sq_err = weights * (pred - tgt) ** 2
            mse_supervised = jnp.sum(sq_err) / (jnp.sum(weights) + 1e-8)
        else:
            pred, tgt = self.problem.constraints(params, self._apply)
            mse_supervised = jnp.sum((pred - tgt) ** 2)

        return jnp.sqrt(jnp.mean(jnp.array([
            self.residual_weight * mse_res, self.supervised_weight * mse_supervised
        ])))
    
    def _make_step(self):
        """Returns a JIT-compiled step function closed over the optimizer."""
        optimizer = self.optimizer

        @jax.jit
        def step(params, opt_state, xt):
            loss_val, grads = jax.value_and_grad(self._loss)(params, xt)
            updates, opt_state_new = optimizer.update(grads, opt_state, params)
            params_new = optax.apply_updates(params, updates)

            return params_new, opt_state_new, loss_val
        
        return step
    
    def train(
            self,
            steps: int = 10_000,
            batch_size: int = 50,
            tol: float = 0.05,
            log_every: int = 1_000,
            seed: int = 303
    ):
        """
        Run training loop.
        Returns list of mean losses recorded at each log interval.
        """

        step_fn = self._make_step()
        key = jax.random.PRNGKey(seed)
        history = []
        cum_loss = 0.0
        t0 = time.time()

        for i in range(steps):
            key, subkey = jax.random.split(key)
            xt = self._sample_collocation(batch_size, subkey)

            self.params, self.opt_state, loss_val = step_fn(
                self.params, self.opt_state, xt
            )
            loss_val = float(loss_val)
            cum_loss += loss_val

            if (i + 1) % log_every == 0:
                mean_loss = cum_loss / log_every

                # --- per-variable residual norms (Phase 2+) ---
                residual = self.problem.residual(self.params, self._apply, xt)
                res_per_var = jnp.mean(residual ** 2, axis=0)  #shape (k,)
                
                history.append({
                    'step': i + 1,
                    'mean_loss': mean_loss,
                    'res_per_var': res_per_var
                })
                elapsed = time.time() - t0
                print(f"step {i+1:>6d} | mean loss: {mean_loss:.4f} | {elapsed:.1f}s")
                cum_loss = 0.0

            if loss_val <= tol:
                print(f"converged at step {i+1} with loss {loss_val:.4f}")
                history.append({'step': i + 1, 'mean_loss': loss_val, 'res_per_var': None})
                break

        return history
