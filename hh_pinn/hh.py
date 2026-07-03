import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import solve_ivp

from .base import DEProblem


_V_SCALE = 100.0

def _u(params, apply_fn, t_scalar):
    """
    Network output at a single time point, with output normalization.

    Returns [V, m, h, n], shape (4,).
        - V: raw output * 100 (network predicts V / 100)
        - m, h, n: sigmoid of raw output (gates bounded in (0, 1))
    """

    raw = apply_fn(params, t_scalar.reshape(1, 1)).squeeze()   #shape (4,)
    V = raw[0] * _V_SCALE
    m = jax.nn.sigmoid(raw[1])
    h = jax.nn.sigmoid(raw[2])
    n = jax.nn.sigmoid(raw[3])

    return jnp.array([V, m, h, n])

def _du_dt(params, apply_fn, t_scalar):
    """
    Time derivatives of all four state variables via forward-mode AD.
    Returns [dV/dt, dm/dt, dh/dt, dn/dt], shape (4,).
    jacfwd computes all four derivatives in one pass (cf. pde2.py per variable grad).
    """

    return jax.jacfwd(lambda t: _u(params, apply_fn, t))(t_scalar)  #shape (4,)

def _alpha_m(V):
    dV = V + 40.0
    return jnp.where(jnp.abs(dV) < 1e-7, 1.0, 0.1 * dV / (1.0 - jnp.exp(-dV / 10.0))) #L'hopital limit

def _beta_m(V):
    return 4.0 * jnp.exp(-(V + 65.0) / 18.0)

def _alpha_h(V):
    return 0.07 * jnp.exp(-(V + 65.0) / 20.0)

def _beta_h(V):
    return 1.0 / (1.0 + jnp.exp(-(V + 35.0) / 10.0))

def _alpha_n(V):
    dV = V + 55.0
    return jnp.where(jnp.abs(dV) < 1e-7, 0.1, 0.01 * dV / (1.0 - jnp.exp(-dV / 10.0)))

def _beta_n(V):
    return 0.125 * jnp.exp(-(V + 65.0) / 80.0)


class HHProblem(DEProblem):
    """
    Hodgkin-Huxley ODE system:

        C_m * dV/dt = I_ext - gNa*m^3*h*(V-ENa) - gK*n^4*(V-EK) - gL*(V-EL)
        dm/dt = alpha_m(V)*(1-m) - beta_m(V)*m
        dh/dt = alpha_h(V)*(1-h) - beta_h(V)*h
        dn/dt = alpha_n(V)*(1-n) - beta_n(V)*n
    
    Network: t (scalar) -> [V, m, h, n] (output_dim=4)
    """

    def __init__(
            self,
            t_start: float = 0.0,      #ms
            t_end:   float = 50.0,     #ms
            I_ext:   float = 10.0,     #uA/cm^2
            C_m:     float = 1.0,      #uF/cm^2
            gNa:     float = 120.0,    #mS/cm^2
            gK:      float = 36.0,     #mS/cm^2
            gL:      float = 0.3,      #mS/cm^2
            ENa:     float = 50.0,     #mV
            EK:      float = -77.0,    #mV
            EL:      float = -54.387,  #mV
            V0:      float = -65.0,    #mV -- resting IC
            m0:      float = 0.0529,
            h0:      float = 0.5961,
            n0:      float = 0.3177,
            data_t=None,
            data_u=None,
            data_mask=None,
            ic_weight: float = 50.0
    ):
        self.t_start = t_start
        self.t_end = t_end
        self.I_ext = I_ext
        self.C_m = C_m
        self.gNa = gNa
        self.gK = gK
        self.gL = gL
        self.ENa = ENa
        self.EK = EK
        self.EL = EL
        self.V0 = V0
        self.m0 = m0
        self.h0 = h0
        self.n0 = n0
        self.ic_weight = ic_weight

        if data_t is not None:
            self._data_t = jnp.asarray(data_t).reshape(-1)
            self._data_u = jnp.asarray(data_u).reshape(-1, 4)            
            self._data_u = self._data_u.at[:, 0].set(self._data_u[:, 0] / _V_SCALE)
            self._data_mask = jnp.asarray(
                data_mask if data_mask is not None else [True, True, True, True], dtype=bool
            )
        else:
            self._data_t = None
            self._data_u = None
            self._data_mask = None

    def build_supervised(self, params, apply_fn):
        """
        Build the unified supervised batch combining IC and (optionally) data.

        Returns:
            pred: network predictions at all supervised time points,
                  normalised (V / _V_SCALE, gates as-is), shape (M, 4)
            target: corresponding targets, same normalisation, shape (M, 4)
            weights: per-point per-variable weights
                     IC row   -> ic_weight where observed, 0 elsewhere, shape (M, 4)  

        The trainer computes:
            mse_supervised = mean(weights * (pred - target)**2)
        Zero-weight entries contribute nothing to the loss, so unobserved
        variables (data_mask=False) are naturally ignored without branching.        
        """

        t0 = jnp.array(self.t_start)
        _scale = jnp.array([_V_SCALE, 1.0, 1.0, 1.0])
        ic_pred = _u(params, apply_fn, t0) / _scale   #(4,) normalised
        ic_tgt = jnp.array([
            self.V0 / _V_SCALE, self.m0, self.h0, self.n0
        ])
        ic_w = jnp.full((4,), self.ic_weight)

        if self._data_t is None:
            # No data: IC only -- original behaviour preserved
            pred = ic_pred.reshape(1, 4)
            tgt = ic_tgt.reshape(1, 4)
            weights = ic_w.reshape(1, 4)

            return pred, tgt, weights

        data_pred = jax.vmap(lambda ti: _u(params, apply_fn, ti) / _scale)(self._data_t)
        data_w = jnp.where(
            self._data_mask[None, :],
            jnp.ones((len(self._data_t), 4)),
            jnp.zeros((len(self._data_t), 4))
        )

        pred = jnp.concatenate([ic_pred.reshape(1, 4), data_pred], axis=0)
        tgt = jnp.concatenate([ic_tgt.reshape(1, 4), self._data_u], axis=0)
        weights = jnp.concatenate([ic_w.reshape(1, 4), data_w], axis=0)

        return pred, tgt, weights


    def residual(self, params, apply_fn, t):
        """
        t: (batch, 1) collocation points in ms.
        Returns (batch, 4) residuals -- one column per ODE.
        """

        def res_single(ti):
            state = _u(params, apply_fn, ti)       #[V, m, h, n]
            dstate = _du_dt(params, apply_fn, ti)  #[dV/dt, dm/dt, dh/dt, dn/dt]

            V, m, h, n = state[0], state[1], state[2], state[3]

            INa = self.gNa * m**3 * h * (V - self.ENa)
            IK  = self.gK  * n**4     * (V - self.EK)
            IL  = self.gL             * (V - self.EL)

            am, bm = _alpha_m(V), _beta_m(V)
            ah, bh = _alpha_h(V), _beta_h(V)
            an, bn = _alpha_n(V), _beta_n(V)

            res_V = (dstate[0] - (self.I_ext - INa - IK - IL) / self.C_m) / _V_SCALE
            res_m = dstate[1] - (am * (1.0 - m) - bm * m)
            res_h = dstate[2] - (ah * (1.0 - h) - bh * h)
            res_n = dstate[3] - (an * (1.0 - n) - bn * n)

            return jnp.array([res_V, res_m, res_h, res_n])
            
        return jax.vmap(res_single)(t[:, 0])   #shape (batch, 4)
        
    def constraints(self, params, apply_fn):
        """
        Required by DEProblem interface.
        Delegates to build_supervised() and returns (pred, target) in the
        flat (M*4, 1) shape expected by the trainer's existing _loss.
        The weights are handled separately via build_supervised() when the
        trainer calls it directly.
        """

        pred, tgt, _ = self.build_supervised(params, apply_fn)

        return pred.reshape(-1, 1), tgt.reshape(-1, 1)
    
    def domain(self):
        """1-D time domain — matches the ODE branch in the trainer."""
        return {'lb': self.t_start, 'ub': self.t_end}
    
    def true_solution(self, t_arr):
        """
        Numerical reference via scipy solve_ivp with Radau (stiff solver).
        t_arr: 1-D Numpy array of time points.
        Returns [V, m, h, n] at each point, shape (N, 4) 
        """

        def _ode_rhs(t, y):
            V, m, h, n = y
            INa = self.gNa * m**3 * h * (V - self.ENa)
            IK  = self.gK  * n**4     * (V - self.EK)
            IL  = self.gL             * (V - self.EL)
            am, bm = float(_alpha_m(V)), float(_beta_m(V))
            ah, bh = float(_alpha_h(V)), float(_beta_h(V))
            an, bn = float(_alpha_n(V)), float(_beta_n(V))
            dVdt = (self.I_ext - INa - IK - IL) / self.C_m
            dmdt = am * (1.0 - m) - bm * m
            dhdt = ah * (1.0 - h) - bh * h
            dndt = an * (1.0 - n) - bn * n

            return [dVdt, dmdt, dhdt, dndt]
        
        sol = solve_ivp(
            _ode_rhs,
            [self.t_start, self.t_end],
            [self.V0, self.m0, self.h0, self.n0],
            t_eval=np.asarray(t_arr),
            method='Radau',
            rtol=1e-8,
            atol=1e-10
        )

        return sol.y.T    #shape (N, 4) - [V, m, h, n]


        