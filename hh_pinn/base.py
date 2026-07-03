from abc import ABC, abstractmethod


class DEProblem(ABC):
    """
    Abstract interface for a DE problem (ODE or PDE, any output dimension).

    Subclasses implement:
        - residual(params, apply_fn, x): returns the ODE/PDE residual array
        - constraints(params, apply_fn): returns (predicted, target) arrays
        - domain(): returns the sampling bounds as a dict consumed by the trainer
        - true_solution (optional): callable for error evaluation

    The interface supports scalar and vector-valued systems.
    For a system with k state variables, residual() should return shape
    (batch, k) and the network should have output_dim=k.
    """

    @abstractmethod
    def residual(self, params:dict, apply_fn: callable, t):
        """
        Compute PDE/ODE residuals at collocation points.

        Parameters:
            - params: pytree -- network parameters
            - apply_fn: callable -- network forward pass
            - t: jnp.ndarray, shape (batch, input_dim)
        
        Returns:
            jnp.ndarray, shape (batch, output_dim)
        """
    
    @abstractmethod
    def constraints(self, params: dict, apply_fn: callable):
        """
        Returns (prediction, target) for boundary / initial conditions.

        Returns:
            - pred: jnp.ndarray, shape (n_constraints, 1)
            - target: jnp.ndarray, shape (n_constraints, 1)
        """
    
    @abstractmethod
    def domain(self):
        """
        Return domain bounds as a dict.
        
        For 1-D ODE: {'lb': 0.0, 'ub': T}
        For 2-D PDE: {'lb': [x_lo, t_lo], 'ub': [x_hi, t_hi]}
        """
    
    def true_solution(self, t_arr):
        """Optional: numerical reference solution for validation."""
        raise None