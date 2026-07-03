import jax
import jax.numpy as jnp


_activations = {
    'sigmoid': jax.nn.sigmoid,
    'tanh': jax.nn.tanh,
    'relu': jax.nn.relu
}

def init_params(
        input_dim: int,
        hidden: tuple = (128, 64),
        output_dim: int = 1,
        activation: str = 'sigmoid',
        key: jax.Array = None
):
    """
    Initialise MLP parameters using Glorot uniform for weights, small normal for biases.
    Returns a dict: {'Ws': [...], 'bs': [...], 'activation': str}
    """

    if key is None:
        key = jax.random.PRNGKey(0)
    
    layer_sizes = [input_dim] + list(hidden) + [output_dim]
    Ws, bs = [], []

    for i in range(len(layer_sizes) - 1):
        fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
        limit = jnp.sqrt(6.0 / (fan_in + fan_out))    #Glorot uniform init
        key, subkey = jax.random.split(key)
        Ws.append(jax.random.uniform(subkey, (fan_in, fan_out), minval=-limit, maxval=limit))
        bs.append(jnp.zeros(fan_out))
    
    return {'Ws': Ws, 'bs': bs}, activation

def apply(params: dict, x, activation: str = 'sigmoid'):
    """
    Forward pass: x -> MLP output.
    x shape: (batch, input_dim) or (input_dim,) for scalar input problems.
    """

    act = _activations[activation]
    h = x
    for W, b in zip(params['Ws'][:-1], params['bs'][:-1]):
        h = act(h @ W + b)

    return h @ params['Ws'][-1] + params['bs'][-1]