import jax
import jax.numpy as jnp
import haiku as hk

class ReLU(hk.Module):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.relu(x)

class LeakyReLU(hk.Module):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.leaky_relu(x)
