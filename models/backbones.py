import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd
import haiku as hk

from . import nnutils as nn


class ResBlock(hk.Module):
    def __init__(self, dim, inner_dim=32):
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        init  = hk.Conv2D(self.inner_dim, 1)
        conv1 = hk.Conv2D(self.inner_dim, 3)
        conv2 = hk.Conv2D(self.inner_dim, 3)
        post  = hk.Conv2D(self.dim, 1)

        skip = x
        x = init(jax.nn.relu(x))
        x = conv1(jax.nn.relu(x))
        x = conv2(jax.nn.relu(x))
        x = post(jax.nn.relu(x))
        return skip + x


class SimpleBackbone(hk.Module):
    def __call__(self, x):
        block1 = hk.Sequential([
            hk.Conv2D(64, 4, 2),
            ResBlock(64),
            ResBlock(64)
        ], name='BackboneBlock1')

        block2 = hk.Sequential([
            hk.Conv2D(128, 4, 2),
            ResBlock(128),
            ResBlock(128)
        ], name='BackboneBlock2')

        block3 = hk.Sequential([
            hk.Conv2D(256, 4, 2),
            ResBlock(256),
            ResBlock(256)
        ], name='BackboneBlock2')

        block4 = hk.Sequential([
            hk.Conv2D(512, 4, 2),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512)
        ], name='BackboneBlock2')

        x = block1(x)
        x = block2(x)
        x = block3(x)
        x = block4(x)
        return x
