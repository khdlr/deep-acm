import jax
import jax.numpy as jnp
import haiku as hk

from . import backbones
from . import nnutils as nn
from .snake_utils import SnakeHead, AuxHead, channel_dropout


class UNet:
    def __init__(self, width):
        self.width = width

    def __call__(self, x, is_training=False):
        skip_connections = []

        W = self.width
        channel_seq = [W, 2*W, 4*W, 8*W]
        for channels in channel_seq:
            x = Convx2(channels)(x)
            skip_connections.append(x)
            x = hk.max_pool(x, 2, 2, padding='SAME')

        x = Convx2(16*W)(x)

        for channels, skip in zip(reversed(channel_seq), reversed(skip_connections)):
            B,  H,  W,  C  = x.shape
            B_, H_, W_, C_ = skip.shape

            upsampled = jax.image.resize(x, [B, H_, W_, C], method='bilinear')
            x = jax.nn.relu(hk.Conv2D(C, 2)(upsampled))
            x = Convx2(channels)(jnp.concatenate([x, skip], axis=-1))

        x = hk.Conv2D(1, 1)(x)
        return x


def Convx2(channels):
    return hk.Sequential([
        hk.Conv2D(channels, 3),
        jax.nn.relu,
        hk.Conv2D(channels, 3),
        jax.nn.relu,
    ])
