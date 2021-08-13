import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd
import haiku as hk

from .backbones import SimpleBackbone
from . import nnutils as nn


def sample_at_vertices(vertices: jnp.ndarray, features: jnp.ndarray) -> jnp.ndarray:
    def resample_feature(feature_map: jnp.ndarray):
        return jnd.map_coordinates(feature_map, vertices.T, order=1, mode='constant')

    resampled = jax.vmap(resample_feature, in_axes=-1, out_axes=-1)(features)

    return resampled


class SnakeHead(hk.Module):
    def __call__(self, vertices, features):
        blocks = hk.Sequential([
            hk.Conv1D(512, 3), nn.ReLU(),
            hk.Conv1D(512, 3, rate=3), nn.ReLU(),
            hk.Conv1D(512, 3, rate=9), nn.ReLU(),
            hk.Conv1D(512, 3, rate=9), nn.ReLU(),
            hk.Conv1D(512, 3, rate=3), nn.ReLU(),
            hk.Conv1D(512, 3), nn.ReLU(),
        ], name='SnakeBlocks')

        mk_offset = hk.Conv1D(2, 1, with_bias=False, w_init=hk.initializers.Constant(0))

        sampled_features = jax.vmap(sample_at_vertices, [0, 0])(vertices, features)
        convolved_features = blocks(sampled_features)
        offsets = mk_offset(convolved_features)
        return vertices + offsets


class DeepSnake(hk.Module):
    def __call__(self, imagery, initialization, iterations=5):
        backbone = SimpleBackbone()
        head = SnakeHead()

        vertices = initialization
        feature_map = backbone(imagery)
        for _ in range(iterations):
            vertices = head(vertices, feature_map)
        return vertices
