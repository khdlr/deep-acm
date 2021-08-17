import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd
import haiku as hk

from .backbones import SimpleBackbone
from . import nnutils as nn


def sample_at_vertices(vertices: jnp.ndarray, features: jnp.ndarray) -> jnp.ndarray:
    H, W, C = features.shape
    vertices = (vertices + 1.0) * (jnp.array([H, W]) / 2.0)
    def resample_feature(feature_map: jnp.ndarray):
        return jnd.map_coordinates(feature_map, vertices.T, order=1, mode='constant')

    resampled = jax.vmap(resample_feature, in_axes=-1, out_axes=-1)(features)

    return resampled


class SnakeHead(hk.Module):
    def __init__(self, multiplier):
        super().__init__()
        self.multiplier = multiplier

    def __call__(self, vertices, features):
        C = 8 * self.multiplier

        blocks = hk.Sequential([
            hk.Conv1D(C, 3), nn.ReLU(),
            hk.Conv1D(C, 3, rate=3), nn.ReLU(),
            hk.Conv1D(C, 3, rate=9), nn.ReLU(),
            hk.Conv1D(C, 3, rate=9), nn.ReLU(),
            hk.Conv1D(C, 3, rate=3), nn.ReLU(),
            hk.Conv1D(C, 3), nn.ReLU(),
        ], name='SnakeBlocks')

        mk_offset = hk.Conv1D(2, 1, with_bias=False, w_init=hk.initializers.Constant(0))

        sampled_features = jax.vmap(sample_at_vertices, [0, 0])(vertices, features)
        coord_features = vertices
        input_features = jnp.concatenate([sampled_features, coord_features], axis=-1)

        convolved_features = blocks(input_features)
        offsets = mk_offset(convolved_features)
        return vertices + offsets


class DeepSnake():
    def __init__(self, multiplier=64, output_intermediates=False, iterations=5):
        super().__init__()
        self.output_intermediates = output_intermediates
        self.multiplier = multiplier
        self.iterations = iterations

    def __call__(self, imagery, initialization):
        backbone = SimpleBackbone(self.multiplier)
        head = SnakeHead(self.multiplier)

        vertices = initialization
        feature_map = backbone(imagery)

        steps = []
        for _ in range(self.iterations):
            vertices = head(vertices, feature_map)
            steps.append(vertices)

        if self.output_intermediates:
            return jnp.stack(steps, axis=1)
        else:
            return vertices
