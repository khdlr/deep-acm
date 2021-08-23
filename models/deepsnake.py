import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd
import haiku as hk

from .backbones import SimpleBackbone, ResNet50, SlimResNet50
from . import nnutils as nn


def sample_at_vertices(vertices: jnp.ndarray, features: jnp.ndarray) -> jnp.ndarray:
    H, W, C = features.shape
    vertices = (vertices + 1.0) * (jnp.array([H-1, W-1]) / 2.0)
    def resample_feature(feature_map: jnp.ndarray):
        return jnd.map_coordinates(feature_map, vertices.T, order=1, mode='constant')

    resampled = jax.vmap(resample_feature, in_axes=-1, out_axes=-1)(features)

    return resampled


class SnakeHead(hk.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def __call__(self, vertices, feature_maps):
        C = self.channels

        blocks = hk.Sequential([
            hk.Conv1D(C, 3), nn.ReLU(),
            hk.Conv1D(C, 3, rate=3), nn.ReLU(),
            hk.Conv1D(C, 3, rate=9), nn.ReLU(),
            hk.Conv1D(C, 3, rate=9), nn.ReLU(),
            hk.Conv1D(C, 3, rate=3), nn.ReLU(),
            hk.Conv1D(C, 3), nn.ReLU(),
        ], name='SnakeBlocks')

        # Initialize offset predictors with 0 -> default to no change
        mk_offset = hk.Conv1D(2, 1, with_bias=False, w_init=hk.initializers.Constant(0.0))

        features = []
        for feature_map in feature_maps:
            features.append(jax.vmap(sample_at_vertices, [0, 0])(vertices, feature_map))
        # For coordinate features
        # features.append(vertices)
        input_features = jnp.concatenate(features, axis=-1)

        convolved_features = blocks(input_features)
        offsets = mk_offset(convolved_features)
        return vertices + offsets


class DeepSnake():
    def __init__(self, multiplier=64, iterations=5):
        super().__init__()
        # self.output_intermediates = output_intermediates
        self.multiplier = multiplier
        self.iterations = iterations

    def __call__(self, imagery, initialization, is_training=False):
        backbone = ResNet50()
        head = SnakeHead(2048)

        vertices = initialization
        feature_maps = backbone(imagery, is_training)

        steps = []
        for _ in range(self.iterations):
            vertices = head(vertices, feature_maps)
            steps.append(vertices)

        if is_training:
            return vertices
        else:
            return jnp.stack(steps, axis=1)
