import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd
import haiku as hk
from . import nnutils as nn


def subdivide_polyline(polyline):
    B, T, C = polyline.shape
    T_new = T * 2 - 1
    resized = jax.vmap(jax.partial(jax.image.resize, shape=(T_new, C), method='linear'))(polyline)
    return resized


def sample_at_vertices(vertices: jnp.ndarray, features: jnp.ndarray) -> jnp.ndarray:
    H, W, C = features.shape
    vertices = (vertices + 1.0) * (jnp.array([H-1, W-1]) / 2.0)
    def resample_feature(feature_map: jnp.ndarray):
        return jnd.map_coordinates(feature_map, vertices.T, order=1, mode='constant')

    resampled = jax.vmap(resample_feature, in_axes=-1, out_axes=-1)(features)

    return resampled


class SnakeHead(hk.Module):
    def __init__(self, channels, coord_features=False):
        super().__init__()
        self.channels = channels
        self.coord_features = coord_features

    def __call__(self, vertices, feature_maps):
        C = self.channels

        blocks = hk.Sequential([
            hk.Conv1D(C, 1), nn.ReLU(),
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
        if self.coord_features:
            features.append(vertices)
        input_features = jnp.concatenate(features, axis=-1)

        convolved_features = blocks(input_features)
        offsets = mk_offset(convolved_features)
        return vertices + offsets

