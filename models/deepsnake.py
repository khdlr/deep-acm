import jax
import jax.numpy as jnp
import haiku as hk

from . import backbones
from . import nnutils as nn
from .snake_utils import SnakeHead, subdivide_polyline

class DeepSnake():
    def __init__(self, backbone, vertices=64,
            model_dim=64, iterations=5, coord_features=False, subdivide=False, weight_sharing=True):
        super().__init__()
        self.backbone = getattr(backbones, backbone)
        self.model_dim = model_dim
        self.iterations = iterations
        self.coord_features = coord_features
        self.subdivide = subdivide
        self.weight_sharing = weight_sharing
        self.vertices = vertices

    def __call__(self, imagery, is_training=False):
        backbone = self.backbone()
        feature_maps = backbone(imagery, is_training)

        if self.subdivide:
            vertices = jnp.zeros([imagery.shape[0], 3, 2])
        else:
            vertices = jnp.zeros([imagery.shape[0], self.vertices, 2])
        steps = [vertices]

        if self.weight_sharing:
            _head = SnakeHead(self.model_dim, self.coord_features)
            head = lambda x, y: _head(x, y)
        else:
            head = lambda x, y: SnakeHead(self.model_dim, self.coord_features)(x, y)

        if self.subdivide:
            vertices = head(vertices, feature_maps)
            steps.append(vertices)
        for _ in range(self.iterations):
            if self.subdivide:
                vertices = subdivide_polyline(vertices)
            vertices = head(vertices, feature_maps)
            steps.append(vertices)

        if is_training:
            return vertices
        else:
            return steps
