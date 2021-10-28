import jax
import jax.numpy as jnp
import haiku as hk

from . import backbones
from . import nnutils as nn
from .snake_utils import SnakeHead, AuxHead

class DeepSnake():
    def __init__(self, backbone, vertices=64,
            model_dim=64, iterations=5, coord_features=False, stop_grad=True):
        super().__init__()
        self.backbone = getattr(backbones, backbone)
        self.model_dim = model_dim
        self.iterations = iterations
        self.coord_features = coord_features
        self.vertices = vertices
        self.stop_grad = stop_grad

    def __call__(self, imagery, is_training=False):
        backbone = self.backbone()
        feature_maps = backbone(imagery, is_training)

        vertices = jnp.zeros([imagery.shape[0], self.vertices, 2])
        steps = [vertices]

        _head = SnakeHead(self.model_dim, self.coord_features)
        head = lambda x, y: _head(x, y)

        for _ in range(self.iterations):
            if self.stop_grad:
                vertices = jax.lax.stop_grad(vertices)
            vertices = vertices + head(vertices, feature_maps)
            steps.append(vertices)

        # aux_pred = AuxHead(*imagery.shape[1:3])(feature_maps)
        return steps, aux_pred
