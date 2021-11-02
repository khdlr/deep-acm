import jax
import jax.numpy as jnp
import haiku as hk
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from . import backbones
from . import nnutils as nn
from .snake_utils import channel_dropout, sample_at_vertices


class ProbabilisticDeepSnake():
    def __init__(self, backbone, vertices=64,
            model_dim=64, iterations=5, coord_features=False, stop_grad=True):
        super().__init__()
        self.backbone = getattr(backbones, backbone)
        self.model_dim = model_dim
        self.iterations = iterations
        self.coord_features = coord_features
        self.vertices = vertices
        self.stop_grad = stop_grad

    def __call__(self, imagery, inference_mode='sample', is_training=False):
        backbone = self.backbone()
        feature_maps = backbone(imagery, is_training)

        if is_training:
            feature_maps = [channel_dropout(f, 0.5) for f in feature_maps]

        vertices = jnp.zeros([imagery.shape[0], self.vertices, 2])
        steps = []

        head = ProbabilisticSnakeHead(self.model_dim, self.coord_features)

        for _ in range(self.iterations):
            if self.stop_grad:
                vertices = jax.lax.stop_gradient(vertices)

            mu, tril = head(vertices, feature_maps)
            loc = vertices + mu
            params = {'loc': loc, 'scale_tril': tril}
            # TODO: Don't validate args when sure that they're fine
            P = tfd.MultivariateNormalTriL(**params, validate_args=True)
            steps.append(params)

            if inference_mode == 'sample':
                vertices = P.sample(seed=hk.next_rng_key())
            elif inference_mode == 'mean':
                vertices = P.loc
            else:
              raise ValueError(f"Unknown inference_mode '{inference_mode}'")

        return steps


class ProbabilisticSnakeHead(hk.Module):
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

        # Initialize offset predictors with 0 -> default to some std gaussian
        zero = hk.initializers.Constant(0.0)
        mk_mu      = hk.Conv1D(2, 1, with_bias=False, w_init=zero)
        mk_diag    = hk.Conv1D(2, 1, with_bias=False, w_init=zero)
        mk_offdiag = hk.Conv1D(1, 1, with_bias=False, w_init=zero)

        features = []
        for feature_map in feature_maps:
            features.append(jax.vmap(sample_at_vertices, [0, 0])(vertices, feature_map))
        # For coordinate features
        if self.coord_features:
            diff = vertices[:,1:] - vertices[:,:-1]
            diff = jnp.pad(diff, [(0,0), (1,1), (0,0)])
            features.append(diff[:, 1:])
            features.append(diff[:, :-1])
        input_features = jnp.concatenate(features, axis=-1)

        convolved_features = blocks(input_features)

        @jax.vmap
        @jax.vmap
        def inflate(diag, offdiag):
          diag = jax.nn.softplus(diag)
          return jnp.diag(diag) + jnp.array([[0, 0], [1, 0]]) * offdiag

        # B x T x 2
        mu = mk_mu(convolved_features)
        tril_diag    = mk_diag(convolved_features)
        tril_offdiag = mk_offdiag(convolved_features)
        tril = inflate(tril_diag, tril_offdiag)

        return mu, tril
