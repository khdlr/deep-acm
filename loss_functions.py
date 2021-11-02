import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from abc import ABC, abstractmethod

from utils import pad_inf, fmt, distance_matrix
from metrics import squared_distance_points_to_best_segment
from lib.jump_flood import jump_flood
from einops import rearrange
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions


def l2_loss(prediction, ground_truth):
    if prediction.shape[0] < ground_truth.shape[0]:
        prediction = jax.image.resize(prediction, ground_truth.shape, 'linear')
    elif ground_truth.shape[0] < prediction.shape[0]:
        ground_truth = jax.image.resize(ground_truth, prediction.shape, 'linear')
    loss = jnp.sum(jnp.square(prediction - ground_truth), axis=-1)
    loss = jnp.mean(loss)
    return loss


def l1_loss(prediction, ground_truth):
    if prediction.shape[0] < ground_truth.shape[0]:
        prediction = jax.image.resize(prediction, ground_truth.shape, 'linear')
    elif ground_truth.shape[0] < prediction.shape[0]:
        ground_truth = jax.image.resize(ground_truth, prediction.shape, 'linear')
    loss = jnp.sum(jnp.abs(prediction - ground_truth), axis=-1)
    return loss


def min_min_loss(prediction, ground_truth):
    D = distance_matrix(prediction, ground_truth)
    min1 = D.min(axis=0)
    min2 = D.min(axis=1)
    min_min = 0.5 * (jnp.mean(min1) + jnp.mean(min2))
    return min_min


class AbstractDTW(ABC):
    @abstractmethod
    def minimum(self, *args):
        pass

    def build_distance_matrix(self, prediction, ground_truth):
        return distance_matrix(prediction, ground_truth)

    def __call__(self, prediction, ground_truth):
        return self.dtw(prediction, ground_truth)

    def dtw(self, prediction, ground_truth):
        D = self.build_distance_matrix(prediction, ground_truth)
        # wlog: H >= W
        if D.shape[0] < D.shape[1]:
            D = D.T
        H, W = D.shape

        rows = []
        for row in range(H):
            rows.append( pad_inf(D[row], row, H-row-1) )

        model_matrix = jnp.stack(rows, axis=1)

        init = (
            pad_inf(model_matrix[0], 1, 0),
            pad_inf(model_matrix[1] + model_matrix[0, 0], 1, 0)
        )

        def scan_step(carry, current_antidiagonal):
            two_ago, one_ago = carry

            diagonal = two_ago[:-1]
            right    = one_ago[:-1]
            down     = one_ago[1:]
            best     = self.minimum(jnp.stack([diagonal, right, down], axis=-1))

            next_row = best + current_antidiagonal
            next_row = pad_inf(next_row, 1, 0)

            return (one_ago, next_row), next_row

        # Manual unrolling:
        # carry = init
        # for i, row in enumerate(model_matrix[2:]):
        #     carry, y = scan_step(carry, row)

        carry, ys = jax.lax.scan(scan_step, init, model_matrix[2:], unroll=4)
        return carry[1][-1]


class ProbDTWMixin:
    def build_distance_matrix(self, prediction, ground_truth):
        P = tfd.MultivariateNormalTriL(**prediction)
        ground_truth = rearrange(ground_truth, 'T C -> T 1 C')
        log_prob = jax.vmap(P.log_prob)(ground_truth)
        return log_prob


class DTW(AbstractDTW):
    __name__ = 'DTW'

    def minimum(self, args):
        return jnp.min(args, axis=-1)


def make_softmin(gamma, custom_grad=True):
    """
    We need to manually define the gradient of softmin
    to ensure (1) numerical stability and (2) prevent nans from
    propagating over valid values.
    """
    def softmin_raw(array):
        return -gamma * logsumexp(array / -gamma, axis=-1)
    
    if not custom_grad:
        return softmin_raw

    softmin = jax.custom_vjp(softmin_raw)

    def softmin_fwd(array):
        return softmin(array), (array / -gamma, )

    def softmin_bwd(res, g):
        scaled_array, = res
        grad = jnp.where(jnp.isinf(scaled_array),
            jnp.zeros(scaled_array.shape),
            jax.nn.softmax(scaled_array) * jnp.expand_dims(g, 1)
        )
        return grad,

    softmin.defvjp(softmin_fwd, softmin_bwd)
    return softmin


class SoftDTW(AbstractDTW):
    """
    SoftDTW as proposed in the paper "Soft-DTW: a Differentiable Loss Function for Time-Series"
    by Marco Cuturi and Mathieu Blondel (https://arxiv.org/abs/1703.01541)
    """
    __name__ = 'SoftDTW'

    def __init__(self, gamma=1.0):
        assert gamma > 0, "Gamma needs to be positive."
        self.gamma = gamma
        self.__name__ = f'SoftDTW({self.gamma})'
        self.minimum_impl = make_softmin(gamma)

    def minimum(self, args):
        return self.minimum_impl(args)

        # args = jnp.stack(args, axis=-1) / -self.gamma
        # maxval = jnp.max(args, axis=-1, keepdims=True)
        # logsumexp = jnp.log(jnp.sum(jnp.exp(args - maxval), axis=-1)) + maxval[..., 0]
        # return -self.gamma * logsumexp


class ProbDTW(ProbDTWMixin, DTW):
    pass


class ProbSoftDTW(ProbDTWMixin, SoftDTW):
    pass


def forward_mae(prediction, ground_truth):
    squared_dist = squared_distance_points_to_best_segment(prediction, ground_truth)
    return jnp.mean(jnp.sqrt(squared_dist))


def backward_mae(prediction, ground_truth):
    squared_dist = squared_distance_points_to_best_segment(ground_truth, prediction)
    return jnp.mean(jnp.sqrt(squared_dist))


def forward_rmse(prediction, ground_truth):
    squared_dist = squared_distance_points_to_best_segment(prediction, ground_truth)
    return jnp.sqrt(jnp.mean(squared_dist))


def backward_rmse(prediction, ground_truth):
    squared_dist = squared_distance_points_to_best_segment(ground_truth, prediction)
    return jnp.sqrt(jnp.mean(squared_dist))


def closest_point_loss(prediction, mask):
    true_offsets = jax.lax.stop_gradient(jump_flood(mask[..., 0]))
    error  = jnp.sum(jnp.square(prediction - true_offsets), axis=-1)
    length = jnp.sum(jnp.square(true_offsets), axis=-1)

    return jnp.mean(error / length)
