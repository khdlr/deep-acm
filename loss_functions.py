import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import haiku as hk

from abc import ABC, abstractmethod
from inspect import signature

from utils import pad_inf, fmt, distance_matrix, min_pool
from metrics import squared_distance_points_to_best_segment
from einops import rearrange



def call_loss(loss_fn, prediction, mask, snake, key='loss'):
    sig = signature(loss_fn)
    args = {}
    if 'snake' in sig.parameters:
        args['snake'] = snake
    if 'mask' in sig.parameters:
        args['mask'] = mask

    loss_terms = {}
    if isinstance(prediction, list):
        for i, pred in enumerate(prediction, 1):
            loss_terms[f'{key}_{i}'] = jnp.mean(jax.vmap(loss_fn)(prediction=pred, **args))
    else:
        loss_terms[key] = jnp.mean(jax.vmap(loss_fn)(prediction=prediction, **args))

    return loss_terms


def l2_loss(prediction, snake):
    if prediction.shape[0] < snake.shape[0]:
        prediction = jax.image.resize(prediction, snake.shape, 'linear')
    elif snake.shape[0] < prediction.shape[0]:
        snake = jax.image.resize(snake, prediction.shape, 'linear')
    loss = jnp.sum(jnp.square(prediction - snake), axis=-1)
    loss = jnp.mean(loss)
    return loss


def l1_loss(prediction, snake):
    if prediction.shape[0] < snake.shape[0]:
        prediction = jax.image.resize(prediction, snake.shape, 'linear')
    elif snake.shape[0] < prediction.shape[0]:
        snake = jax.image.resize(snake, prediction.shape, 'linear')
    loss = jnp.sum(jnp.abs(prediction - snake), axis=-1)
    return loss


def min_min_loss(prediction, snake):
    D = distance_matrix(prediction, snake)
    min1 = D.min(axis=0)
    min2 = D.min(axis=1)
    min_min = 0.5 * (jnp.mean(min1) + jnp.mean(min2))
    return min_min


def bce(prediction, mask):
    prediction = prediction[..., 0]
    assert prediction.shape == mask.shape

    logp   = jax.nn.log_sigmoid(prediction)
    log1mp = jax.nn.log_sigmoid(-prediction)
    loss   = -logp * mask - (log1mp) * (1-mask)

    return jnp.mean(loss)


def iou_loss(prediction, mask):
    """corresponds to -log(iou_score) in the CALFIN code"""
    eps = 1e-1
    prediction = jax.nn.sigmoid(prediction[..., 0])

    intersection = jnp.sum(prediction * mask, axis=[0, 1])
    union        = jnp.sum(jnp.maximum(prediction, mask), axis=[0, 1])
    score        = (intersection + eps) / (union + eps)
    return -jnp.log(score)


def calfin_loss(prediction, mask):
    edge = hk.max_pool(mask, [5, 5], [1, 1], "SAME") != min_pool(mask, [5, 5], [1, 1], "SAME")

    seg_loss  = jnp.mean(bce(prediction[..., :1], mask)) + \
                jnp.mean(iou_loss(prediction[..., :1], mask))
    edge_loss = jnp.mean(bce(prediction[..., 1:], edge)) + \
                jnp.mean(iou_loss(prediction[..., 1:], edge))

    return (1/26) * seg_loss + (25/26) * edge_loss


class AbstractDTW(ABC):
    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth
        
    @abstractmethod
    def minimum(self, *args):
        pass

    def __call__(self, prediction, snake):
        return self.dtw(prediction, snake)

    def dtw(self, prediction, snake):
        D = distance_matrix(prediction, snake)
        # wlog: H >= W
        if D.shape[0] < D.shape[1]:
            D = D.T
        H, W = D.shape
        
        if self.bandwidth is not None:
            i, j = jnp.mgrid[0:H, 0:W]
            D = jnp.where(jnp.abs(i - j) > self.bandwidth,
                jnp.inf,
                D
            )
        
        y, x = jnp.mgrid[0:W+H-1, 0:H]
        indices = y - x
        model_matrix = jnp.where((indices < 0) | (indices >= W),
            jnp.inf,
            jnp.take_along_axis(D.T, indices, axis=0)
        )

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

        carry, ys = jax.lax.scan(scan_step, init, model_matrix[2:], unroll=4)
        return carry[1][-1]


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

    def __init__(self, gamma=1.0, bandwidth=None):
        super().__init__(bandwidth)
        assert gamma > 0, "Gamma needs to be positive."
        self.gamma = gamma
        self.__name__ = f'SoftDTW({self.gamma})'
        self.minimum_impl = make_softmin(gamma)

    def minimum(self, args):
        return self.minimum_impl(args)


def forward_mae(prediction, snake):
    squared_dist = squared_distance_points_to_best_segment(prediction, snake)
    return jnp.mean(jnp.sqrt(squared_dist))


def backward_mae(prediction, snake):
    squared_dist = squared_distance_points_to_best_segment(snake, prediction)
    return jnp.mean(jnp.sqrt(squared_dist))


def forward_rmse(prediction, snake):
    squared_dist = squared_distance_points_to_best_segment(prediction, snake)
    return jnp.sqrt(jnp.mean(squared_dist))


def backward_rmse(prediction, snake):
    squared_dist = squared_distance_points_to_best_segment(snake, prediction)
    return jnp.sqrt(jnp.mean(squared_dist))
