import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from abc import ABC, abstractmethod

from utils import pad_inf, fmt, distance_matrix


def l2_loss(predictions, ground_truth):
    loss = jnp.sum(jnp.square(predictions - ground_truth), axis=-1)
    loss = jnp.mean(loss)
    return loss


def l1_loss(predictions, ground_truth):
    loss = jnp.sum(jnp.abs(predictions - ground_truth), axis=-1)
    loss = jnp.mean(loss)
    return loss


def min_min_loss(predictions, ground_truth):
    D = jax.vmap(distance_matrix)(predictions, ground_truth)
    min1 = D.min(axis=1)
    min2 = D.min(axis=2)
    min_min = 0.5 * (jnp.mean(min1) + jnp.mean(min2))
    return min_min


class AbstractExplicitDTW(ABC):
    @abstractmethod
    def minimum(self, *args):
        pass

    def __call__(self, predictions, ground_truth):
        dtw_fun = jax.vmap(self.dtw, in_axes=[0, 0])
        return jnp.mean(dtw_fun(predictions, ground_truth))

    def dtw(self, prediction, ground_truth):
        D = distance_matrix(prediction, ground_truth)
        fmt(D)
        H, W = D.shape
        antidiagonals = [jnp.diag(D[::-1], i) for i in range(-H+1, W)]
        running_cost = [antidiagonals[0], antidiagonals[1] + antidiagonals[0]]

        for i in range(2, H+W-1):    
            ad = antidiagonals[i]
            N = len(ad)
            R2 = len(running_cost[-2])
            R1 = len(running_cost[-1])
            diag  = running_cost[-2]    
            down  = running_cost[-1]
            right = running_cost[-1]

            d_diag = R2 - N
            d_other = R1 - N

            if d_diag == -2:
                diag = jnp.pad(diag, (1, 1), constant_values=jnp.inf)
            elif d_diag == -1:
                if H > W:
                    diag = jnp.pad(diag, (1, 0), constant_values=jnp.inf)
                elif H < W:
                    diag = jnp.pad(diag, (0, 1), constant_values=jnp.inf)
                else:
                    raise ValueError("d_diag = {d_diag} and H == W! This is fishy...")
            elif d_diag == 0:
                if H > W:
                    diag = jnp.pad(diag[:-1], (1, 0), constant_values=jnp.inf)
                elif H < W:
                    diag = jnp.pad(diag[1:], (0, 1), constant_values=jnp.inf)
                else:
                    pass
            elif d_diag == 1:
                if H > W:
                    diag = diag[:-1]
                elif H < W:
                    diag = diag[1:]
                else:
                    raise ValueError("d_diag = {d_diag} and H == W! This is fishy...")

            elif d_diag == 2:
                diag = diag[1:-1]
            else:
                raise NotImplementedError(f"d_diag = {d_diag} not yet implemented!")

            if d_other == -1:
                down  = jnp.pad(down,  (0, 1), constant_values=jnp.inf)
                right = jnp.pad(right, (1, 0), constant_values=jnp.inf)
            elif d_other == 0:
                if H > W:
                    down = jnp.pad(down[1:], (0, 1), constant_values=jnp.inf)
                elif H < W:
                    right = jnp.pad(right[:-1], (1, 0), constant_values=jnp.inf)
            elif d_other == 1:
                down = down[1:]
                right = right[:-1]
            else:
                raise NotImplementedError(f"d_other = {d_other} not yet implemented!")

            print(f'Step {i}')
            fmt('diag ', diag)
            fmt('down ', down)
            fmt('right', down)
            minimum = self.minimum(diag, down, right)
            cost = minimum + antidiagonals[i]

            running_cost.append(cost)

        return running_cost[-1]


class AbstractDTW(ABC):
    @abstractmethod
    def minimum(self, *args):
        pass

    def __call__(self, predictions, ground_truth):
        dtw_fun = jax.vmap(self.dtw, in_axes=[0, 0])
        return jnp.mean(dtw_fun(predictions, ground_truth))

    def dtw(self, prediction, ground_truth):
        D = distance_matrix(prediction, ground_truth)
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


class MixedDTW(AbstractDTW):
    """
    A mix of DTW and SoftDTW to circumvent numerical issues.
    """
    __name__ = 'SoftDTW'

    def __init__(self, softness=1.0):
        assert gamma > 0, "Gamma needs to be positive."
        self.softness = softness
        self.__name__ = f'SoftDTW({self.gamma})'

    def minimum(self, args, axis):
        softmin = -logsumexp(-args, axis=axis)
        min = jnp.min(args, axis=axis)
        return (1 - self.softness) * min + self.softness * softmin
