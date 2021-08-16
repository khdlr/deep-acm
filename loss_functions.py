import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from functools import reduce
from einops import rearrange

def l2_loss(predictions, ground_truth):
    loss = jnp.sum(jnp.square(predictions - ground_truth), axis=-1)
    loss = jnp.mean(loss)
    return loss


def l1_loss(predictions, ground_truth):
    loss = jnp.sum(jnp.abs(predictions - ground_truth), axis=-1)
    loss = jnp.mean(loss)
    return loss


def min_min_loss(predictions, ground_truth):
    predictions  = rearrange(predictions,  'batch (true pred) d -> batch true pred d', true=1, d=2)
    ground_truth = rearrange(ground_truth, 'batch (true pred) d -> batch true pred d', pred=1, d=2)
    DMatrix = jnp.sum(jnp.square(predictions - ground_truth), axis=-1)
    min1 = DMatrix.min(axis=1)
    min2 = DMatrix.min(axis=2)
    min_min = 0.5 * (jnp.mean(min1) + jnp.mean(min2))
    return min_min


def dtw_loss(predictions, ground_truth):
    predictions  = rearrange(predictions,  'batch (true pred) d -> batch true pred d', true=1, d=2)
    ground_truth = rearrange(ground_truth, 'batch (true pred) d -> batch true pred d', pred=1, d=2)
    D = jnp.sum(jnp.square(predictions - ground_truth), axis=-1)



class AbstractDTW(ABC):
    @abstractmethod
    def minimum(self, *args):
        pass

    def __call__(self, predictions, ground_truth):
        return jnp.mean(self.dtw(predictions, ground_truth))

    @jax.partial(jax.vmap, in_axes=[None, 0, 0])
    def dtw(self, prediction, ground_truth):
        prediction  = rearrange(prediction,   '(true pred) d -> true pred d', true=1, d=2)
        ground_truth = rearrange(ground_truth, '(true pred) d -> true pred d', pred=1, d=2)
        D = jnp.sum(jnp.square(prediction - ground_truth), axis=-1)

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

            minimum = self.minimum(diag, down, right)
            cost = minimum + antidiagonals[i]

            running_cost.append(cost)

        return running_cost[-1]

class DTW(AbstractDTW):
    __name__ = 'DTW'

    def minimum(self, *args):
        return reduce(jnp.minimum, args)
