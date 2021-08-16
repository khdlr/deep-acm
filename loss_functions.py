import jax
import jax.numpy as jnp

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


