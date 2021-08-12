import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from generate_data import generate_image 
from functools import partial

from models.deepsnake import DeepSnake, SimpleModel
from typing import NamedTuple, Mapping

import wandb
from tqdm import trange

from plotting import log_image

BATCH_SIZE = 4


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState


def loss_fn(predictions, ground_truth):
    loss_fwd = jnp.sum(jnp.square(predictions - ground_truth), axis=-1)
    loss_bwd = jnp.sum(jnp.square(predictions[:, ::-1] - ground_truth), axis=-1)

    loss = jnp.mean(jnp.minimum(loss_fwd, loss_bwd))

    return loss


def calculate_loss(params, net, imagery, contours):
    init_contours = jnp.roll(contours, 1, 0)
    predictions = net.apply(params, imagery, init_contours)
    loss = loss_fn(predictions, contours)
    return loss


def get_optimizer():
    return optax.adam(1e-3)


def make_batch(key):
    data_keys = jax.random.split(key, BATCH_SIZE)
    return jax.vmap(generate_image)(data_keys)


@jax.partial(jax.jit, static_argnums=2)
def train_step(state, key, net):
    imagery, contours = make_batch(key)
    _, optimizer = get_optimizer()
    calculate_loss(state.params, net, imagery, contours)

    loss_closure = partial(calculate_loss, net=net, imagery=imagery, contours=contours)
    loss, gradients = jax.value_and_grad(loss_closure)(state.params)
    updates, new_opt_state = optimizer(gradients, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)
    return loss, TrainingState(
        params=new_params,
        opt_state=new_opt_state
    )


@jax.partial(jax.jit, static_argnums=2)
def val_step(state, key, net):
    imagery, contours = make_batch(key)
    init_contours = jnp.roll(contours, 1, 0)
    predictions = net.apply(state.params, imagery, init_contours)
    loss = loss_fn(predictions, contours)
    return loss, imagery, contours, predictions, init_contours


def forward_function(imagery, init):
    model = DeepSnake()
    return model(imagery, init)


def main():
    train_key = jax.random.PRNGKey(42)
    persistent_val_key = jax.random.PRNGKey(27)

    net = hk.without_apply_rng(hk.transform(forward_function))

    opt_init, _ = get_optimizer()
    params = net.init(jax.random.PRNGKey(0), *make_batch(jax.random.PRNGKey(0)))
    opt_state = opt_init(params)
    state = TrainingState(params=params, opt_state=opt_state)
    wandb.init(project='Deep Snake Pre-Train')

    for epoch in range(100):
        prog = trange(10000)
        losses = []
        for step in prog:
            train_key, subkey = jax.random.split(train_key)
            loss, state = train_step(state, subkey, net)
            losses.append(loss)
            if step % 100 == 0:
                prog.set_description(f'{np.mean(losses):.3f}')
        wandb.log({f'trn/loss': np.mean(losses)}, step=epoch)
        # Validate
        val_key = persistent_val_key
        losses = []
        for step in range(1):
            val_key, subkey = jax.random.split(val_key)
            loss, *inspection = val_step(state, val_key, net)
            losses.append(loss)
            for i in range(inspection[0].shape[0]):
                print('logging img')
                log_image(*[np.asarray(ary[i]) for ary in inspection], "", epoch)
        wandb.log({f'val/loss': np.mean(losses)}, step=epoch)


if __name__ == '__main__':
    main()
