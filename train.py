import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from generate_data import generate_image 
from functools import partial

from models.deepsnake import DeepSnake, SimpleModel
from typing import NamedTuple, Mapping

from tqdm import trange

BATCH_SIZE = 4


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState


def calculate_loss(params, net, imagery, contours):
    init_contours = jnp.roll(contours, 1, 0)
    predictions = net.apply(params, imagery, init_contours)
    loss = jnp.mean(jnp.sum(jnp.square(predictions - contours), axis=-1))
    return loss


def get_optimizer():
    return optax.adam(1e-3)


def make_batch(key):
    data_keys = jax.random.split(key, BATCH_SIZE)
    return jax.vmap(generate_image)(data_keys)


# @jax.jit
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

    for epoch in range(100):
        prog = trange(100)
        losses = []
        for step in prog:
            train_key, subkey = jax.random.split(train_key)
            loss, state = train_step(state, subkey, net)
            losses.append(loss)
            prog.set_description(f'{np.mean(losses):.3f}')
        # Validate
        val_key = persistent_val_key
        for step in range(1):
            val_key, subkey = jax.random.split(val_key)
            train_step(state, val_key, net)


if __name__ == '__main__':
    main()
