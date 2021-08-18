import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from generate_data import generate_image 
from functools import partial

from typing import NamedTuple, Mapping
import wandb
from tqdm import trange
from pathlib import Path
import pickle

from models.deepsnake import DeepSnake
from loss_functions import l2_loss, min_min_loss, l1_loss, DTW, SoftDTW
from plotting import log_image, log_video


BATCH_SIZE = 16
METRICS = dict(
    l1 = l1_loss,
    l2 = l2_loss,
    min_min = min_min_loss,
    dtw = DTW()
)

def loss_fn(predictions, contours, epoch):
    return SoftDTW(gamma=0.01)(predictions, contours)


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState
    epoch: jnp.ndarray


def calculate_loss(params, net, imagery, contours, epoch):
    init_contours = jnp.roll(contours, 1, 0)
    predictions = net.apply(params, imagery, init_contours)
    loss = loss_fn(predictions, contours, epoch)
    return loss


def get_optimizer():
    optimizer = optax.chain(
      optax.clip_by_global_norm(0.25),
      optax.adam(1e-3, eps=1e-3)
    )
    return optimizer


def make_batch(key):
    data_keys = jax.random.split(key, BATCH_SIZE)
    return jax.vmap(generate_image)(data_keys)


@jax.partial(jax.jit, static_argnums=2)
def train_step(state, key, net):
    imagery, contours = make_batch(key)
    _, optimizer = get_optimizer()

    loss_closure = partial(calculate_loss, net=net, imagery=imagery, contours=contours, epoch=state.epoch)
    loss, gradients = jax.value_and_grad(loss_closure)(state.params)
    updates, new_opt_state = optimizer(gradients, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)
    return loss, TrainingState(
        params=new_params,
        opt_state=new_opt_state,
        epoch=state.epoch
    )


@jax.partial(jax.jit, static_argnums=2)
def val_step(state, key, net):
    imagery, contours = make_batch(key)
    init_contours = jnp.roll(contours, 1, 0)
    predictions = net.apply(state.params, imagery, init_contours)
    metrics = {}
    metrics['loss'] = loss_fn(predictions[:, -1], contours, state.epoch)
    for m in METRICS:
        metrics[m] = METRICS[m](predictions[:, -1], contours)

    return metrics, imagery, contours, predictions, init_contours


def save_state(state, out_path):
    state = jax.device_get(state)
    with out_path.open('wb') as f:
        pickle.dump(state, f)


def set_epoch(state, epoch):
    return TrainingState(
        params=state.params,
        opt_state=state.opt_state,
        epoch=epoch
    )


def main():
    train_key = jax.random.PRNGKey(42)
    persistent_val_key = jax.random.PRNGKey(27)
    multiplier = 64

    net = DeepSnake(multiplier, output_intermediates=False, iterations=5)
    val_net = DeepSnake(multiplier, output_intermediates=True, iterations=7)

    net     = hk.without_apply_rng(hk.transform(net))
    val_net = hk.without_apply_rng(hk.transform(val_net))

    opt_init, _ = get_optimizer()
    params = net.init(jax.random.PRNGKey(0), *make_batch(jax.random.PRNGKey(0)))
    opt_state = opt_init(params)
    state = TrainingState(params=params, opt_state=opt_state, epoch=jnp.array(0))
    wandb.init(project='Deep Snake Pre-Train')

    name = 'snake_head/conv1_d'
    p = params[name]['w']

    for epoch in range(1, 201):
        prog = trange(1, 10001)
        losses = []
        loss_ary = None
        state = set_epoch(state, epoch)
        for step in prog:
            train_key, subkey = jax.random.split(train_key)
            loss, state = train_step(state, subkey, net)
            losses.append(loss)
            if step % 100 == 0:
                if loss_ary is None:
                    loss_ary = jnp.stack(losses)
                else:
                    loss_ary = jnp.concatenate([loss_ary, jnp.stack(losses)])
                losses = []
                prog.set_description(f'{jnp.mean(loss_ary):.3f}')

        # Save Checkpoint
        save_state(state, Path('checkpoints') / f'{wandb.run.id}-latest.npz')
        if (epoch % 10 == 0):
            save_state(state, Path('checkpoints') / f'{wandb.run.id}-{epoch}.npz')

        wandb.log({f'trn/loss': jnp.mean(loss_ary)}, step=epoch)
        # Validate
        val_key = persistent_val_key
        metrics = {m: [] for m in METRICS}
        metrics['loss'] = []
        for step in range(3):
            val_key, subkey = jax.random.split(val_key)
            current_metrics, *inspection = val_step(state, val_key, val_net)
            for m in current_metrics:
                metrics[m].append(current_metrics[m])

            for i in trange(4, desc='Logging Val Images'):
                log_image(*[np.asarray(ary[i]) for ary in inspection], f"Val{step}-{i}", epoch)
                log_video(*[np.asarray(ary[i]) for ary in inspection], f"ValAnim{step}-{i}", epoch)
        wandb.log({f'val/{m}': np.mean(metrics[m]) for m in metrics}, step=epoch)


if __name__ == '__main__':
    main()
