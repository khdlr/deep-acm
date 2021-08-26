# Tried the following for more determinism. But it doesn't seem to help...
# import os
# os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_reductions'
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
import yaml
from typing import NamedTuple, Mapping
import pickle
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from generate_data import generate_image 

import wandb
from tqdm import trange, tqdm

import models
import loss_functions
from plotting import log_image, log_video


BATCH_SIZE = 16
METRICS = dict(
    l1 = loss_functions.l1_loss,
    l2 = loss_functions.l2_loss,
    min_min = loss_functions.min_min_loss,
    dtw = loss_functions.DTW(),
    forward_mae = loss_functions.forward_mae,
    backward_mae = loss_functions.backward_mae,
    forward_rmse = loss_functions.forward_rmse,
    backward_rmse = loss_functions.backward_rmse,
)
PATIENCE = 25


class TrainingState(NamedTuple):
    params: hk.Params
    model_state: hk.State
    opt_state: optax.OptState
    epoch: jnp.ndarray


def calculate_loss(params, model_state, net, imagery, contours, is_training):
    init_contours = jnp.roll(contours, 1, 0)
    predictions, model_state = net.apply(params, model_state, imagery, init_contours, is_training)
    loss = jnp.mean(jax.vmap(loss_fn)(predictions, contours))
    return loss, model_state


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

    loss_closure = jax.partial(calculate_loss,
            model_state=state.model_state, net=net,
            imagery=imagery, contours=contours, is_training=True)
    (loss, new_model_state), gradients = jax.value_and_grad(loss_closure, has_aux=True)(state.params)
    updates, new_opt_state = optimizer(gradients, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)
    return loss, TrainingState(
        params=new_params,
        model_state=new_model_state,
        opt_state=new_opt_state,
        epoch=state.epoch
    )


@jax.partial(jax.jit, static_argnums=2)
def val_step(state, key, net):
    imagery, contours = make_batch(key)
    init_contours = jnp.roll(contours, 1, 0)
    predictions, new_state = net.apply(state.params, state.model_state, imagery, init_contours, is_training=False)
    metrics = {}
    pred = predictions[-1]
    if pred.shape != contours.shape:
        pred = jax.image.resize(pred, contours.shape, 'linear')
    metrics['loss'] = jnp.mean(jax.vmap(loss_fn)(pred, contours))
    for m in METRICS:
        metrics[m] = jnp.mean(jax.vmap(METRICS[m])(pred, contours))

    return metrics, imagery, contours, predictions, init_contours


def save_state(state, out_path):
    state = jax.device_get(state)
    with out_path.open('wb') as f:
        pickle.dump(state, f)


def set_epoch(state, epoch):
    return TrainingState(
        params=state.params,
        model_state=state.model_state,
        opt_state=state.opt_state,
        epoch=epoch
    )


if __name__ == '__main__':
    train_key = jax.random.PRNGKey(42)
    persistent_val_key = jax.random.PRNGKey(27)
    multiplier = 64

    config = yaml.load(open('pretrain_config.yml'), Loader=yaml.SafeLoader)
    lf = config['loss_function']
    if lf.startswith('SoftDTW'):
        gamma = float(lf[8:-1])
        loss_fn = loss_functions.SoftDTW(gamma=gamma)
    elif lf == 'DTW':
        loss_fn = loss_functions.DTW()
    else:
        loss_fn = getattr(loss_functions, lf)

    modelclass = getattr(models, config['model'])
    net = modelclass(backbone=config['backbone'], **config['head'])
    net = hk.without_apply_rng(hk.transform_with_state(net))

    opt_init, _ = get_optimizer()
    params, model_state = net.init(jax.random.PRNGKey(0), *make_batch(jax.random.PRNGKey(0)), is_training=True)
    opt_state = opt_init(params)
    state = TrainingState(params=params, model_state=model_state, opt_state=opt_state, epoch=jnp.array(0))

    name = 'snake_head/conv1_d'

    running_min = np.inf
    last_improvement = 0

    wandb.init(project='Deep Snake Pre-Train', config=config)

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

        wandb.log({f'trn/loss': jnp.mean(loss_ary)}, step=epoch)
        # Validate
        val_key = persistent_val_key
        metrics = {m: [] for m in METRICS}
        metrics['loss'] = []
        imgdata = []
        viddata = []
        for step in trange(64, desc='Validation stuff'):
            val_key, subkey = jax.random.split(val_key)
            current_metrics, *inspection = val_step(state, val_key, net)
            for m in current_metrics:
                metrics[m].append(current_metrics[m])

            if step % 4 == 0:
                imagery, contours, predictions, init_contours = inspection
                predictions = [p[0] for p in predictions]
                imgdata.append((imagery[0], contours[0], predictions, init_contours[0], f"Imgs/Val{step}", epoch))
                viddata.append((imagery[0], contours[0], predictions, init_contours[0], f"Anim/Val{step}", epoch))
        metrics = {m: np.mean(metrics[m]) for m in metrics}
        metric = metrics['loss']
        if metric < running_min:
            last_improvement = epoch
            running_min = metric
            save_state(state, Path('checkpoints') / f'{wandb.run.id}-best.npz')
            for img, vid in tqdm(zip(imgdata, viddata), desc='Logging Media'):
                log_image(*img)
                log_video(*vid)
        if epoch - last_improvement > PATIENCE:
            print(f'Stopping early because there was no improvement for {PATIENCE} epochs.')
            break
        metrics['best_loss'] = running_min

        wandb.log({f'val/{m}': metrics[m] for m in metrics}, step=epoch)
