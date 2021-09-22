import yaml
from typing import NamedTuple, Any
import pickle
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax.profiler import TraceAnnotation
from jax.experimental.host_callback import id_print
import haiku as hk
import optax
from data_loading import get_loader 
from generate_data import generate_image 

import wandb
from tqdm import tqdm

import sys
import augmax

import models
import loss_functions
from plotting import log_image, log_video, log_gan_prediction


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
PATIENCE = 100


def get_optimizer():
    optimizer = optax.chain(
      optax.clip_by_global_norm(0.25),
      optax.adam(1e-3, b1=0.5, b2=0.9)
    )
    return optimizer


class TrainingState(NamedTuple):
    params: hk.Params
    buffers: hk.State
    opt: optax.OptState


class Nets(NamedTuple):
    G: Any
    D: Any
    S: Any


def changed_state(state, params=None, buffers=None, opt=None):
    return NewTrainingState(
        params = state.params if params is None else params,
        buffers = state.buffers if buffers is None else buffers,
        opt = state.opt if opt is None else opt,
    )


def prep(batch, key=None, augment=False, input_types=None):
    ops = []
    if augment: ops += [
        augmax.HorizontalFlip(),
        augmax.VerticalFlip(),
        augmax.Rotate90(),
        # augmax.Rotate(15),
        augmax.Warp(coarseness=16)
    ]
    ops += [augmax.ByteToFloat()]
    # if augment: ops += [
    #     augmax.ChannelShuffle(p=0.1),
    #     augmax.Solarization(p=0.1),
    # ]

    if input_types is None:
        input_types = [
            augmax.InputType.IMAGE,
            augmax.InputType.MASK,
            augmax.InputType.CONTOUR,
        ]
    chain = augmax.Chain(*ops, input_types=input_types)
    if augment == False:
        key = jax.random.PRNGKey(0)
    subkeys = jax.random.split(key, batch[0].shape[0])
    transformation = jax.vmap(chain)
    outputs = list(transformation(subkeys, *batch))
    for i, typ in enumerate(input_types):
        if typ == augmax.InputType.CONTOUR:
            outputs[i] = 2 * (outputs[i] / outputs[0].shape[1]) - 1.0

    return outputs


def make_fake_batch(key, size, batch_size):
    data_keys = jax.random.split(key, batch_size)
    return jax.vmap(generate_image, in_axes=[0, None])(data_keys, size)


def update_net(state, loss_closure):
    _, optimizer = get_optimizer()
    (loss, (buffers, output, metrics)), gradients = jax.value_and_grad(loss_closure, has_aux=True)(state.params)
    updates, opt = optimizer(gradients, state.opt, state.params)
    params = optax.apply_updates(state.params, updates)
    return TrainingState(params, buffers, opt), output, metrics


@jax.partial(jax.jit, static_argnums=0)
def train_step(nets, states, batch, key):
    key0, key1, key2, key3 = jax.random.split(key, 4)
    imagery, mask, contours = prep(batch, key0, augment=True)

    # Update Generator
    input_noise = jax.random.normal(key1, [config['batch_size'], 1024])
    fake_mask, fake_poly = make_fake_batch(key2, config['data_size'], config['batch_size'])

    def calculate_G_loss(params):
        output, state = nets.G(params, states.G.buffers, input_noise, fake_mask, is_training=True)
        fake = jnp.concatenate([output, fake_mask], axis=-1)
        D_out, _ = nets.D(states.D.params, states.D.buffers, fake, is_training=False)
        loss = -jnp.mean(jax.nn.log_sigmoid(D_out))

        metrics = {'Generator': loss}
        return loss, (state, output, metrics)
    G_new, fake_img, G_metrics = update_net(states.G, calculate_G_loss)

    # Update Discriminator
    # fake_img, fake_mask = prep((fake_img, fake_mask), key3, augment=True,
    #         input_types=[augmax.InputType.IMAGE, augmax.InputType.MASK])
    fake = jnp.concatenate([fake_img, fake_mask], axis=-1)
    true = jnp.concatenate([imagery, mask], axis=-1)
    def calculate_D_loss(params):
        state = states.D.buffers
        out_true,  state = nets.D(params, state, true, is_training=True)
        out_false, state = nets.D(params, state, fake, is_training=True)

        loss = - jnp.mean(jax.nn.log_sigmoid( out_true)) \
               - jnp.mean(jax.nn.log_sigmoid(-out_false))

        metrics = {
            'D_true': jnp.mean(jax.nn.sigmoid(out_true)),
            'D_false': jnp.mean(jax.nn.sigmoid(out_false)),
            'margin': jnp.mean(jax.nn.sigmoid(out_true) - jax.nn.sigmoid(out_false)),
        }

        return loss, (state, None, metrics)
    D_new, D_output, D_metrics = update_net(states.D, calculate_D_loss)

    metrics = {**G_metrics, **D_metrics}

    return metrics, Nets(G=G_new, D=D_new, S=states.S) 


@jax.partial(jax.jit, static_argnums=0)
def train_S_step(nets, states, key):
    key1, key2, key3 = jax.random.split(key, 3)
    input_noise = jax.random.normal(key1, [config['batch_size'], 1024])
    fake_mask, fake_poly = make_fake_batch(key2, config['data_size'], config['batch_size'])
    G_output, _ = nets.G(states.G.params, states.G.buffers, input_noise, fake_mask, is_training=False)

    # Update Snake
    def calculate_S_loss(params):
        snake_pred, state = nets.S(params, states.S.buffers, key3, G_output, is_training=True)
        loss = jnp.mean(jax.vmap(loss_fn)(snake_pred, fake_poly))
        metrics = {
            'snake_loss': loss,
        }
        for m in METRICS:
            metrics[m] = jnp.mean(jax.vmap(METRICS[m])(snake_pred, fake_poly))
        return loss, (state, snake_pred, metrics)
    S_new, _, metrics = update_net(states.S, calculate_S_loss)
    return metrics, Nets(G=states.G, D=states.D, S=S_new) 


@jax.partial(jax.jit, static_argnums=0)
def val_step(nets, states, batch, key):
    imagery, mask, contours = prep(batch)
    predictions, _ = nets.S(states.S.params, states.S.buffers, key, imagery, is_training=False)
    metrics = {}
    pred = predictions[-1]
    for m in METRICS:
        metrics[m] = jnp.mean(jax.vmap(METRICS[m])(pred, contours))

    return metrics, imagery, contours, predictions


@jax.partial(jax.jit, static_argnums=0)
def gan_val_step(nets, states, key):
    key1, key2 = jax.random.split(key, 2)
    fake_mask, fake_poly = make_fake_batch(key1, config['data_size'], 8)
    input_noise = jax.random.normal(key2, [8, 1024])
    fake_imagery, _ = nets.G(states.G.params, states.G.buffers, input_noise, fake_mask, is_training=False)
    predictions, _ = nets.S(states.S.params, states.S.buffers, key, fake_imagery, is_training=False)
    metrics = {}
    pred = predictions[-1]
    for m in METRICS:
        metrics[m] = jnp.mean(jax.vmap(METRICS[m])(pred, fake_poly))

    return metrics, fake_imagery, fake_poly, predictions


def save_state(state, out_path):
    state = jax.device_get(state)
    with out_path.open('wb') as f:
        pickle.dump(state, f)


def dict_merge(old, update):
    out = hk.data_structures.to_mutable_dict(old)
    for a in old:
        for b in old[a]:
            if old[a][b].shape == update[a][b].shape:
                out[a][b] = update[a][b]
            else:
                print(f"Shape mismatch for param {a}/{b}. Loaded {old.shape} but need {cur.shape}. Initializing randomly")
    return hk.data_structures.to_immutable_dict(out)


def split_params(params, split_mode):
    predicate = lambda m, n, p: True
    if split_mode == 'head':
        predicate = lambda m, n, p: not m.startswith('snake_head')
    elif split_mode == 'none':
        predicate = lambda m, n, p: True
    else:
        raise ValueError("TODO")

    return hk.data_structures.partition(predicate, params)


def log_metrics(metrics, prefix, epoch):
    metrics = {m: np.mean(metrics[m]) for m in metrics}
    for m in metrics:
        if 'rmse' in m or 'mae' in m:
            metrics[m] = metrics[m] / 2 * 512 * 30  # Convert to meters

    wandb.log({f'{prefix}{m}': metrics[m] for m in metrics}, step=epoch)
    print(f'{prefix}metrics')
    print(', '.join(f'{k}: {v:.3f}' for k, v in metrics.items()))


if __name__ == '__main__':
    train_key = jax.random.PRNGKey(42)
    persistent_val_key = jax.random.PRNGKey(27)
    multiplier = 64

    config = yaml.load(open('config.yml'), Loader=yaml.SafeLoader)
    lf = config['loss_function']
    if lf.startswith('SoftDTW'):
        gamma = float(lf[8:-1])
        loss_fn = loss_functions.SoftDTW(gamma=gamma)
    elif lf == 'DTW':
        loss_fn = loss_functions.DTW()
    else:
        loss_fn = getattr(loss_functions, lf)

    model_args = config['model_args']
    model_args['vertices'] = config['vertices']
    modelclass = getattr(models, config['model'])
    S = modelclass(**model_args)
    S = hk.transform_with_state(S)

    # initialize data loading
    train_key, subkey = jax.random.split(train_key)
    train_loader = get_loader(config['batch_size'], 1, 'train', subkey)
    val_loader   = get_loader(16, 1, 'val', None)
    img, mask, snake = prep(next(iter(train_loader)))

    # Initialize model and optimizer state
    opt_init, _ = get_optimizer()
    params, buffers = S.init(jax.random.PRNGKey(39), img, is_training=True)
    S_state = TrainingState(params=params, buffers=buffers, opt=opt_init(params))

    # from jax.tools.jax_to_hlo import jax_to_hlo
    # from jax.lib import xla_client
    # from jax.tree_util import tree_flatten, tree_leaves
    # from jax.api_util import flatten_fun
    # from jax.linear_util import wrap_init

    # partial = jax.partial(S.apply, params, buffers, jax.random.PRNGKey(0))
    # hlo = jax_to_hlo(partial, [
    #     ("img", xla_client.Shape(f"f32[{','.join(map(str, img.shape))}]"))
    # ])[1]

    # with open("hlo.txt", "w") as f:
    #     f.write(hlo)

    G = models.gan.Generator(output_channels=len(config['bands']))
    G = hk.without_apply_rng(hk.transform_with_state(G))
    params, buffers = G.init(jax.random.PRNGKey(129), jnp.zeros([config['batch_size'], 1024]), mask, True)
    G_state = TrainingState(params=params, buffers=buffers, opt=opt_init(params))

    D = models.gan.Discriminator()
    D = hk.without_apply_rng(hk.transform_with_state(D))
    B, H, W, C = img.shape
    params, buffers = D.init(jax.random.PRNGKey(891), jnp.zeros([B, H, W, C+1]), True)
    D_state = TrainingState(params=params, buffers=buffers, opt=opt_init(params))

    nets   = Nets(G.apply, D.apply, S.apply)
    states = Nets(G_state, D_state, S_state)

    running_min = np.inf
    last_improvement = 0
    wandb.init(project='Deep Snake', config=config)

    for epoch in range(1, 2001):
        wandb.log({f'epoch': epoch}, step=epoch)
        prog = tqdm(train_loader, desc=f'Ep {epoch} Trn')
        trn_metrics = {}
        loss_ary = None
        for step, batch in enumerate(prog, 1):
            train_key, subkey = jax.random.split(train_key)
            metrics, states = train_step(nets, states, batch, subkey)

            if epoch > 0:
                train_key, subkey = jax.random.split(train_key)
                metrics2, states = train_S_step(nets, states, subkey)
                metrics = {**metrics, **metrics2}

            for m in metrics:
              if m not in trn_metrics: trn_metrics[m] = []
              trn_metrics[m].append(metrics[m])
            if step % 10 == 0 or step == len(prog):
                losses = []
                prog.set_description(f'{np.mean(trn_metrics["margin"]):.3f}')

        # Save Checkpoint
        ckpt_dir = Path('checkpoints')
        ckpt_dir.mkdir(exist_ok=True)
        save_state(states, ckpt_dir / f'{wandb.run.id}-latest.npz')
        save_state(states, ckpt_dir / f'{wandb.run.id}-{epoch:04d}.npz')
        log_metrics(trn_metrics, 'trn/', epoch)

        # Validate GAN
        val_key = persistent_val_key
        gan_metrics = {}
        for step in tqdm(range(32), desc=f'Ep {epoch} GAN Val'):
            val_key, subkey = jax.random.split(val_key)
            metrics, *inspection = gan_val_step(nets, states, subkey)
            imagery, contours, predictions = inspection

            for m in metrics:
              if m not in gan_metrics: gan_metrics[m] = []
              gan_metrics[m].append(metrics[m])
            if step > 0: continue

            for i, (img, contour) in enumerate(zip(imagery, contours)):
                preds = [p[i] for p in predictions]
                if len(preds) > 1:
                    log_video(img, contour, preds, f'GAN_Anim/{i:02d}', epoch)
                log_image(img, contour, preds, f'GAN_Img/{i:02d}', epoch)

        # Validate
        val_key = persistent_val_key
        val_metrics = {}
        for step, batch in enumerate(tqdm(val_loader, desc=f'Ep {epoch} Val')):
            val_key, subkey = jax.random.split(val_key)
            metrics, *inspection = val_step(nets, states, batch, subkey)
            for m in metrics:
              if m not in val_metrics: val_metrics[m] = []
              val_metrics[m].append(metrics[m])

            imagery, contours, predictions = inspection
            predictions = [p[0] for p in predictions]
            log_image(imagery[0], contours[0], predictions, f"Imgs/Val{step}", epoch)
            if len(preds) > 1:
                log_video(imagery[0], contours[0], predictions, f"Anim/Val{step}", epoch)

        log_metrics(val_metrics, 'val/', epoch)

