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
sys.path.append('augmax')
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


def prep(batch, key=None, augment=False):
    ops = []
    if augment: ops += [
        augmax.HorizontalFlip(),
        augmax.VerticalFlip(),
        augmax.Rotate90(),
        augmax.Rotate(15),
        augmax.Warp(coarseness=16)
    ]
    ops += [augmax.ByteToFloat()]
    if augment: ops += [
        augmax.ChannelShuffle(p=0.1),
        augmax.Solarization(p=0.1),
    ]

    input_types =  [
        augmax.InputType.IMAGE,
        augmax.InputType.MASK,
        augmax.InputType.CONTOUR,
    ]
    chain = augmax.Chain(*ops, input_types=input_types)
    if augment == False:
        key = jax.random.PRNGKey(0)
    subkeys = jax.random.split(key, batch[0].shape[0])
    transformation = jax.vmap(chain)
    image, mask, contours = transformation(subkeys, *batch)
    contours = 2 * (contours / image.shape[1]) - 1.0
    return image, mask, contours


def make_fake_batch(key, size, batch_size):
    data_keys = jax.random.split(key, batch_size)
    return jax.vmap(generate_image, in_axes=[0, None])(data_keys, size)


def update_net(net, state, loss_closure):
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
        loss, _ = nets.D(states.D.params, states.D.buffers, fake, is_training=False)
        metrics = {'Generator': loss}
        return jnp.mean(loss), (state, output, metrics)
    G_new, G_output, G_metrics = update_net(nets.G, states.G, calculate_G_loss)

    # Update Discriminator
    fake = jnp.concatenate([G_output, fake_mask], axis=-1)
    true = jnp.concatenate([imagery, mask], axis=-1)
    def calculate_D_loss(params):
        def img_to_output(img):
            x, _ = nets.D(params, states.D.buffers, img, is_training=True)
            return jnp.sum(jnp.mean(x, axis=-1))

        t = jax.random.uniform(key3, [fake.shape[0], 1, 1, 1])
        interp = t * fake + (1-t) * true

        grad = jax.grad(img_to_output)(interp)
        gradnorm = jnp.sum(jnp.square(grad), axis=[1,2,3])
        gradient_penalty = jnp.mean(jnp.square(gradnorm - 1))

        state = states.D.buffers
        out_true,  state = nets.D(params, state, true, is_training=True)
        out_false, state = nets.D(params, state, fake, is_training=True)

        loss = jnp.mean(out_true) - jnp.mean(out_false)

        metrics = {
            'margin': -loss,
            '∇_penalty': gradient_penalty
        }

        return loss + 10 * gradient_penalty, (state, None, metrics)
    D_new, D_output, D_metrics = update_net(nets.D, states.D, calculate_D_loss)

    # Update Snake
    def calculate_S_loss(params):
        snake_pred, state = nets.S(params, states.S.buffers, G_output, is_training=True)
        loss = jnp.mean(jax.vmap(loss_fn)(snake_pred, fake_poly))
        metrics = {
            'snake_loss': loss,
        }
        for m in METRICS:
            metrics[m] = jnp.mean(jax.vmap(METRICS[m])(snake_pred, fake_poly))
        return loss, (state, snake_pred, metrics)
    S_new, S_output, S_metrics = update_net(nets.S, states.S, calculate_S_loss)

    metrics = {**G_metrics, **D_metrics, **S_metrics}

    return metrics, Nets(G=G_new, D=D_new, S=S_new) 


@jax.partial(jax.jit, static_argnums=0)
def val_step(nets, states, batch, key):
    imagery, mask, contours = prep(batch)
    predictions, _ = nets.S(states.S.params, states.S.buffers, imagery, is_training=False)
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
    output, _ = nets.G(states.G.params, states.G.buffers, input_noise, fake_mask, is_training=False)
    return output, fake_mask, fake_poly


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

    modelclass = getattr(models, config['model'])
    S = modelclass(backbone=config['backbone'], **config['head'])
    S = hk.without_apply_rng(hk.transform_with_state(S))

    # initialize data loading
    train_key, subkey = jax.random.split(train_key)
    train_loader = get_loader(config['batch_size'], 1, 'train', subkey)
    val_loader   = get_loader(16, 1, 'val', None)
    img, mask, snake = prep(next(iter(train_loader)))

    # Initialize model and optimizer state
    opt_init, _ = get_optimizer()
    params, buffers = S.init(jax.random.PRNGKey(39), img, is_training=True)
    S_state = TrainingState(params=params, buffers=buffers, opt=opt_init(params))

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

    for epoch in range(1, 201):
        wandb.log({f'epoch': epoch}, step=epoch)
        prog = tqdm(train_loader, desc=f'Ep {epoch} Trn')
        trn_metrics = {}
        loss_ary = None
        for step, batch in enumerate(prog, 1):
            # if epoch == 1 and step == 100:
            #     jax.profiler.start_trace("tensorboard/train")
            train_key, subkey = jax.random.split(train_key)
            metrics, states = train_step(nets, states, batch, subkey)
            for m in metrics:
              if m not in trn_metrics: trn_metrics[m] = []
              trn_metrics[m].append(metrics[m])
            # if epoch == 1 and step == 105:
            #     for m in metrics:
            #         metrics[m].block_until_ready()
            #     jax.profiler.stop_trace()
            if step % 100 == 0 or step == len(prog):
                losses = []
                prog.set_description(f'{np.mean(trn_metrics["snake_loss"]):.3f}')

        # Save Checkpoint
        save_state(states, Path('checkpoints') / f'{wandb.run.id}-latest.npz')
        log_metrics(trn_metrics, 'trn/', epoch)

        # Validate
        val_key = persistent_val_key
        val_metrics = {}
        imgdata = []
        viddata = []
        for step, batch in enumerate(tqdm(val_loader, desc=f'Ep {epoch} Val')):
            val_key, subkey = jax.random.split(val_key)
            metrics, *inspection = val_step(nets, states, batch, val_key)
            for m in metrics:
              if m not in val_metrics: val_metrics[m] = []
              val_metrics[m].append(metrics[m])

            imagery, contours, predictions = inspection
            predictions = [p[0] for p in predictions]
            imgdata.append((imagery[0], contours[0], predictions, f"Imgs/Val{step}", epoch))
            viddata.append((imagery[0], contours[0], predictions, f"Anim/Val{step}", epoch))

        log_metrics(val_metrics, 'val/', epoch)
        for img in imgdata:
            log_image(*img)
        for vid in tqdm(viddata, desc='Logging Videos'):
            log_video(*vid)

        imgs, _, contours = gan_val_step(nets, states, persistent_val_key)
        for i, (img, contour) in enumerate(zip(imgs, contours)):
            log_gan_prediction(img, contour, f'GAN/gen{i:02d}', epoch)
