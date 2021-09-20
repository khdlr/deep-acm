import jax
import jax.numpy as jnp
import numpy as np
import wandb
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from argparse import ArgumentParser
from einops import rearrange


def draw_snake(draw, snake, dashed=False, **kwargs):
    if dashed:
        for (y0, x0, y1, x1) in snake.reshape((-1, 4)):
            draw.line((x0, y0, x1, y1), **kwargs)
    else:
        for (y0, x0), (y1, x1) in zip(snake, snake[1:]):
            draw.line((x0, y0, x1, y1), **kwargs)


def log_gan_prediction(img, truth, tag, step):
    H, W, C = img.shape

    img = np.asarray(jax.image.resize(img, (512, 512, C), method='linear'))
    RGB = [2, 1, 0]
    if C == 1:
        RGB = [0, 0, 0]
    img = img[:, :, RGB]
    img = np.clip(255 * img[:,:, RGB], 0, 255).astype(np.uint8)

    H, W, C = img.shape
    truth = 0.5 * H * (1 + truth)

    img = Image.fromarray(img, mode='RGB')

    draw = ImageDraw.Draw(img)
    draw_snake(draw, truth, fill=(255, 0, 0), width=3)

    img = np.asarray(img).astype(np.float32) / 255
    wandb.log({tag: wandb.Image(img)}, step=step)


def log_image(img, truth, preds, tag, step):
    H, W, C = img.shape

    img = np.asarray(jax.image.resize(img, (512, 512, C), method='linear'))
    RGB = [2, 1, 0]
    if C == 1:
        RGB = [0, 0, 0]
    img = img[:, :, RGB]
    img = (255 * img[:,:, RGB]).astype(np.uint8)

    H, W, C = img.shape
    truth = 0.5 * H * (1 + truth)
    preds = [0.5 * H * (1 + p) for p in preds]

    img = Image.fromarray(img, mode='RGB')

    draw = ImageDraw.Draw(img)
    draw_snake(draw, truth, fill=(255, 0, 0), width=3)

    for i, snake in enumerate(preds, 1):
        kwargs = dict(fill=(0, 255, 0))
        if i == len(preds):
            kwargs['width'] = 3
        draw_snake(draw, snake, **kwargs)

    img = np.asarray(img).astype(np.float32) / 255
    wandb.log({tag: wandb.Image(img)}, step=step)


def log_video(img, truth, preds, tag, step):
    H, W, C = img.shape
    img = np.asarray(jax.image.resize(img, (256, 256, C), method='linear'))
    RGB = [2, 1, 0]
    if C == 1:
        RGB = [0, 0, 0]
    img = img[:, :, RGB]
    img = (255 * img[:,:, RGB]).astype(np.uint8)

    H, W, C = img.shape
    truth = 0.5 * H * (1 + truth)
    preds = [0.5 * H * (1 + p) for p in preds]

    lerped_preds = []
    t = jnp.linspace(0, 1, 20).reshape(-1, 1, 1)
    for pred0, pred1 in zip(preds, preds[1:]):
        if pred0.shape != pred1.shape:
            pred0 = jax.image.resize(pred0, pred1.shape, 'linear')
        lerped_preds += list( (1-t) * pred0 + t * pred1 )

    base = Image.fromarray(img, mode='RGB')
    draw = ImageDraw.Draw(base)
    draw_snake(draw, truth, fill=(255, 0, 0), width=2)
    frames = []
    for pred in lerped_preds:
        frame = base.copy()
        draw = ImageDraw.Draw(frame)
        draw_snake(draw, pred, fill=(0, 255, 0), width=2)
        frames.append(np.asarray(frame).astype(np.uint8))

    # Freeze-frame the last step
    for i in range(20):
        frames.append(frames[-1])

    frames = jnp.stack(frames)
    frames = rearrange(frames, 't h w c -> t c h w')
    frames = np.asarray(frames)

    wandb.log({tag: wandb.Video(frames, fps=20, format='gif')}, step=step)
