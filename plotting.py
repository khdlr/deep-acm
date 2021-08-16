import jax
import jax.numpy as jnp
import numpy as np
import wandb
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from einops import rearrange


def draw_snake(draw, snake, dashed=False, **kwargs):
    if dashed:
        for (y0, x0, y1, x1) in snake.reshape((-1, 4)):
            draw.line((x0, y0, x1, y1), **kwargs)
    else:
        for (y0, x0), (y1, x1) in zip(snake, snake[1:]):
            draw.line((x0, y0, x1, y1), **kwargs)


def log_image(img, truth, preds, init, tag, step):
    img = np.asarray(jax.image.resize(img, (512, 512, 1), method='linear'))
    RGB = [0, 0, 0]
    img = img[:, :, RGB]
    img = (255 * img[:,:,RGB]).astype(np.uint8)

    H, W, C = img.shape
    truth = 0.5 * H * (1 + truth)
    init  = 0.5 * H * (1 + init)
    preds = 0.5 * H * (1 + preds)

    img = Image.fromarray(img, mode='RGB')

    draw = ImageDraw.Draw(img)
    draw_snake(draw, init, fill=(0, 0, 255))
    draw_snake(draw, truth, fill=(255, 0, 0), width=3)

    for i, snake in enumerate(preds, 1):
        kwargs = dict(fill=(0, 255, 0))
        if i == preds.shape[0]:
            kwargs['width'] = 3
        draw_snake(draw, snake, **kwargs)

    img = np.asarray(img).astype(np.float32) / 255
    wandb.log({tag: wandb.Image(img)}, step=step)


def log_video(img, truth, preds, init, tag, step):
    img = np.asarray(jax.image.resize(img, (256, 256, 1), method='linear'))
    RGB = [0, 0, 0]
    img = img[:, :, RGB]
    img = (255 * img[:,:,RGB]).astype(np.uint8)

    H, W, C = img.shape
    truth = 0.5 * H * (1 + truth)
    init  = 0.5 * H * (1 + init)
    preds = 0.5 * H * (1 + preds)

    lerped_preds = []
    t = jnp.linspace(0, 1, 30).reshape(-1, 1, 1)
    for pred0, pred1 in zip(preds, preds[1:]):
        lerped_preds.append( (1-t) * pred0 + t * pred1 )
    lerped_preds = jnp.concatenate(lerped_preds, axis=0)

    frames = []
    for pred in lerped_preds:
        frame = Image.fromarray(img, mode='RGB')
        draw = ImageDraw.Draw(frame)
        draw_snake(draw, pred, fill=(0, 255, 0), width=2)
        frames.append(np.asarray(frame).astype(np.uint8))

    # Freeze-frame the last step
    for i in range(30):
        frames.append(frames[-1])

    frames = jnp.stack(frames)
    frames = rearrange(frames, 't h w c -> t c h w')
    frames = np.asarray(frames)

    wandb.log({tag: wandb.Video(frames, fps=30)}, step=step)
