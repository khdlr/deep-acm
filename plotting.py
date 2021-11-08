import jax
import jax.numpy as jnp
import numpy as np
import wandb
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from io import BytesIO
from PIL import Image
import numpy as np
import base64

from argparse import ArgumentParser
from einops import rearrange

RGB = [0, 1, 2]

def draw_snake(draw, snake, dashed=False, **kwargs):
    if dashed:
        for (y0, x0, y1, x1) in snake.reshape((-1, 4)):
            draw.line((x0, y0, x1, y1), **kwargs)
    else:
        for (y0, x0), (y1, x1) in zip(snake, snake[1:]):
            draw.line((x0, y0, x1, y1), **kwargs)


def log_image(img, truth, preds, tag, step):
    H, W, C = img.shape

    img = np.asarray(jax.image.resize(img, (512, 512, C), method='linear'))
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


def make_path_string(vertices):
    return 'M' + ' L'.join(f'{x:.2f},{y:.2f}' for y, x in vertices)


def animated_path(paths):
    pathvalues = ";".join(make_path_string(path) for path in paths)
    keytimes = ";".join(f'{x:.2f}' for x in np.linspace(0, 1, len(paths)))
    return f"""<path fill="none" stroke="rgb(255, 255, 0)" stroke-width="1">
          <animate attributeName="d" values="{pathvalues}" keyTimes="{keytimes}" dur="3s" repeatCount="indefinite" />
          </path>
          """


def log_anim(img, truth, preds, tag, step):
    H, W, C = img.shape

    img = img[:, :, RGB]
    img = (255 * img[:,:, RGB]).astype(np.uint8)
    img = Image.fromarray(np.asarray(img))
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    imgbase64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    truth = 0.5 * H * (1 + truth)
    gtpath = make_path_string(truth)

    path_html = ""
    for pred in preds:
        pred = [0.5 * H * (1 + p) for p in pred]
        pred = pred + [pred[-1], pred[-1]]
        path_html += animated_path(pred)

    html = f"""
    <!DOCTYPE html>
    <html>
    <meta charset = "UTF-8">
    <body>
      <svg xmlns="http://www.w3.org/2000/svg" height="100%" viewBox="0 0 256 256">
        <image href="data:image/jpeg;charset=utf-8;base64,{imgbase64}" width="256px" height="256px"/>
        <path fill="none" stroke="hsl(0, 99%, 56%)" stroke-width="2"
            d="{gtpath}" />
        </path>
        {path_html}
      </svg>
    </body>
    </html>
    """

    wandb.log({tag: wandb.Html(html, inject=False)}, step=step)
