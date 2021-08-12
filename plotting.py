import numpy as np
import wandb
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def draw_snake(draw, snake, dashed=False, **kwargs):
    if dashed:
        for (y0, x0, y1, x1) in snake.reshape((-1, 4)):
            draw.line((x0, y0, x1, y1), **kwargs)
    else:
        for (y0, x0), (y1, x1) in zip(snake, snake[1:]):
            draw.line((x0, y0, x1, y1), **kwargs)


def log_image(img, snake, pred, init, tag, step):
    H, W, C = img.shape
    snake   = 0.5 * H * (1 + snake)
    init    = 0.5 * H * (1 + init)
    pred    = 0.5 * H * (1 + pred)

    if C == 1:
        RGB = [0, 0, 0]
    elif C == 3:
        RGB = [0, 1, 2]
    else:
        RGB = [3, 2, 1]

    img = (255 * img[:,:,RGB]).astype(np.uint8)
    img = Image.fromarray(img, mode='RGB')
    draw = ImageDraw.Draw(img)

    draw_snake(draw, init, fill=(0, 0, 255))
    draw_snake(draw, snake, fill=(255, 0, 0), width=3)

    kwargs = dict(fill=(0, 255, 0), width=3)
    draw_snake(draw, pred, **kwargs)

    img = np.asarray(img).astype(np.float32) / 255
    wandb.log({tag: wandb.Image(img)}, step=step)
