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


def log_image(img, truth, preds, init, tag, step):
    H, W, C = img.shape
    truth = 0.5 * H * (1 + truth)
    init  = 0.5 * H * (1 + init)
    preds = 0.5 * H * (1 + preds)

    RGB = [0, 0, 0]

    img = (255 * img[:,:,RGB]).astype(np.uint8)
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
