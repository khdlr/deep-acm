import os
import hashlib

import torch.utils.data
import numpy as np
import jax
import jax.numpy as jnp
import rasterio as rio
from pathlib import Path
from skimage.measure import find_contours
import yaml


def md5(obj):
    obj = str(obj).encode('utf8')
    return hashlib.md5(obj).hexdigest()[:16]


def _isval(gt_path):
    year = int(gt_path.stem[:4])
    return year >= 2020


class DeterministicShuffle(torch.utils.data.Sampler):
    def __init__(self, length, rng_key, repetitions=10):
        self.rng_key = rng_key
        self.length = length
        self.repetitions = repetitions

    def __iter__(self):
        self.rng_key, *subkeys = jax.random.split(self.rng_key, self.repetitions+1)
        permutations = jnp.concatenate([
            jax.random.permutation(subkey, self.length) for subkey in subkeys
        ])
        return permutations.__iter__()

    def __len__(self):
        return self.length * self.repetitions


def numpy_collate(batch):
    """Collate tensors as numpy arrays, taken from
    https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html"""
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def get_loader(batch_size, data_threads, mode, rng_key):
    data = UC1SnakeDataset(mode=mode)

    kwargs = dict(
        batch_size = batch_size,
        num_workers = data_threads,
        collate_fn = numpy_collate
    )
    if mode == 'train':
        kwargs['sampler'] = DeterministicShuffle(len(data), rng_key)

    return torch.utils.data.DataLoader(data, **kwargs)


def snakify(gt, vertices):
    contours = find_contours(gt, 0.5)
    # Select the longest contour
    if len(contours) == 0:
        empty = 0.5 * jnp.ones([vertices, 2], np.float32)
        return empty

    contour = max(contours, key=lambda x: x.shape[0])
    contour = (2. * contour.astype(np.float32) / gt.shape[0] - 1.).astype(np.float32)
    contour = contour.view(np.complex64)[:, 0]
    C_space = np.linspace(0, 1, len(contour), dtype=np.float32)
    S_space = np.linspace(0, 1, vertices, dtype=np.float32)

    snake = np.interp(S_space, C_space, contour)
    snake = snake[:, np.newaxis].view(np.float64).astype(np.float32)

    return snake


class UC1SnakeDataset(torch.utils.data.Dataset):
    def __init__(self, mode='train'):
        super().__init__()
        self.config = yaml.load(open('config.yml'), Loader=yaml.SafeLoader)
        self.root = Path(self.config['data_root'])
        self.cachedir = self.root.parent / 'cache'
        self.cachedir.mkdir(exist_ok=True)
        self.confighash = md5((self.config['bands'], self.config['data_size']))

        self.gts = sorted(list(self.root.glob('ground_truth/*/*/*_30m.tif')))
        if mode == 'train':
            self.gts = [g for g in self.gts if not _isval(g)]
        else:
            self.gts = [g for g in self.gts if _isval(g)]

    def __getitem__(self, idx):
        path = self.gts[idx]
        *_, site, date, gtname = path.parts
        snake_cache = self.cachedir / f'{site}_{date}_snake.npy'
        ref_cache = self.cachedir / f'{site}_{date}_ref_{self.confighash}.npy'

        if snake_cache.exists():
            snake = np.load(snake_cache)
        else:
            with rio.open(path) as raster:
                gt = raster.read(1)
            snake = snakify(gt, self.config['vertices'])
            jnp.save(snake_cache, snake)
        snake = snake.astype(np.float32)

        if ref_cache.exists():
            try:
                ref = np.load(ref_cache)
            except:
                print(f"Failed loading {ref_cache}")
        else:
            ref_root = self.root / 'reference_data' / site / date / '30m'

            ref = []
            for band in self.config['bands']:
                try:
                    with rio.open(ref_root / f'{band}.tif') as raster:
                        H = self.config['data_size']
                        ref.append(jax.image.resize(raster.read(1), (H, H), 'linear').astype(np.uint8))
                except rio.errors.RasterioIOError:
                    print(f'RasterioIOError when opening {ref_root}/{band}.tif')
                    return None
            ref = np.stack(ref, axis=-1).astype(np.float32)
            jnp.save(ref_cache, ref)
        ref = ref.astype(np.float32) / 255

        return ref, snake

    def __len__(self):
        return len(self.gts)


if __name__ == '__main__':
    ds = UC1SnakeDataset('train')
    a = ds[0]
