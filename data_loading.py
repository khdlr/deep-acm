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
import blosc
from tqdm import tqdm


def md5(obj):
    obj = str(obj).encode('utf8')
    return hashlib.md5(obj).hexdigest()[:16]


def _isval(gt_path):
    year = int(gt_path.stem[:4])
    return year >= 2020


class DeterministicShuffle(torch.utils.data.Sampler):
    def __init__(self, length, rng_key, repetitions=50):
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
        collate_fn = numpy_collate,
        drop_last = True
    )
    if mode == 'train':
        kwargs['sampler'] = DeterministicShuffle(len(data), rng_key)

    return torch.utils.data.DataLoader(data, **kwargs)


def snakify(gt, vertices):
    contours = find_contours(gt, 0.5)
    # Select the longest contour
    if len(contours) == 0:
        empty = 0.0 * np.zeros([vertices, 2], np.float32)
        return empty

    contour = max(contours, key=lambda x: x.shape[0])
    contour = contour.astype(np.float32)
    contour = contour.view(np.complex64)[:, 0]
    C_space = np.linspace(0, 1, len(contour), dtype=np.float32)
    S_space = np.linspace(0, 1, vertices, dtype=np.float32)

    snake = np.interp(S_space, C_space, contour)
    snake = snake[:, np.newaxis].view(np.float64).astype(np.float32)

    snake = snake / gt.shape[1]

    return snake


def save(path, ary):
    with open(path, 'wb') as f:
        f.write(blosc.pack_array(ary, cname='zlib'))

def load(path):
    with open(path, 'rb') as f:
        return blosc.unpack_array(f.read())



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

        self.ref_cache_path = self.cachedir / f'{mode}_ref_{self.confighash}.npy'
        self.snake_cache_path = self.cachedir / f'{mode}_snakes_{self.confighash}.npy'

        self.assert_cache()

    def assert_cache(self):
        snake_args = dict(
            filename=str(self.snake_cache_path),
            dtype=np.float32,
            shape=(len(self.gts), self.config['vertices'], 2)
        )

        if not self.snake_cache_path.exists():
            cache = np.memmap(**snake_args, mode='w+')
            for i in tqdm(range(len(self.gts)), desc='Building Snake Cache'):
                cache[i] = self.loadsnake(i) * self.config['data_size']
            cache.flush()
        self.snake_cache = np.memmap(**snake_args, mode='r')

        ref_args = dict(
            filename=self.ref_cache_path,
            dtype=np.uint8,
            shape=(len(self.gts), self.config['data_size'], self.config['data_size'], len(self.config['bands']))
        )

        if not self.ref_cache_path.exists():
            cache = np.memmap(**ref_args, mode='w+')
            for i in tqdm(range(len(self.gts)), desc='Building Ref Cache'):
                cache[i] = self.loadref(i)
            cache.flush()
        self.ref_cache = np.memmap(**ref_args, mode='r')

    def loadsnake(self, idx):
        path = self.gts[idx]
        *_, site, date, gtname = path.parts

        with rio.open(path) as raster:
            gt = raster.read(1)
        snake = snakify(gt, self.config['vertices'])
        return snake

    def loadref(self, idx):
        path = self.gts[idx]
        *_, site, date, gtname = path.parts
        ref_root = self.root / 'reference_data' / site / date / '30m'

        ref = []
        for band in self.config['bands']:
            try:
                with rio.open(ref_root / f'{band}.tif') as raster:
                    b = raster.read(1).astype(np.uint8)
                    ref.append(b)
            except rio.errors.RasterioIOError:
                print(f'RasterioIOError when opening {ref_root}/{band}.tif')
                return None
        ref = jnp.stack(ref, axis=-1)
        ref = jax.image.resize(ref,
            [self.config['data_size'], self.config['data_size'], ref.shape[2]],
            "linear"
        ).astype(np.uint8)
        return ref

    def __getitem__(self, idx):
        snake = self.snake_cache[idx]
        ref = self.ref_cache[idx]
        return ref, snake

    def __len__(self):
        return len(self.gts)


if __name__ == '__main__':
    ds = UC1SnakeDataset('train')
    for a in tqdm(ds):
        pass
