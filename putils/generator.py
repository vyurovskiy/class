from torch.utils.data import Dataset, RandomSampler
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import torch
import cv2

from ._mir_hook import mir

_READER = mir.MultiResolutionImageReader()


def noisify_colors(x, sigma=10):
    return np.random.normal(x, scale=sigma).clip(0, 255).astype('uint8')


def _noisify(x, sigma=5):
    scales = (1, 2, 4)
    octaves = (
        np.random.normal(
            scale=sigma * s, size=(x.shape[0] // s, x.shape[1] // s)
        ).astype('float32') for s in scales
    )
    simplex = sum(
        cv2.resize(oct, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        for s, oct in zip(scales, octaves)
    )
    return (x + simplex).clip(0, 255).astype('uint8')


def normalize(x, noise: bool):
    x = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
    l, *ab = cv2.split(x)

    clahe = cv2.createCLAHE(2., (8, 8))
    l = clahe.apply(l)
    if noise:
        l = _noisify(l)

    x = cv2.merge([l, *ab])
    x = cv2.cvtColor(x, cv2.COLOR_LAB2BGR)
    return x


class RandomSamplerOk(RandomSampler):
    def __len__(self):
        return self.num_samples


class CancerDatset(Dataset):
    def __init__(
        self,
        _list,
        patch_size,
        zoom,
        transforms=None,
        noise=True,
        normalize=True
    ):
        self._list = _list
        self.patch_size = patch_size
        self.zoom = zoom
        self.transforms = transforms
        self.noise = noise
        self.normalize = normalize

    def __len__(self):
        return len(self._list)

    def __getitem__(self, index):
        slide = _READER.open(self._list[index].slide_path)
        slide_patch = slide.getUCharPatch(
            startY=self._list[index].coord[0],
            startX=self._list[index].coord[1],
            height=314,
            width=314,
            level=self.zoom
        )

        slide_patch = Image.fromarray(slide_patch).convert('RGB')

        if self.transforms is not None:
            slide_patch = self.transforms(slide_patch)

        slide_patch = F.center_crop(slide_patch, self.patch_size)

        if self.normalize:
            slide_patch = np.asarray(slide_patch)
            slide_patch = normalize(slide_patch, noise=self.noise)
            if self.noise:
                slide_patch = noisify_colors(slide_patch)
            slide_patch = Image.fromarray(slide_patch)

        slide_patch = F.to_tensor(slide_patch)

        y = self._list[index].label
        y = torch.tensor([y], dtype=torch.float32)
        # y = float(y)

        # if __debug__:
        #     np_image = slide_patch.numpy().transpose(1, 2, 0)[..., ::-1]
        #     cv2.imshow('image', np_image)
        #     cv2.waitKey(16)

        return slide_patch, y


class TestDataset(Dataset):
    def __init__(self, slide_path, _list, patch_size, zoom):
        self.slide_path = slide_path
        self._list = _list
        self.patch_size = patch_size
        self.zoom = zoom

    def __len__(self):
        return len(self._list)

    def __getitem__(self, index):
        slide = _READER.open(self.slide_path)
        slide_patch = slide.getUCharPatch(
            startX=int(self._list[index][0]),
            startY=int(self._list[index][1]),
            height=self.patch_size[0],
            width=self.patch_size[1],
            level=self.zoom
        )
        slide_patch = Image.fromarray(slide_patch).convert('RGB')
        slide_patch = F.to_tensor(slide_patch)
        return slide_patch, 0
