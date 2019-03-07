import itertools
import os
import sys
import gc
from contextlib import contextmanager
from multiprocessing import freeze_support
from multiprocessing import Pool
# from multiprocessing.dummy import Pool

import cv2
import numpy as np
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, create_supervised_evaluator
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from torchvision.transforms import functional as F

from putils._mir_hook import mir
from putils.data import get_test, normalize, _READER
from putils.generator import TestDataset

if sys.platform == 'win32':
    _DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    _DEVICE = torch.device(os.environ.get('CUDA_VISIBLE_DEVICES', 'cpu'))

if __name__ == '__main__':
    from tqdm import tqdm
else:
    from tqdm import tqdm_notebook as tqdm

_BATCH_SIZE = os.cpu_count()
# _BATCH_SIZE = 1


@contextmanager
def open_slide(path):
    slide = mir.MultiResolutionImageReader().open(path)
    try:
        yield slide
    finally:
        slide.close()


def _grouped(iterable, size):
    iterator = iter(iterable)
    yield from iter(lambda: list(itertools.islice(iterator, size)), [])


class get_patch:
    def __init__(self, path, patch_size, mask_path):
        self.path, self.patch_size, self.mask_path = path, patch_size, mask_path

    def __call__(self, args):
        i, xy = args
        with open_slide(self.path) as slide:
            tensor = F.to_tensor(
                normalize(
                    slide.getUCharPatch(
                        startY=int(xy[0]),
                        startX=int(xy[1]),
                        height=self.patch_size,
                        width=self.patch_size,
                        level=0
                    )
                )
            )
        with open_slide(self.mask_path) as mask:
            label = mask.getUCharPatch(
                startY=int(xy[0]),
                startX=int(xy[1]),
                height=self.patch_size,
                width=self.patch_size,
                level=0
            ).any()

        gc.collect()
        return i, (tensor, label)


def make_odd(x):
    x = int(x)
    if x % 2 == 0:
        return x + 1
    return x


def load_model(path_to_model):
    model = inception_v3(num_classes=1, aux_logits=False)
    model.load_state_dict(torch.load(path_to_model))
    model = model.to(_DEVICE)
    model.eval()
    return model


def predict(model, path, patch_size=224, batch_size=_BATCH_SIZE, overlap=0.1):
    level = 4
    mask_path = path.replace('slides', 'masks').replace('.tif', '_M.tif')

    with open_slide(path) as slide:
        zoom = 1 / slide.getLevelDownsample(level)

        width, height = slide.getDimensions()
        thumbnail = slide.getUCharPatch(
            startY=0,
            startX=0,
            height=int(height * zoom),
            width=int(width * zoom),
            level=level
        )

    gray = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 10, 250)
    closed = cv2.morphologyEx(
        edged, cv2.MORPH_DILATE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (make_odd(patch_size * zoom), ) * 2
        )
    )

    # ys = np.arange(12000, 102000, int(patch_size * (1 - overlap)))
    # xs = np.arange(15000, 81000, int(patch_size * (1 - overlap)))
    ys = np.arange(0, height, int(patch_size * (1 - overlap)))
    xs = np.arange(0, width, int(patch_size * (1 - overlap)))

    test_points = itertools.product(ys, xs)
    test_points = (
        (
            xy,
            closed[int(xy[0] * zoom), int(xy[1] * zoom)],
        ) for xy in test_points
    )

    nb_points = len(ys) * len(xs)
    target = np.ones(nb_points, dtype='uint8')
    prediction = np.ones(nb_points, dtype='float32')

    def filter_easy(ps):
        for i, (xy, usable) in enumerate(ps):
            if usable:
                yield i, xy
            else:
                prediction[i] = 0
                target[i] = 0

    test_points = tqdm(
        test_points, total=nb_points, desc='patches extracted', leave=True
    )
    test_points = filter_easy(test_points)
    gc.collect()

    with Pool(_BATCH_SIZE) as pool:
        patch_tuples = pool.imap(
            get_patch(path, patch_size, mask_path), test_points, chunksize=32
        )

        def select_patches(ps):
            for i, (patch, label) in ps:
                target[i] = label
                yield patch

        patches = select_patches(patch_tuples)

        batches = (
            torch.stack(group).to(_DEVICE)
            for group in _grouped(patches, _BATCH_SIZE)
        )
        with torch.no_grad():
            with tqdm(desc='patches recognized', leave=True) as t:
                flats = sum(
                    (
                        (
                            model(batch)[..., 0].sigmoid().cpu().numpy().
                            tolist(), t.update(_BATCH_SIZE)
                        )[0] for batch in batches
                    ), []
                )

    it = iter(flats)
    try:
        for i in range(len(prediction)):
            if prediction[i]:
                prediction[i] = next(it)
    except StopIteration:
        pass

    prediction = prediction.reshape(len(ys), len(xs))
    target = target.reshape(len(ys), len(xs))

    # thres = .5
    # prediction = (prediction > thres)

    prediction = (prediction * 255).clip(0, 255).astype('uint8')
    cv2.imwrite('target.png', target * 255)
    cv2.imwrite('prediction.png', prediction)

    return prediction, target


# if __name__ == '__main__':
#     freeze_support()
#     main(
#         'D:/ACDC_LUNG_HISTOPATHOLOGY/data/slides/32.tif', (244, 244), 22,
#         'D:/ACDC_LUNG_try2/checkpoints/19-02-22/model_inception_2277_acc=0.967.pth'
#     )

if __name__ == '__main__':
    freeze_support()
    model = load_model(
        'D:/ACDC_LUNG_try2/checkpoints/19-02-22/model_inception_2277_acc=0.967.pth'
    )
    predict(model, 'D:/ACDC_LUNG_HISTOPATHOLOGY/data/slides/21.tif')
