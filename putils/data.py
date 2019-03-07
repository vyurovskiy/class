import os
import cv2
import pickle
import random
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import freeze_support
from queue import Queue
from threading import get_ident, local, Thread
from time import time

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon, MultiPoint
from tqdm import tqdm

from ._mir_hook import mir

_READER = mir.MultiResolutionImageReader()

_SlideInfo = namedtuple(
    '_SlideInfo', ['slide_path', 'mask_path', 'xml_tissue', 'xml_cancer']
)
_PolygonRaw = namedtuple('_PolygonRaw', ['slide_path', 'coord'])
_Polygon = namedtuple('_Polygon', ['slide_path', 'area', 'polygon'])
_Data = namedtuple('_Data', ['slide_path', 'coord', 'label'])

# ######Train part#######


def get_slideinfo(folder_with_data):
    slides_path = folder_with_data + 'slides/'
    masks_path = folder_with_data + 'masks/'
    xmls_path = folder_with_data + 'annotations/'
    for name in os.listdir(slides_path):
        yield _SlideInfo(
            slides_path + name, masks_path + name.split('.')[0] + '_M.tif',
            xmls_path + name.split('.')[0] + '_G.xml',
            xmls_path + name.split('.')[0] + '.xml'
        )


def get_coordinate(_list, cancer=True):
    for el in _list:
        if cancer:
            xml_file = el.xml_cancer
        else:
            xml_file = el.xml_tissue
        for ann in ET.parse(xml_file).getroot().find('Annotations'):
            yx = [
                (int(float(coord.attrib['Y'])), int(float(coord.attrib['X'])))
                for coord in ann.find('Coordinates').findall('Coordinate')
            ]
            yield _PolygonRaw(el.slide_path, yx)


def clear_small_polygon(_list):
    for el in _list:
        p = Polygon(el.coord)
        area = p.area
        if area > 1024 * 1024:
            yield _Polygon(el.slide_path, area, p)


def get_dict(source_files, cancer=True):
    d = defaultdict(list)
    for path, area, poly in clear_small_polygon(
        get_coordinate(source_files, cancer=cancer)
    ):
        d[path].append((area, poly))
    return d


def to_multi(_list):
    return MultiPolygon([[p.exterior.coords, []] for _, p in _list]).buffer(5)


def subtraction(cancer, tissue):
    for k in (set(cancer) & set(tissue)):
        gt = tissue[k]
        canc = cancer[k]
        mp = to_multi(canc)
        tmp = ((poly.buffer(5) - mp) for _, poly in gt)
        tissue[k] = [(p.area, p) for p in tmp]
    return (cancer, tissue)


_TLS = local()


def tls_prng():
    # pylint: disable=no-member
    try:
        return _TLS.prng
    except AttributeError:
        _TLS.prng = np.random.RandomState(
            (get_ident() + np.random.get_state()[1][0]) % 2**32
        )
        return _TLS.prng


def get_random_point(polygon, count):
    prng = tls_prng()

    if not polygon.area:
        return

    minx, miny, maxx, maxy = polygon.bounds
    for _ in range(count):
        while True:
            p = Point(prng.uniform(minx, maxx), prng.uniform(miny, maxy))
            if polygon.contains(p):
                yield (int(p.coords[0][0]), int(p.coords[0][1]))
                break


def get_square(point, size):
    return Polygon(
        [
            (point[0] - size / 2, point[1] - size / 2),
            (point[0] - size / 2, point[1] + size / 2),
            (point[0] + size / 2, point[1] + size / 2),
            (point[0] + size / 2, point[1] - size / 2)
        ]
    )


def bufferize(it, count):
    def produce():
        for item in it:
            q.put(item)
        q.put(None)

    q = Queue(count)
    Thread(target=produce).start()
    yield from iter(q.get, None)


def get_one(args):
    _defdict, target, seed, patch_size, zoom = args

    np.random.seed(seed % 2**32)
    random.seed(seed % 2**32)

    while True:
        path, polys = random.choice(list(_defdict.items()))
        _, poly = random.choice(polys)
        if not poly:
            continue

        point = list(get_random_point(poly, count=1))[0]
        square = get_square(point, 224)
        if not poly.contains(square):
            continue

        mask = _READER.open(
            path.replace('slides', 'masks').replace('.tif', '_M.tif')
        )
        slide = _READER.open(path)
        slide_patch, mask_patch = (
            f.getUCharPatch(
                startY=point[0],
                startX=point[1],
                height=patch_size[0],
                width=patch_size[1],
                level=zoom
            ) for f in (slide, mask)
        )
        for f in (slide, mask):
            f.close()
        mask_patch = mask_patch.astype(bool)
        values, counts = np.unique(mask_patch, return_counts=True)
        area = sum(values * counts) / sum(counts)
        if area == target and len(np.unique(slide_patch)) > 225:
            return _Data(path, point, target)


def get_data(slideinfo, nbpoints, patch_size, zoom):
    seed = int(time() * 1000)

    cancer = get_dict(slideinfo, cancer=True)
    tissue = get_dict(slideinfo, cancer=False)
    cancer, tissue = subtraction(cancer, tissue)

    arg_gen = (
        (_defdict, target, seed + i + nbpoints * target, patch_size, zoom)
        for target, _defdict in enumerate((tissue, cancer))
        for i in tqdm(range(nbpoints))
    )

    with ThreadPoolExecutor(12) as pool:
        #with ProcessPoolExecutor(12) as pool:
        for f in bufferize(
            (pool.submit(get_one, args) for args in arg_gen), count=240
        ):
            yield f.result()


def gen_and_save_data(
    folder_with_data, folder_to_save, nbpoints, patch_size, zoom
):
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)
    slideinfo = list(get_slideinfo(folder_with_data))
    np.random.shuffle(slideinfo)
    train_slideinfo, valid_slideinfo = (
        slideinfo[:-len(slideinfo) // 10], slideinfo[-len(slideinfo) // 10:]
    )
    train = list(get_data(train_slideinfo, nbpoints, patch_size, zoom))
    valid = list(
        get_data(valid_slideinfo, int(nbpoints * 0.1), patch_size, zoom)
    )
    np.random.shuffle(train)
    np.random.shuffle(valid)
    for p in (train, valid):
        if train:
            with open(folder_to_save + 'train.pickle', 'wb') as f:
                pickle.dump(p, f, pickle.HIGHEST_PROTOCOL)
        if valid:
            with open(folder_to_save + 'valid.pickle', 'wb') as f:
                pickle.dump(p, f, pickle.HIGHEST_PROTOCOL)


def load_data(folder_with_data):
    with open(folder_with_data + 'train.pickle', 'rb') as f:
        train = pickle.load(f)
    with open(folder_with_data + 'valid.pickle', 'rb') as f:
        valid = pickle.load(f)
    return train, valid


# ######Test part#######
def get_thumbnail(slide_path, zoom=4):
    slide = _READER.open(slide_path)
    slide_shape = slide.getDimensions()
    thumbnail = slide.getUCharPatch(
        0, 0, slide_shape[0] // 2**zoom, slide_shape[1] // 2**zoom, zoom
    )
    slide.close()
    return (thumbnail, slide_shape)


def normalize(thumbnail):
    if not isinstance(thumbnail, np.ndarray):
        thumbnail = np.array(thumbnail)
    thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2LAB)
    l, *ab = cv2.split(thumbnail)
    clahe = cv2.createCLAHE(2., (8, 8))
    l = clahe.apply(l)
    thumbnail = cv2.merge([l, *ab])
    thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_LAB2BGR)
    return thumbnail


def get_aprox_contours(thumbnail):
    gray = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 10, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    _, contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for cont in contours:
        yield cv2.approxPolyDP(cont, 0.001 * cv2.arcLength(cont, True),
                               True).reshape(-1, 2)


def get_grid_points(shape, patch_size):
    mg = np.stack(
        np.meshgrid(
            np.arange(0, shape[0], patch_size[0] // 2),
            np.arange(0, shape[1], patch_size[1] // 2),
            indexing='ij'
        ),
        axis=-1
    ).reshape(-1, 2)
    return MultiPoint(mg)


def to_multi2(contours, zoom=4):
    return MultiPolygon([p * 2**zoom, []] for p in contours).buffer(0)


def get_test_points(grid_points, multi_poly):
    for point in grid_points.intersection(multi_poly):
        yield point.coords[0]


def get_patch(xy):
    slide = _READER.open()


def get_test(slide_path, patch_size, zoom=4):
    thumbnail, shape = get_thumbnail(slide_path, zoom)
    thumbnail = normalize(thumbnail)
    contours = list(get_aprox_contours(thumbnail))
    multi_poly = to_multi2(contours, zoom)
    grid_points = get_grid_points(shape, patch_size)
    test_points = list(get_test_points(grid_points, multi_poly))
    return test_points
