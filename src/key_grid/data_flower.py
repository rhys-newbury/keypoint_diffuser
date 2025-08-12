"""
Created on Fri Apr 24 22:59:57 2020

@author: eliphat
"""
import os
import random

import h5py
import numpy


def load_h5(h5_filename, normalize=False, include_label=False):
    f = h5py.File(h5_filename, "r")
    data = f["data"][:]  # (n, 2048, 3)
    if normalize:
        dmin = data.min(axis=1, keepdims=True).min(axis=-1, keepdims=True)
        dmax = data.max(axis=1, keepdims=True).max(axis=-1, keepdims=True)
        data = (data - dmin) / (dmax - dmin)
        data = 2.0 * (data - 0.5)
    if include_label:
        label = f["label"][:]
        return data, label
    return data


def all_h5(
    parent,
    normalize=False,
    include_label=False,
    subclasses=tuple(range(40)),
    sample=256,
):
    lazy = (load_h5(x, normalize, include_label) for x in walk_files(parent))
    if include_label:
        xy = tuple(lazy)
        x = [x for x, y in xy]
        y = [y for x, y in xy]
        x = numpy.concatenate(x)
        y = numpy.concatenate(y)
        xf = []
        yf = []
        for xp, yp in zip(x, y):
            if yp[0] in subclasses:
                if sample is None:
                    xf.append(xp)
                else:
                    xf.append(random.choices(xp, k=sample))
                yf.append(numpy.eye(len(subclasses))[subclasses.index(yp[0])])
        return numpy.array(xf), numpy.array(yf)
    return numpy.concatenate(tuple(lazy))


def walk_files(path):
    for r, _ds, fs in os.walk(path):
        for f in fs:
            yield os.path.join(r, f)


def last_dirname(file_path):
    return os.path.basename(os.path.dirname(file_path))


def dataset_split(path):
    flist = list(walk_files(path))
    tr = filter(lambda p: "train" in last_dirname(p), flist)
    te = filter(lambda p: "test" in last_dirname(p), flist)
    return list(tr), list(te)
