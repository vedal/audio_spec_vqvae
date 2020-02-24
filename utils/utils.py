import numpy as np


def minmax(x):
    x -= x.min()
    x /= x.max()
    return x


def count_files_in_folder(path):
    # counts the number of files in a folder
    import os
    path, _, files = next(os.walk(path))
    file_count = len(files)
    return file_count


def slerp(p0, p1, t):
    # Spherical interpolation

    # p0 = first vector
    # p1 = second vector
    # t = time step value (equal-sized steps between 0 and 1)
    omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1


def denormalize(arr, offset=0.0):
    return (arr + offset) * 255.


def softmax(z):
    # z: K x N
    # # normalization by summing over all p_i

    # safety measure from cs231n.github.io/linear-classify/
    # z -= np.max(z,axis=0)
    return np.exp(z) / np.sum(np.exp(z), axis=0)
