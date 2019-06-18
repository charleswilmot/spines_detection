import numpy as np


def cells_starts_ends(imsize, cellsize, cellstride, cells_padding):
    if cells_padding == "VALID":
        starts = np.arange(0, imsize - cellsize + 1, cellstride)
        ends = starts + cellsize
    elif cells_padding == "SAME":
        out_shape = np.ceil(imsize / cellstride)
        pad_total = ((out_shape - 1) * cellstride + cellsize - imsize)
        pad_beg = pad_total // 2
        starts = np.arange(out_shape) * cellstride - pad_beg
        ends = starts + cellsize
    return starts, ends
