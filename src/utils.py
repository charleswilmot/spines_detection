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
    return starts.astype(np.int), ends.astype(np.int)


def chained_cells_starts_ends(image_size, params):
    starts, ends = cells_starts_ends(image_size, *params[0])
    for param in params[1:]:
        length = len(starts)
        print(param, length)
        last_starts, last_ends = cells_starts_ends(length, *param)
        last_starts[np.where(last_starts < 0)] = 0
        last_ends[np.where(last_ends >= length)] = length
        print("starts", last_starts)
        print("ends", last_ends)
        starts = starts[last_starts]
        ends = ends[last_ends - 1]
    return starts, ends


def get_cell_properties(image_size, params_dict):
    starts_xy, ends_xy = chained_cells_starts_ends(image_size, params_dict["xy"])
    if len(starts_xy) < 2:
        raise ValueError("Less than two cells on x/y dim")
    starts_z, ends_z = chained_cells_starts_ends(image_size, params_dict["z"])
    if len(starts_z) < 2:
        raise ValueError("Less than two cells on z dim")
    half_xy = len(starts_xy) // 2
    half_z = len(starts_z) // 2
    cell_size_xy = (ends_xy - starts_xy)[half_xy]
    cells_stride_xy = (starts_xy[1:] - starts_xy[:-1])[half_xy]
    cell_size_z = (ends_z - starts_z)[half_z]
    cells_stride_z = (starts_z[1:] - starts_z[:-1])[half_z]
    return cell_size_xy, cells_stride_xy, cell_size_z, cells_stride_z


if __name__ == "__main__":
    image_size = 512
    params = [
        (4, 4, "SAME"),
        (4, 4, "SAME"),
        (2, 2, "SAME"),
        (2, 1, "SAME")
    ]
    # starts, ends = chained_cells_starts_ends(image_size, params)

    # print(starts)
    # print(ends)
    # print(ends - starts)
    # print(starts[1:] - starts[:-1])
    params_dict = {"xy": params, "z": params}
    cell_size_xy, cells_stride_xy, cell_size_z, cells_stride_z = get_cell_properties(image_size, params_dict)
    print("cell_size_xy, cells_stride_xy, cell_size_z, cells_stride_z")
    print(cell_size_xy, cells_stride_xy, cell_size_z, cells_stride_z)
