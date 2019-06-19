import utils
from cv2 import imencode
from skimage.transform import resize
from skimage import io
import numpy as np
import scipy.io
import sys
import os
from glob import glob
import tensorflow as tf
import argparse
import pickle


parser = argparse.ArgumentParser()
required = parser.add_argument_group('required arguments')
required.add_argument(
    "-i", "--images",
    help="path to directory containing the stacks", required=True)
required.add_argument(
    "-l", "--labels",
    nargs='+',
    help="path to the files containing the labels (use wildcards)", required=True)
required.add_argument(
    "-n", "--name",
    help="A name for the dataset", required=True)
parser.add_argument(
    "-s", "--image-size",
    type=int,
    default=512,
    help="Output images size (squared images)")
parser.add_argument(
    "-csixy", "--cells-size-xy",
    type=int,
    default=32,
    help="Size of the cells in the output images")
parser.add_argument(
    "-csiz", "--cells-size-z",
    type=int,
    default=1,
    help="Size of the cells in the output images")
parser.add_argument(
    "-cstxy", "--cells-stride-xy",
    type=int,
    default=32,
    help="Stride for the cells placement")
parser.add_argument(
    "-cstz", "--cells-stride-z",
    type=int,
    default=1,
    help="Stride for the cells placement")
###### ADDITIONAL ARGS ... #######
parser.add_argument(
    "-cpd", "--cells-padding",
    default="VALID",
    choices=["SAME", "VALID"],
    help="Padding for the cells placement")
parser.add_argument(
    "-rs", "--records-size",
    type=int,
    help="Maximum tfrecord size in Mb")
parser.add_argument(
    "-f", "--format",
    default="jpg",
    choices=["png", "jpg"],
    help="format of the encoding inside the tfrecords")


args = parser.parse_args()
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)
dtlabel = np.dtype([
    ("stackindex", "<U50"),
    ("spinemaxcoordinatex", np.float64),
    ("spinemaxcoordinatey", np.float64),
    ("frameindexmaxspine", np.int32)])


def read_tiff_stack(path):
    return np.moveaxis(io.imread(path), 0, -1)


def resize_image(tiff, image_size):
    return resize(tiff, (args.image_size ,args.image_size))


def to_uint8(images):
    mini = images.min()
    maxi = images.max()
    return (255 * (images - mini) / (maxi - mini)).astype(np.uint8)


def encode_image(images):
    images = images.reshape((images.shape[0], -1, 1))
    images = to_uint8(images)
    if args.format is "png":
        sucess, data = imencode(".png", images)
        return data.tobytes()
    elif args.format is "jpg":
        sucess, data = imencode(".jpg", images)
        return data.tobytes()


def read_labels_file(path):
    tmp = scipy.io.loadmat(path)["savevariable"][0]
    ret = np.zeros(shape=tmp.shape[0], dtype=dtlabel)
    ret["stackindex"] = [arr[0] if arr[0].dtype.kind is 'U' else '' for arr in tmp["stackindex"]]
    ret["spinemaxcoordinatex"] = [arr.squeeze() or -1 for arr in tmp["spinemaxcoordinatex"]]
    ret["spinemaxcoordinatey"] = [arr.squeeze() or -1 for arr in tmp["spinemaxcoordinatey"]]
    ret["frameindexmaxspine"] = [arr.squeeze() or -1 for arr in tmp["frameindexmaxspine"]]
    return ret


def all_labels_files(paths):
    return np.concatenate([read_labels_file(path) for path in paths])


def resize_coords(stack_x, stack_y):
    return stack_x / 512 * args.image_size, stack_y / 512 * args.image_size


def cells_starts_ends(imsize, cellsize, cellstride):
    return utils.cells_starts_ends(imsize, cellsize, cellstride, args.cells_padding)


def indices_cells_one_dimension(imsize, cellsize, cellstride, coord):
    starts, ends = cells_starts_ends(imsize, cellsize, cellstride)
    indices, = np.where(np.logical_and(coord < ends, coord >= starts))
    return indices


def _remap(mini, maxi, val):
    maxi -= 1
    diff = maxi - mini
    return 2 * val / diff - (mini + maxi) / diff


def to_cells_coords(stack_x, stack_y, stack_z, stack_depth):
    stack_x, stack_y = resize_coords(stack_x, stack_y)
    indices_x = indices_cells_one_dimension(args.image_size, args.cells_size_xy, args.cells_stride_xy, stack_x)
    indices_y = indices_cells_one_dimension(args.image_size, args.cells_size_xy, args.cells_stride_xy, stack_y)
    indices_z = indices_cells_one_dimension(stack_depth, args.cells_size_z, args.cells_stride_z, stack_z)
    cells_coords = np.array(np.meshgrid(indices_x, indices_y, indices_z)).T.reshape(-1,3)
    starts_x, ends_x = cells_starts_ends(args.image_size, args.cells_size_xy, args.cells_stride_xy)
    starts_y, ends_y = cells_starts_ends(args.image_size, args.cells_size_xy, args.cells_stride_xy)
    starts_z, ends_z = cells_starts_ends(stack_depth, args.cells_size_z, args.cells_stride_z)
    starts = np.stack([starts_x[cells_coords[:, 0]], starts_y[cells_coords[:, 1]], starts_z[cells_coords[:, 2]]], axis=-1)
    ends = np.stack([ends_x[cells_coords[:, 0]], ends_y[cells_coords[:, 1]], ends_z[cells_coords[:, 2]]], axis=-1)
    within_cells_coords = _remap(starts, ends, np.array([stack_x, stack_y, stack_z]))
    return cells_coords, within_cells_coords


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float64_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def stack_example(images, labels):
    image_shape = images.shape
    all_stack_x = labels["spinemaxcoordinatex"]
    all_stack_y = labels["spinemaxcoordinatey"]
    all_stack_z = labels["frameindexmaxspine"]
    all_cells_coords = []
    all_within_cells_coords = []
    for stack_x, stack_y, stack_z in zip(all_stack_x, all_stack_y, all_stack_z):
        cells_coords, within_cells_coords = to_cells_coords(stack_x, stack_y, stack_z, image_shape[-1])
        all_cells_coords.append(cells_coords)
        all_within_cells_coords.append(within_cells_coords)
    all_cells_coords = np.concatenate(all_cells_coords, axis=0)
    all_within_cells_coords = np.concatenate(all_within_cells_coords, axis=0)
    feature = {
        'height': _int64_feature([image_shape[0]]),                               # single value
        'width': _int64_feature([image_shape[1]]),                                # single value
        'depth': _int64_feature([image_shape[2]]),                                # single value
        'true_coords_x': _float64_feature(all_stack_x),                           # list
        'true_coords_y': _float64_feature(all_stack_y),                           # list
        'true_coords_z': _int64_feature(all_stack_z),                             # list
        'cells_coords_x': _int64_feature(all_cells_coords[:, 0]),                 # list
        'cells_coords_y': _int64_feature(all_cells_coords[:, 1]),                 # list
        'cells_coords_z': _int64_feature(all_cells_coords[:, 2]),                 # list
        'within_cells_coords_x': _float64_feature(all_within_cells_coords[:, 0]), # list
        'within_cells_coords_y': _float64_feature(all_within_cells_coords[:, 1]), # list
        'within_cells_coords_z': _float64_feature(all_within_cells_coords[:, 2]), # list
        'stack': _bytes_feature([encode_image(images)])                           # single value
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def args_to_path():
    return (
        "../tfrecords/"
        "cell_size_{:03d}_{:03d}__"
        "cell_strides_{:03d}_{:03d}__"
        "padding_{}__"
        "image_size_{:03d}/"
    ).format(args.cells_size_xy,
             args.cells_size_z,
             args.cells_stride_xy,
             args.cells_stride_z,
             args.cells_padding,
             args.image_size)


def to_tfrecords(labels):
    pass


if __name__ == "__main__":
    # example of a command line for making a dataset:
    # clear; p3 make_tf_records.py -i /home/kaschube-shared/spines/SpineMeasurements_XYcoordinates/SR51N1/ -l /home/kaschube-shared/spines/SpineMeasurements_XYcoordinates/SR51N1/SR51*eu*end*ay*.mat -n test -cstxy 8 -cstz 2 -csiz 4 -cpd SAME

    output_path = args_to_path()
    tfrecord_path = output_path + args.name + ".tfrecord"
    if os.path.exists(output_path) and os.path.exists(tfrecord_path):
        print("A dataset with those parameters already exists. Do nothing.")
        sys.exit(0)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    labels = all_labels_files(args.labels)
    stack_names = np.unique(labels["stackindex"])
    stack_names = stack_names[np.where(stack_names != "")[0]]
    print("Found {} stacks in the labeling files and {} spines:".format(len(stack_names), len(labels)))
    for stack_name in stack_names:
        print(" - ", stack_name, "" if os.path.exists(args.images + "/" + stack_name) and os.path.isfile(args.images + "/" + stack_name) else "stack file not found")
    with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
        for stack_name in stack_names:
            path_to_tiff = args.images + "/" + stack_name
            if os.path.exists(path_to_tiff) and os.path.isfile(path_to_tiff):
                print("processing stack '{}'".format(stack_name))
                images = read_tiff_stack(path_to_tiff)
                images = resize_image(images, args.image_size)
                where = np.where(labels["stackindex"] == stack_name)
                stack_labels = labels[where]
                example = stack_example(images, stack_labels)
                writer.write(example.SerializeToString())
    with open(output_path + args.name + ".args", "wb") as f:
        pickle.dump(args, f)
