import time
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from utils import cells_starts_ends
from dataset import naked_dataset
import argparse
import pickle


parser = argparse.ArgumentParser()
parser.add_argument(
    "path",
    help="path to the tfrecord file"
)

args = parser.parse_args()


dirpath, name = os.path.split(args.path)
name = name.replace(".tfrecord", "")
with open(dirpath + '/' + name + ".args", "rb") as f:
    generation_args = pickle.load(f)
image_size = generation_args.image_size
starts_xy, ends_xy = cells_starts_ends(generation_args.image_size, generation_args.cells_size_xy, generation_args.cells_stride_xy, generation_args.cells_padding)
starts_z, ends_z = cells_starts_ends(100, generation_args.cells_size_z, generation_args.cells_stride_z, generation_args.cells_padding)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
image = ax.imshow(np.random.randint(0, 256, size=(image_size, image_size)))
methode1 = plt.Circle((0, 0), 10, color='r', alpha=0.2)
methode2 = plt.Rectangle((0, 0), 14, 14, color='g', alpha=0.5)
ax.add_artist(methode1)
ax.add_artist(methode2)
fig.show()
#### some stuff missing here


dataset = naked_dataset(args.path)
dataset_iterator = dataset.make_initializable_iterator()
datapoint = dataset_iterator.get_next()

with tf.Session() as sess:
    sess.run(dataset_iterator.initializer)
    try:
        while(True):
            datadict = sess.run(datapoint)
            n_spines = len(datadict["true_coords_x"])
            for i in range(n_spines):
                x1 = datadict["true_coords_x"][i]
                y1 = datadict["true_coords_y"][i]
                z1 = datadict["true_coords_z"][i]
                methode1.set_center((x1, y1))

                cell_x = datadict["cells_coords_x"][i]
                cell_y = datadict["cells_coords_y"][i]
                cell_z = datadict["cells_coords_z"][i]
                within_cell_x = datadict["within_cells_coords_x"][i]
                within_cell_y = datadict["within_cells_coords_y"][i]
                within_cell_z = datadict["within_cells_coords_z"][i]
                sx = starts_xy[cell_x]
                sy = starts_xy[cell_y]
                sz = starts_z[cell_z]
                ex = ends_xy[cell_x]
                ey = ends_xy[cell_y]
                ez = ends_z[cell_z]
                x2 = sx + (ex - sx - 1) * (within_cell_x + 1) / 2
                y2 = sy + (ey - sy - 1) * (within_cell_y + 1) / 2
                z2 = sz + (ez - sz - 1) * (within_cell_z + 1) / 2
                methode2.set_xy((x2 - 7, y2 - 7))

                print(x1, y1, z1, "  ", x2, y2, z2)
                frame = datadict["stack"][0, :, :, z1, 0]
                image.set_data(frame)
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(1)
    except tf.errors.OutOfRangeError:
        pass
