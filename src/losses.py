import numpy as np
import tensorflow as tf


def get_losses(net_out, cells_coords, within_cells_coords, lambda_coord, lambda_noobj):
    # put net_out to the format of coords:
    n_detectors = net_out.get_shape().as_list()[-1] // 4
    detectors = tf.gather_nd(net_out[0], cells_coords)       # shape = [None, 4 * n_detectors]
    detectors = tf.reshape(detectors, (-1, n_detectors, 4))  # shape = [None, n_detectors, 4]
    predicted_coords = detectors[..., :-1]                   # shape = [None, n_detectors, 3]
    confidence = detectors[..., -1]                          # shape = [None, n_detectors]
    true_coords = within_cells_coords[:, tf.newaxis]         # shape = [None, 1, 3]
    distances = tf.reduce_sum((true_coords - predicted_coords) ** 2, axis=-1)  # [None, n_detectors]
    min_distance = tf.reduce_min(distances, axis=-1)         # [None]
    arg_chosen_detectors = tf.argmin(distances, axis=-1)     # [None]
    index = tf.cast(tf.range(tf.shape(arg_chosen_detectors)[0]), dtype=tf.int64)
    coord_chosen_detectors = tf.stack([index, arg_chosen_detectors], axis=-1)
    trick_coef = tf.stop_gradient(1 + lambda_noobj * confidence / (confidence - 1))   # error when confidence == 1  --> solution = argsort instead of argmin + gather_nd on non minimal indices
    min_distance_loss = min_distance * lambda_coord
    confidence_loss = (tf.gather_nd(confidence, coord_chosen_detectors) - 1) ** 2 * trick_coef
    noobj_loss = net_out[..., 3::4] ** 2 * lambda_noobj
    min_distance_loss_sum = tf.reduce_sum(min_distance_loss)
    confidence_loss_sum = tf.reduce_sum(confidence_loss)
    noobj_loss_sum = tf.reduce_sum(noobj_loss)
    return min_distance_sum, confidence_loss_sum, noobj_loss_sum


if __name__ == "__main__":
    from network import SimpleLinearModel
    from dataset import naked_dataset
    import argparse
    import pickle
    import os


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

    dataset = naked_dataset(args.path)
    dataset_iterator = dataset.make_initializable_iterator()
    datapoint = dataset_iterator.get_next()

    n_detectors_per_cell = 3
    net = SimpleLinearModel(generation_args, n_detectors_per_cell)
    inp = tf.cast(datapoint["stack"], tf.float32) / 127.5 - 1
    net_out = net(inp)
    cells_coords = tf.stack([
        datapoint["cells_coords_x"],
        datapoint["cells_coords_y"],
        datapoint["cells_coords_z"]], axis=1)
    within_cells_coords = tf.stack([
        datapoint["within_cells_coords_x"],
        datapoint["within_cells_coords_y"],
        datapoint["within_cells_coords_z"]], axis=1)
    lambda_coord = 5
    lambda_noobj = 0.5
    loss = get_losses(net_out, cells_coords, within_cells_coords, lambda_coord, lambda_noobj)

    with tf.Session() as sess:
        sess.run([dataset_iterator.initializer, tf.global_variables_initializer()])
        np_loss = sess.run(loss)

    print(np_loss)
