import numpy as np
import tensorflow as tf


def get_losses_old_and_probably_wrong(net_out, cells_coords, within_cells_coords, lambda_coord, lambda_noobj):
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


def get_losses_old_2(net_out, cells_coords, within_cells_coords, lambda_coord, lambda_noobj):
    # put net_out to the format of coords:
    n_detectors = net_out.get_shape().as_list()[-1] // 4
    detectors = tf.gather_nd(net_out[0], cells_coords)       # shape = [None, 4 * n_detectors]
    detectors = tf.reshape(detectors, (-1, n_detectors, 4))  # shape = [None, n_detectors, 4]
    predicted_coords = detectors[..., :-1]                   # shape = [None, n_detectors, 3]
    confidence = detectors[..., -1]                          # shape = [None, n_detectors]
    true_coords = within_cells_coords[:, tf.newaxis]         # shape = [None, 1, 3]
    distances = tf.reduce_sum((true_coords - predicted_coords) ** 2, axis=-1)  # [None, n_detectors]
    ## distance loss
    min_distance = tf.reduce_min(distances, axis=-1)         # [None]
    min_distance_loss = lambda_coord * min_distance
    ## confidence losses
    # obj loss
    arg_chosen_detectors = tf.argmin(distances, axis=-1)     # [None]
    indices = tf.cast(tf.range(tf.shape(arg_chosen_detectors)[0]), dtype=tf.int64)
    coord_chosen_detectors = tf.stack([indices, arg_chosen_detectors], axis=-1)
    confidence_loss = (tf.gather_nd(confidence, coord_chosen_detectors) - 1) ** 2
    # noobj loss
    length = tf.shape(indices)[0]
    indices = tf.concat([tf.zeros(length, dtype=tf.int64)[:, tf.newaxis], cells_coords, arg_chosen_detectors[:, tf.newaxis]], axis=-1)   # WARNING: how about the batch dim...?
    updates = tf.ones(length)
    shape = tf.cast(tf.shape(net_out[..., 3::4]), tf.int64)
    mask = 1 - tf.scatter_nd(indices, updates, shape)
    noobj_loss = lambda_noobj * mask * net_out[..., 3::4] ** 2
    # sums
    min_distance_loss_sum = tf.reduce_sum(min_distance_loss)
    confidence_loss_sum = tf.reduce_sum(confidence_loss)
    noobj_loss_sum = tf.reduce_sum(noobj_loss)
    return min_distance_loss_sum, confidence_loss_sum, noobj_loss_sum


def get_losses(net_out, cells_coords, within_cells_coords, lambda_coord, lambda_noobj):
    # put net_out to the format of coords:
    NO_BATCH = 0
    n_detectors = net_out.get_shape().as_list()[-1] // 4
    n_spines = tf.shape(cells_coords)[1]
    print("cells_coords", cells_coords.get_shape())             # shape = [None, n_spines, 3]
    print("net_out", net_out.get_shape())                       # shape = [None, x, y, z, 4 * n_detectors]
    detectors = tf.gather_nd(net_out[NO_BATCH], cells_coords[NO_BATCH])       # shape = [None, 4 * n_detectors]
    detectors = tf.reshape(detectors, (-1, n_detectors, 4))     # shape = [n_spines, n_detectors, 4]
    print("detectors", detectors.get_shape())                   # shape = [n_spines, n_detectors, 4]
    predicted_coords = detectors[..., :-1]                      # shape = [n_spines, n_detectors, 3]
    confidence = detectors[..., -1]                             # shape = [n_spines, n_detectors]
    print("within_cells_coords", within_cells_coords.get_shape())                      # shape = [None, n_spines, 3]
    true_coords = tf.transpose(within_cells_coords[NO_BATCH, tf.newaxis], perm=[1, 0, 2])            # shape = [None, 1, 3, n_spines]
    print("true_coords", true_coords.get_shape())                      # shape = [None, 3, None]
    distances = tf.reduce_sum((true_coords - predicted_coords) ** 2, axis=-1)  # [None, n_detectors]
    ## distance loss
    min_distance = tf.reduce_min(distances, axis=-1)            # [None]
    min_distance_loss = lambda_coord * min_distance
    ## confidence losses
    length = tf.shape(cells_coords)[1:2]
    arg_chosen_detectors = tf.argmin(distances, axis=-1)        # [None]
    indices = tf.concat([
        cells_coords[NO_BATCH],
        arg_chosen_detectors[:, tf.newaxis]], axis=-1)
    updates = tf.ones(length)
    shape = tf.cast(tf.shape(net_out[NO_BATCH, ..., 3::4]), tf.int64)
    target_confidence = tf.scatter_nd(indices, updates, shape)
    confidence_loss = (net_out[NO_BATCH, ..., 3::4] - target_confidence) ** 2 * (target_confidence * (1 - lambda_noobj) + lambda_noobj)
    # sums
    min_distance_loss_sum = tf.reduce_sum(min_distance_loss)
    confidence_loss_sum = tf.reduce_sum(confidence_loss)
    min_distance_loss_mean = tf.reduce_mean(min_distance)
    abs_distance = tf.abs(net_out[NO_BATCH, ..., 3::4] - target_confidence)
    confidence_mean_abs_distance = tf.reduce_mean(abs_distance)
    mean_distance_to_1_confidence = tf.reduce_sum(tf.where(tf.equal(target_confidence, 1), x=abs_distance, y=tf.zeros_like(abs_distance))) / tf.reduce_sum(target_confidence)
    ret = {
        "min_distance_loss_sum": min_distance_loss_sum,
        "min_distance_loss_mean": min_distance_loss_mean,
        "confidence_loss_sum": confidence_loss_sum,
        "confidence_mean_abs_distance": confidence_mean_abs_distance,
        "mean_distance_to_1_confidence": mean_distance_to_1_confidence
    }
    return ret


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
