import os
import tensorflow as tf
from dataset import naked_dataset
import network
from losses import get_losses
import multiprocessing
import time
import socket
import subprocess


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def get_available_port():
    port = 6006
    while is_port_in_use(port):
        port += 1
    return port


def tensorboard_server_func(logdir, port):
    subprocess.Popen(["tensorboard", "--logdir", logdir, "--port", str(port)])


def chromium_func(port):
    while not is_port_in_use(port):
        time.sleep(1)
    os.system('chromium-browser http://localhost:{} > /dev/null 2>&1'.format(port))


def terminate_process_safe(p):
    p.terminate()
    while p.is_alive():
        time.sleep(0.1)


def tensorboard_func(logdir):
    port = get_available_port()
    p1 = multiprocessing.Process(target=tensorboard_server_func, args=(logdir, port), daemon=True)
    p1.start()
    time.sleep(2)
    p2 = multiprocessing.Process(target=chromium_func, args=(port,), daemon=True)
    p2.start()
    p2.join()
    terminate_process_safe(p1)


def tensorboard(logdir):
    p1 = multiprocessing.Process(target=tensorboard_func, args=(logdir, ))
    p1.start()
    return p1


if __name__ == "__main__":
    import argparse
    import pickle


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        help="path to the tfrecord file"
    )

    parser.add_argument(
        "-n", "--name",
        help="name for the run"
    )

    parser.add_argument(
        "-d", "--detectors",
        type=int,
        default=3,
        help="number of detectors per cell"
    )

    parser.add_argument(
        "-t", "--network-type",
        default="SimpleLinearModel",
        help="Class name of the network to use"
    )

    parser.add_argument(
        "-lr", "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the training"
    )

    parser.add_argument(
        "-ne", "--n-epochs",
        type=int,
        default=100,
        help="Number of epochs to train on"
    )

    parser.add_argument(
        "-tb", "--tensorboard",
        action="store_true",
        help="Starts tensorboard"
    )


    # clear ; p3 training_local.py ../tfrecords/cell_size_032_004__cell_strides_008_002__padding_SAME__image_size_512/test3.tfrecord -n test -ne 1 -tb
    args = parser.parse_args()
    name_suffix = ("_" + args.name) if args.name else ""
    output_dir = time.strftime("../training_runs/%Y_%m_%d_%H_%M_%S", time.localtime()) + name_suffix
    checkpoint_dir = output_dir + "/checkpoint"
    tensorboard_dir = output_dir + "/tensorboard"
    os.mkdir(output_dir)
    os.mkdir(checkpoint_dir)
    os.mkdir(tensorboard_dir)


    dirpath, name = os.path.split(args.path)
    name = name.replace(".tfrecord", "")
    with open(dirpath + '/' + name + ".args", "rb") as f:
        generation_args = pickle.load(f)

    dataset = naked_dataset(args.path, stack_type=tf.float32).batch(1)
    dataset = dataset.repeat(args.n_epochs)
    dataset = dataset.shuffle(buffer_size=50)
    dataset_iterator = dataset.make_initializable_iterator()
    datapoint = dataset_iterator.get_next()

    if args.network_type is "SimpleLinearModel":
        net = network.SimpleLinearModel(generation_args, args.detectors)
    else:
        raise ValueError("Network type not recognized")

    inp = datapoint["stack"]
    net_out = net(inp)
    cells_coords = tf.stack([
        datapoint["cells_coords_x"],
        datapoint["cells_coords_y"],
        datapoint["cells_coords_z"]], axis=-1)
    within_cells_coords = tf.stack([
        datapoint["within_cells_coords_x"],
        datapoint["within_cells_coords_y"],
        datapoint["within_cells_coords_z"]], axis=-1)
    lambda_coord = 5
    lambda_noobj = 0.05
    min_distance_loss, confidence_loss = get_losses(net_out, cells_coords, within_cells_coords, lambda_coord, lambda_noobj)
    min_distance_summary = tf.summary.scalar("min_distance_loss", min_distance_loss)
    confidence_summary = tf.summary.scalar("confidence_loss", confidence_loss)
    # noobj_summary = tf.summary.scalar("noobj_loss", noobj_loss)
    summary = tf.summary.merge([min_distance_summary, confidence_summary])


    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    train_op = optimizer.minimize(min_distance_loss + confidence_loss)

    ### TODO:
    ### summaries

    with tf.summary.FileWriter(tensorboard_dir) as summary_writer:
        if args.tensorboard:
            tensorboard_process = tensorboard(tensorboard_dir)
        with tf.Session() as sess:
            sess.run([dataset_iterator.initializer, tf.global_variables_initializer()])
            try:
                iteration = 0
                while(True):
                    np_loss, _, np_summary = sess.run([(min_distance_loss, confidence_loss), train_op, summary])
                    print(np_loss)
                    summary_writer.add_summary(np_summary, global_step=iteration)
                    iteration += 1
            except tf.errors.OutOfRangeError:
                pass
        if args.tensorboard:
            terminate_process_safe(tensorboard_process)
