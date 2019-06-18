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


def tensorboard_func(self):
    port = get_available_port()
    p1 = multiprocessing.Process(target=tensorboard_server_func, args=(self.logdir, port), daemon=True)
    p1.start()
    time.sleep(2)
    p2 = multiprocessing.Process(target=chromium_func, args=(port,), daemon=True)
    p2.start()
    p2.join()
    terminate_process_safe(p1)


if __name__ == "__main__":
    import argparse
    import pickle


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        help="path to the tfrecord file"
    )

    parser.add_argument(
        "-n", "--name"
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



    args = parser.parse_args()
    name_suffix = ("_" + args.name) if args.name else ""
    output_dir = time.strftime("../training_runs/%Y_%m_%d_%H_%M_%S", time.localtime()) + name_suffix
    checkpoint_dir = output_dir + "/checkpoint"
    tensorboard_dir = output_dir + "/tensorboard"

    dirpath, name = os.path.split(args.path)
    name = name.replace(".tfrecord", "")
    with open(dirpath + '/' + name + ".args", "rb") as f:
        generation_args = pickle.load(f)

    dataset = naked_dataset(args.path)
    dataset = dataset.repeat(args.n_epochs)
    dataset = dataset.shuffle(buffer_size=50)
    dataset_iterator = dataset.make_initializable_iterator()
    datapoint = dataset_iterator.get_next()

    if args.network_type is "SimpleLinearModel":
        net = network.SimpleLinearModel(generation_args, args.detectors)
    else:
        raise ValueError("Network type not recognized")

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
    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    train_op = optimizer.minimize(loss)

    ### TODO:
    ### add start tensorboard to argsparser
    ### add batch size to argsparser
    ### start tensorboard
    ### fix loss
    ### summaries

    summary_writer = tf.summary.FileWriter(tensorboard_dir)

    with tf.Session() as sess:
        sess.run([dataset_iterator.initializer, tf.global_variables_initializer()])
        try:
            while(True):
                np_loss = sess.run(loss)
                print(np_loss)
        except tf.errors.OutOfRangeError:
            pass
