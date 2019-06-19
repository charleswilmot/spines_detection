import os
import tensorflow as tf
import pickle


stack_feature_description = {
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),
    'depth': tf.FixedLenFeature([], tf.int64),
    'true_coords_x': tf.VarLenFeature(tf.float32),
    'true_coords_y': tf.VarLenFeature(tf.float32),
    'true_coords_z': tf.VarLenFeature(tf.int64),
    'cells_coords_x': tf.VarLenFeature(tf.int64),
    'cells_coords_y': tf.VarLenFeature(tf.int64),
    'cells_coords_z': tf.VarLenFeature(tf.int64),
    'within_cells_coords_x': tf.VarLenFeature(tf.float32),
    'within_cells_coords_y': tf.VarLenFeature(tf.float32),
    'within_cells_coords_z': tf.VarLenFeature(tf.float32),
    'stack': tf.FixedLenFeature([], tf.string)
}


def _get_parsing_function(path, stack_type):
    dirpath, name = os.path.split(path)
    name = name.replace(".tfrecord", "")
    with open(dirpath + '/' + name + ".args", "rb") as f:
        generation_args = pickle.load(f)
    image_size = generation_args.image_size

    def _parse_example_function(example):
        # Parse the input tf.Example proto using the dictionary above.
        datadict = tf.parse_single_example(example, stack_feature_description)
        shape = tf.stack([image_size, image_size, -1, 1], axis=0)
        datadict["stack"] = tf.reshape(tf.image.decode_jpeg(datadict["stack"]), shape)
        if stack_type is tf.float32:
            datadict["stack"] = tf.cast(datadict["stack"], tf.float32) / 127.5 - 1
        datadict["true_coords_x"] = tf.sparse_tensor_to_dense(datadict["true_coords_x"])
        datadict["true_coords_y"] = tf.sparse_tensor_to_dense(datadict["true_coords_y"])
        datadict["true_coords_z"] = tf.sparse_tensor_to_dense(datadict["true_coords_z"])
        datadict["cells_coords_x"] = tf.sparse_tensor_to_dense(datadict["cells_coords_x"])
        datadict["cells_coords_y"] = tf.sparse_tensor_to_dense(datadict["cells_coords_y"])
        datadict["cells_coords_z"] = tf.sparse_tensor_to_dense(datadict["cells_coords_z"])
        datadict["within_cells_coords_x"] = tf.sparse_tensor_to_dense(datadict["within_cells_coords_x"])
        datadict["within_cells_coords_y"] = tf.sparse_tensor_to_dense(datadict["within_cells_coords_y"])
        datadict["within_cells_coords_z"] = tf.sparse_tensor_to_dense(datadict["within_cells_coords_z"])
        return datadict

    return _parse_example_function


def naked_dataset(path, stack_type=tf.uint8):
    dataset = tf.data.TFRecordDataset(path)
    return dataset.map(_get_parsing_function(path, stack_type))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        help="path to the tfrecord file"
    )

    args = parser.parse_args()

    dataset = naked_dataset(args.path)
    print(dataset.output_shapes, '\n')
    print(dataset.output_classes, '\n')
    print(dataset.output_types, '\n')
