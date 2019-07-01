import tensorflow as tf
from tensorflow import keras as kr


class SimpleLinearModel(kr.Model):
    def __init__(self, generation_args, n_detectors_per_cell):
        super(SimpleLinearModel, self).__init__()
        self.layer = kr.layers.Conv3D(
            filters=n_detectors_per_cell * (3 + 1),  # x,y,z + prob
            kernel_size=(generation_args.cells_size_xy, generation_args.cells_size_xy, generation_args.cells_size_z),
            strides=(generation_args.cells_stride_xy, generation_args.cells_stride_xy, generation_args.cells_stride_z),
            padding=generation_args.cells_padding,
            use_bias=True
        )

    def call(self, inputs):
        return self.layer(inputs)


class Type1Model(kr.Model):
    def __init__(self, generation_args, n_detectors_per_cell):
        # cell_size_xy, cells_stride_xy, cell_size_z, cells_stride_z
        # 64,           32,              4,           2
        super(Type1Model, self).__init__()
        self.layer = [
            kr.layers.Conv3D(
                filters=32,
                kernel_size=(4, 4, 4),
                strides=(4, 4, 2),
                padding=generation_args.cells_padding,
                use_bias=True,
                activation=tf.nn.relu
            ),
            kr.layers.Conv3D(
                filters=32,
                kernel_size=(4, 4, 1),
                strides=(4, 4, 1),
                padding=generation_args.cells_padding,
                use_bias=True,
                activation=tf.nn.relu
            ),
            kr.layers.Conv3D(
                filters=32,
                kernel_size=(2, 2, 1),
                strides=(2, 2, 1),
                padding=generation_args.cells_padding,
                use_bias=True,
                activation=tf.nn.relu
            ),
            kr.layers.Conv3D(
                filters=n_detectors_per_cell * (3 + 1),
                kernel_size=(2, 2, 1),
                strides=(1, 1, 1),
                padding=generation_args.cells_padding,
                use_bias=True
            )
        ]

    def call(self, inputs):
        for l in self.layers:
            inputs = l(inputs)
        return inputs
