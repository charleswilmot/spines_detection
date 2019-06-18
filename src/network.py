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
