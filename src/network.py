import tensorflow as tf
from tensorflow import keras as kr
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.framework import ops


class SimpleLinearModel(kr.Sequential):
    def __init__(self, generation_args, n_detectors_per_cell, name=None):
        super(SimpleLinearModel, self).__init__(name=name)
        self.layer = kr.layers.Conv3D(
            filters=n_detectors_per_cell * (3 + 1),  # x,y,z + prob
            kernel_size=(generation_args.cells_size_xy, generation_args.cells_size_xy, generation_args.cells_size_z),
            strides=(generation_args.cells_stride_xy, generation_args.cells_stride_xy, generation_args.cells_stride_z),
            padding=generation_args.cells_padding,
            use_bias=True,
            input_shape=(generation_args.image_size, generation_args.image_size, None, 1)
        )
        self.add(self.layer)
        # self.built = False
        # if not name:
        #     prefix = 'sequential_'
        #     name = prefix + str(K.get_uid(prefix))
        # self._name = name
        # self._scope = None
        # self._reuse = None
        # self._base_name = name
        # self._graph = ops.get_default_graph()
        # self._input_layers = []
        # self._dtype = None
        # self._activity_regularizer = None

    # def call(self, inputs):
    #     return self.layer(inputs)
