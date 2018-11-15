import keras
import keras.backend as K


class ExpandLayer(keras.layers.Layer):

    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        ax = self.axis
        input_shape = list(input_shape)
        if ax < 0:
            ax = len(input_shape) + ax
        input_shape.insert(ax+1, 1)
        return tuple(input_shape)

    def call(self, inputs, **kwargs):
        return K.expand_dims(inputs, axis=self.axis)


class SqueezeLayer(ExpandLayer):

    def compute_output_shape(self, input_shape):
        ax = self.axis
        input_shape = list(input_shape)
        if ax < 0:
            ax = len(input_shape) + ax
        if input_shape[ax] == 1:
            input_shape.pop(ax)
        else:
            raise ValueError('Dimension ', ax, 'is not equal to 1!')
        return tuple(input_shape)

    def call(self, inputs, **kwargs):
        return K.squeeze(inputs, axis=self.axis)

