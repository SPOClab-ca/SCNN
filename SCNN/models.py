from .layers import *


class BestSCNN(keras.Model):

    def __init__(self, inputshape, outputshape, output_classifier=(6, 170, 70)):

        temp_layers = 4
        steps = 2
        temporal = 22
        temp_pool = 10
        self.lunits = [120, 10] + list(output_classifier)
        self.activation = keras.activations.selu
        self.reg = 0.00409
        self.do = 0.7432

        convs = [inputshape[-1] // steps for _ in range(1, steps)]
        convs += [inputshape[-1] - sum(convs) + len(convs)]

        ins = keras.layers.Input(inputshape)

        conv = ExpandLayer()(ins)

        for i, c in enumerate(convs):
            conv = keras.layers.Conv2D(self.lunits[0] // len(convs), (1, c), activation=self.activation,
                                       use_bias=False, name='spatial_conv_{0}'.format(i),
                                       kernel_regularizer=keras.layers.regularizers.l2(self.reg))(conv)
        conv = keras.layers.BatchNormalization()(conv)
        conv = keras.layers.SpatialDropout2D(self.do)(conv)

        for i in range(temp_layers):
            conv = keras.layers.Conv2D(self.lunits[1], (temporal, 1), activation=self.activation,
                                       use_bias=False, name='temporal_conv_{0}'.format(i),
                                       kernel_regularizer=keras.layers.regularizers.l2(self.reg))(conv)
        conv = keras.layers.BatchNormalization()(conv)
        conv = keras.layers.AveragePooling2D((temp_pool, 1,))(conv)
        conv = keras.layers.SpatialDropout2D(self.do)(conv)

        outs = keras.layers.Flatten()(conv)

        for units in self.lunits[2:]:
            outs = keras.layers.Dense(units, activation=self.activation,
                                      kernel_regularizer=keras.layers.regularizers.l2(self.reg))(outs)
            outs = keras.layers.BatchNormalization()(outs)
            outs = keras.layers.Dropout(self.do*2)(outs)
        outs = keras.layers.Dense(outputshape, activation='softmax', name='OUT',
                                  kernel_regularizer=keras.layers.regularizers.l2(self.reg))(outs)

        super(keras.Model, self).__init__(ins, outs, name=self.__class__.__name__)

    def compile(self, **kwargs):
        extra_metrics = kwargs.pop('metrics', [])
        super().compile(optimizer='Adam', loss=keras.losses.categorical_crossentropy,
                        metrics=[keras.metrics.categorical_crossentropy, keras.metrics.categorical_accuracy,
                                 *extra_metrics], **kwargs)


class RaSCNN(keras.Model):

    def __init__(self, inputshape, outputshape, activation=keras.activations.selu, params=None):

        do = 0.298
        reg = 1.09e-3

        lunits = [6, 58]
        ret_seq = True
        temp_layers = 6
        att_depth = 4
        attention = 76
        steps = 3
        temporal = 12
        temp_pool = 12

        convs = [inputshape[-1]//steps for _ in range(1, steps)]
        convs += [inputshape[-1] - sum(convs) + len(convs)]

        ins = keras.layers.Input(inputshape)

        conv = ExpandLayer()(ins)

        for i, c in enumerate(convs):
            conv = keras.layers.Conv2D(lunits[0]//len(convs), (1, c), activation=activation,
                                       name='spatial_conv_{0}'.format(i),
                                       kernel_regularizer=keras.layers.regularizers.l2(reg))(conv)
        conv = keras.layers.BatchNormalization()(conv)
        conv = keras.layers.SpatialDropout2D(do)(conv)

        for i in range(temp_layers):
            conv = keras.layers.Conv2D(lunits[1], (temporal, 1), activation=activation,
                                       use_bias=False, name='temporal_conv_{0}'.format(i),
                                       kernel_regularizer=keras.layers.regularizers.l2(reg))(conv)
        conv = keras.layers.BatchNormalization()(conv)
        conv = keras.layers.AveragePooling2D((temp_pool, 1,))(conv)
        conv = keras.layers.SpatialDropout2D(do)(conv)
        conv = SqueezeLayer(-2)(conv)

        attn = keras.layers.Bidirectional(AttentionLSTMIn(attention,
                                                          implementation=2,
                                                          # dropout=self.do,
                                                          return_sequences=ret_seq,
                                                          alignment_depth=att_depth,
                                                          style='global',
                                                          # kernel_regularizer=keras.layers.regularizers.l2(self.reg),
                                                          ))(conv)
        conv = keras.layers.BatchNormalization()(attn)

        if ret_seq:
            conv = keras.layers.Flatten()(conv)
        outs = conv
        for units in lunits[2:]:
            outs = keras.layers.Dense(units, activation=activation,
                                      kernel_regularizer=keras.layers.regularizers.l2(reg))(outs)
            outs = keras.layers.BatchNormalization()(outs)
            outs = keras.layers.Dropout(do)(outs)
        outs = keras.layers.Dense(outputshape, activation='softmax')(outs)

        super(keras.Model, self).__init__(ins, outs, name=self.__class__.__name__)

    def compile(self, optimizer='SGD', loss=keras.losses.categorical_crossentropy,
                metrics=(keras.metrics.categorical_crossentropy, keras.metrics.categorical_accuracy), **kwargs):
        super().compile(optimizer=optimizer, **kwargs)


