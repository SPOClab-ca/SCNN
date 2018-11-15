from .layers import *


class AttentionLSTMIn(keras.layers.LSTM):
    """
    Keras LSTM layer (all keyword arguments preserved) with the addition of attention weights

    Attention weights are calculated as a function of the previous hidden state to the current LSTM step.
    Weights are applied either locally (across channels) or globally (across the entire sequence with respect to each
    channel.
    """
    ATT_STYLES = ['local', 'global']

    def __init__(self, units, alignment_depth: int = 1, style='local', alignment_units=None, implementation=2,
                 **kwargs):
        implementation = implementation if implementation > 0 else 2
        alignment_depth = max(0, alignment_depth)
        if isinstance(alignment_units, (list, tuple)):
            self.alignment_units = [int(x) for x in alignment_units]
            self.alignment_depth = len(self.alignment_units)
        else:
            self.alignment_depth = alignment_depth
            self.alignment_units = [alignment_units if alignment_units else units for _ in range(alignment_depth)]
        if style not in self.ATT_STYLES:
            raise TypeError('Could not understand style: ' + style)
        else:
            self.style = style
        super().__init__(units, implementation=implementation, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) > 2
        self.samples = input_shape[1]
        self.channels = input_shape[2]

        if self.style is self.ATT_STYLES[0]:
            # local attends over input vector
            units = [self.units + input_shape[-1]] + self.alignment_units + [self.channels]
        else:
            # global attends over the whole sequence for each feature
            units = [self.units + input_shape[1]] + self.alignment_units + [self.samples]
        self.attention_kernels = [self.add_weight(shape=(units[i-1], units[i]),
                                                name='attention_kernel_{0}'.format(i),
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
                                                trainable=True,
                                                constraint=self.kernel_constraint)
                                  for i in range(1, len(units))]

        if self.use_bias:
            self.attention_bias = [self.add_weight(shape=(u,),
                                                   name='attention_bias',
                                                   trainable=True,
                                                   initializer=self.bias_initializer,
                                                   regularizer=self.bias_regularizer,
                                                   constraint=self.bias_constraint)
                                   for u in units[1:]]
        else:
            self.attention_bias = None
        super(AttentionLSTMIn, self).build(input_shape)

    def preprocess_input(self, inputs, training=None):
        self.input_tensor_hack = inputs
        return inputs

    def step(self, inputs, states):
        h_tm1 = states[0]

        if self.style is self.ATT_STYLES[0]:
            energy = K.concatenate((inputs, h_tm1))
        elif self.style is self.ATT_STYLES[1]:
            h_tm1 = K.repeat_elements(K.expand_dims(h_tm1), self.channels, -1)
            energy = K.concatenate((self.input_tensor_hack, h_tm1), 1)
            energy = K.permute_dimensions(energy, (0, 2, 1))
        else:
            raise NotImplementedError('{0}: not implemented'.format(self.style))

        for i, kernel in enumerate(self.attention_kernels):
            energy = K.dot(energy, kernel)
            if self.use_bias:
                energy = K.bias_add(energy, self.attention_bias[i])
            energy = self.activation(energy)

        alpha = K.softmax(energy)

        if self.style is self.ATT_STYLES[0]:
            inputs = inputs * alpha
        elif self.style is self.ATT_STYLES[1]:
            alpha = K.permute_dimensions(alpha, (0, 2, 1))
            weighted = self.input_tensor_hack * alpha
            inputs = K.sum(weighted, 1)

        return super(AttentionLSTMIn, self).step(inputs, states)


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


