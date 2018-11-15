import numpy as np
from .models import *


def f_decay_noise(shape, decay_factor=0.5, cutoff_f=None):
    """
    Create noise that has spectral activity decrease with 1/f
    :param shape:
    :param cutoff_f: enforce a cut-off frequency
    :return:
    """
    single_dims = tuple(i for i, n in enumerate(shape) if n == 1)
    shape = list(shape)
    for d in single_dims:
        shape.pop(d)
    uneven = shape[0] % 2
    x = np.random.randn(shape[0]//2 + 1 + uneven, *shape[1:]) + 1j*np.random.rand(shape[0]//2 + 1 + uneven, *shape[1:])
    S = np.power(np.arange(x.shape[0]) + 1., decay_factor)[:, np.newaxis]
    x[0, :] = 0.0
    y = np.real(np.fft.irfft(x/S, axis=0))
    if uneven:
        y = y[:-1]
    for d in single_dims:
        y = np.expand_dims(y, axis=d)
    return y


def _regularizer(input, output, l2=0.05, spectral_profile_penalty=0.05, spectral_profile_deg=0.5):
    # TODO profile regularization
    return l2 * K.mean(K.square(input))


def maximize_activation(input, output, lr=0.2, input_generator=f_decay_noise, regularizer=_regularizer, max_steps=1e4,
                        verbose=True, patience=5):
    """
    Given input and output tensors, perform gradient ascent from data created by input generator until there is some
    semblance of convergence, or a maximum number of iterations have occurred.

    Tensors must have an input and output relationship for a previously defined (and compiled) nework.

    Some of this code taken from: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
    :param input_generator: function with a single argument for size/shape, and returns a numpy array with said shape
    :return: The new input data achieved, as np.array
    """
    activations = list()
    for filter_index in range(output._keras_shape[-1]):
        if verbose:
            print('Filter: {0}/{1}'.format(filter_index+1, output._keras_shape[-1]))
        loss = K.mean(output[..., filter_index]) - _regularizer(input, output)
        grads = K.gradients(loss, input)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + K.constant(1e-5))

        # this function returns the loss and grads given the input picture
        iterate = K.function([input, K.learning_phase()], [loss, grads])

        # Create a noise input signal
        in_data = input_generator(input._keras_shape[1:])[np.newaxis, :]

        p = patience
        best = -np.inf
        steps = 0.0
        while p > 0 and steps < max_steps:
            loss_value, grads_value = iterate([in_data, 0])
            if verbose:
                print('Loss:', loss_value)
            in_data += grads_value * lr
            if loss_value > best:
                best = loss_value
                p = 5
            else:
                p -= 1
            steps += 1
        else:
            if verbose:
                print('Best Loss:', best)
        activations.append(in_data.squeeze())

    return np.stack(activations, axis=-1)


def extract_spatial_stage(model: BestSCNN):
    return model.input, [l for l in model.layers if 'spatial_' in l.name][-1]
