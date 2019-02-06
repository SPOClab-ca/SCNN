import numpy as np
import matplotlib.pyplot as plt
import argparse
import tqdm

from pathlib import Path
from keras.models import load_model
from .models import *
from mne.viz import plot_topomap
from mne.channels import read_montage


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


def maximize_activation(input, output, args, input_generator=f_decay_noise, regularizer=_regularizer):
    """
    Given input and output tensors, perform gradient ascent from data created by input generator until there is some
    semblance of convergence, or a maximum number of iterations have occurred.

    Tensors must have an input and output relationship for a previously defined (and compiled) nework.

    Some of this code taken from: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
    :param input_generator: function with a single argument for size/shape, and returns a numpy array with said shape
    :return: The new input data achieved, as np.array
    """
    activations = list()
    for filter_index in tqdm.trange(output._keras_shape[-1], desc='Filter:', unit='filters'):
        loss = K.mean(output[..., filter_index]) - regularizer(input, output)
        grads = K.gradients(loss, input)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + K.constant(1e-5))

        # this function returns the loss and grads given the input picture
        iterate = K.function([input, K.learning_phase()], [loss, grads])

        # Create a input signal distribution
        in_data = input_generator(input._keras_shape[1:])[np.newaxis, :]

        p = args.impatience
        best = -np.inf
        steps = 0
        while p > 0 and steps < args.epochs:
            loss_value, grads_value = iterate([in_data, 0])
            if args.verbose:
                tqdm.tqdm.write('Loss:' + str(loss_value))
            in_data += grads_value * args.lr
            if loss_value > best:
                best = loss_value
                p = args.impatience
            else:
                p -= 1
            steps += 1
        else:
            if args.verbose:
                tqdm.tqdm.write('Best Loss:' + str(best))
        activations.append(in_data.squeeze())

    return np.stack(activations, axis=-1)


MODELS = dict(SCNN=BestSCNN, RaSCNN=RaSCNN)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains one of the SCNN model variants using the BCI IV 2a '
                                                 'competition dataset.')
    parser.add_argument('saved_model', help='Trained model to use to maximize outputs.')
    parser.add_argument('--dest', help='Destination for produced images.', default='.')
    parser.add_argument('--hide-images', action='store_true', help="Don't show images while processing.")
    parser.add_argument('--model', help='Which model type to train', choices=MODELS.keys(), default='SCNN')
    parser.add_argument('--epochs', '-e', type=int, help='Number of epochs (maximization steps) to perform.',
                        default=10000)
    parser.add_argument('--impatience', help='Number of epochs to wait for improvement before just giving up.',
                        type=int, default=5)
    parser.add_argument('--verbose', '-v', action='store_true', help='Print verbose output while finding maximas.')
    parser.add_argument('--lr', type=float, help='Ascent rate finding maximas.', default=0.2)
    parser.add_argument('--montage', type=str, help='The channel montage, formatted as described in '
                                                    'mne.channels.read_montage. Can be path to montage file or standard'
                                                    ' listed type.', default='standard_1020')
    args = parser.parse_args()
    args.dest = Path(args.dest)

    model = load_model(args.saved_model, custom_objects=dict(ExpandLayer=ExpandLayer, SqueezeLayer=SqueezeLayer,
                                                             AttentionLSTMIn=AttentionLSTMIn), compile=True)
    if args.verbose:
        model.summary()

    # Spatial Stage Maximization
    for i, s_out in enumerate(tqdm.tqdm(
            [l for l in model.layers if 'spatial_conv' in l.name], desc='Spatial Stages', unit='layer')):
        activation = maximize_activation(model.input, s_out.output, args)
        chans = read_montage(args.montage).get_pos2d()[:activation.shape[1], :]
        # Same function is applied over time, take first slice
        activation = activation[0, :, :]
        for j in range(activation.shape[-1]):
            im, cn = plot_topomap(activation[..., j], chans, show=not args.hide_images)
            plt.title('Spatial Layer {0} Filter {1}'.format(i, j))
            if args.dest is not None:
                save_loc = args.dest / s_out.name
                save_loc.mkdir(parents=True, exist_ok=True)
                plt.savefig(str(save_loc / 'spatial_component_{0}.png'.format(i)))
            plt.clf()

    # Output Maximization wrt Temporal Layers
    for i, temporal_layer in enumerate(tqdm.tqdm(
            [l for l in model.layers if 'temporal_conv' in l.name], desc='Temporal Stages', unit='layer')):
        activation = maximize_activation(temporal_layer.input, model.output, args)
        for out_class in range(activation.shape[-1]):
            for spat_filter in range(activation.shape[-2]):
                plt.specgram(activation[..., spat_filter, out_class].squeeze(), Fs=200, cmap='bwr', NFFT=64, noverlap=50)
                plt.colorbar()
                plt.title('Output Class {0} Component {1}'.format(out_class+1, spat_filter))
                plt.ylabel('Frequency (Hz)')
                plt.xlabel('Time (s)')
                if not args.hide_images:
                    plt.show()
                if args.dest is not None:
                    directory = args.dest / temporal_layer.name / 'class_{0}'.format(out_class+1)
                    directory.mkdir(parents=True, exist_ok=True)
                    plt.savefig(str(directory / 'component_{0}'.format(spat_filter)))
                plt.clf()
