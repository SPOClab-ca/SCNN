import mne
import argparse
from pathlib import Path
from SCNN.models import *
from keras.callbacks import *
from keras.models import load_model
from keras.utils import to_categorical


def zscore(data: np.ndarray, axis=-1):
    return (data - data.mean(axis, keepdims=True)) / (data.std(axis, keepdims=True) + 1e-12)


def emwa(data: np.ndarray, alpha=0.1):
    """
    Fast emwa mostly taken from:
    https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
    :param data: Assuming at least three axes, with the first definitely the batch axis.
    :param axis:
    :return:
    """
    alpha_rev = 1 - alpha
    n = data.shape[-1]
    pows = alpha_rev ** (np.arange(n + 1))

    scale_arr = 1 / pows[:-1]
    offset = data[..., 0, np.newaxis] * pows[1:]
    pw0 = alpha * alpha_rev ** (n - 1)

    mult = data * pw0 * scale_arr
    cumsums = mult.cumsum(axis=-1)
    out = offset + cumsums * scale_arr[::-1]
    return out


def exp_moving_whiten(data: np.ndarray, factor_new=0.01):
    """
    Exponentially whitening

    Some code in this function taken from:
    https://github.com/robintibor/braindecode/blob/master/braindecode/datautil/signalproc.py
    :param data:
    :param factor_new:
    :return:
    """
    meaned = emwa(data, alpha=factor_new).mean()
    demeaned = data - meaned
    squared = demeaned * demeaned
    sq_mean = emwa(squared, alpha=factor_new).mean()
    return demeaned / np.maximum(1e-4, np.sqrt(sq_mean))


def load_data(args):
    def arrays(path):
        raw = mne.io.read_raw_fif(str(path), preload=True)
        picks = mne.pick_types(raw.info, eog=True, stim=False, meg=False, eeg=True)
        events = mne.find_events(raw)
        epochs = mne.Epochs(raw, events, tmin=args.tmin, tmax=args.tmin + args.tlen - 1 / raw.info['sfreq'],
                            preload=True, picks=picks, baseline=None)
        x = epochs.get_data()
        return exp_moving_whiten(x).transpose((0, 2, 1)), to_categorical(epochs.events[:, -1] - 1, 4)

    return [arrays(args.toplevel / 'A0{}{}.raw.fif'.format(args.subject, i)) for i in 'TE']


MODELS = dict(SCNN=BestSCNN, RaSCNN=RaSCNN)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains one of the SCNN model variants using the BCI IV 2a '
                                                 'competition dataset.')
    parser.add_argument('subject', help='Which subject (1-9) to train the model for.', type=int, choices=range(1, 10))
    parser.add_argument('--model', help='Which model to train', choices=MODELS.keys(), default='SCNN')
    parser.add_argument('--toplevel', default='mne_ready/', type=Path, help='Location with all needed fif files.')
    parser.add_argument('--tmin', default=-0.5, type=float)
    parser.add_argument('--tlen', default=4.5, type=float)
    parser.add_argument('--no-test', action='store_true', help="Don't test the model after training.")
    parser.add_argument('--test-output', help='Name of file containing two numpy arrays, "predictions" and "truth"'
                                              'for future performance analysis.', default='results.npz')
    parser.add_argument('--save-model', help='File name for the best validation loss trained model. If exists, the '
                                             'model is tested.', default='best_model.h5',
                                             type=str)
    parser.add_argument('--force-train', '-f', action='store_true', help='Train even if model exists.')
    parser.add_argument('--lr', help='Learning rate, defaults to best found for SCNN.', type=float, default=5e-4)
    parser.add_argument('--epochs', '-e', type=int, help='Number of epochs (runs through the data) to perform.',
                        default=400)
    parser.add_argument('--batch-size', type=int, default=60)
    parser.add_argument('--impatience', help='Number of epochs to wait for improvement before lowering learning rate.',
                        type=int, default=50)
    args = parser.parse_args()

    print('Training {} with data from subject {}. Best model saved to {}.'.format(
        args.model, args.subject, args.save_model))

    training_data, testing_data = load_data(args)

    if not Path(args.save_model).exists() or args.force_train:
        model = MODELS[args.model](training_data[0].shape[1:], 4)
        model.compile(optimizer=keras.optimizers.Adam(args.lr), loss=keras.losses.categorical_crossentropy,
                      metrics=[keras.metrics.categorical_crossentropy, keras.metrics.categorical_accuracy])
        model.summary()
        model.fit(*training_data, batch_size=args.batch_size, epochs=args.epochs, validation_split=0.2, callbacks=[
            ReduceLROnPlateau(patience=args.impatience, factor=0.5),
            ModelCheckpoint(args.save_model, save_best_only=True, save_weights_only=False, verbose=True,
                            monitor='val_categorical_accuracy')])
        del model

    if not args.no_test:
        print('Loading best model from ', args.save_model)
        model = load_model(args.save_model, custom_objects=dict(ExpandLayer=ExpandLayer, SqueezeLayer=SqueezeLayer,
                                                                AttentionLSTMIn=AttentionLSTMIn), compile=True)
        print('Loaded Model.')
        model.summary()
        predictions = model.predict(x=testing_data[0], batch_size=args.batch_size, verbose=True)
        print('\nPrediction Accuracy: {:.2f}'.format(100 * np.mean(
            predictions.argmax(axis=-1) == testing_data[1].argmax(axis=-1))))
        np.savez(args.test_output, predictions=predictions, truth=testing_data[1])
        print('Saved results to {}'.format(args.test_output))

