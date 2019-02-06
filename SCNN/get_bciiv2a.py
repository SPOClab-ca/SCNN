import mne
import numpy as np
from scipy.io import loadmat

import argparse
import wget
import tqdm
from pathlib import Path

URL = 'http://bnci-horizon-2020.eu/database/data-sets/001-2014/'
NUM_SUBJECTS = 9
ch_names = ['EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7',
            'EEG-C4', 'EEG-8', 'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16',
            'EOG-left', 'EOG-central', 'EOG-right']
ch_types = [*['eeg']*22, *['eog']*3]


def create_raw(X, y, trial_starts, artifacts, misc):
    """
    Create an instance of mne raw. TODO add artifacts labels
    """
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=misc['fs'])
    info['lowpass'] = 100.
    info['highpass'] = 0.1
    info['subject_info'] = {k: misc[k] for k in misc.keys() if k != 'fs'}
    raw = mne.io.RawArray(X.T, info)
    events = np.vstack([trial_starts, np.zeros_like(y), y]).T
    stinfo = mne.create_info(['STI 014'], raw.info['sfreq'], ['stim'])
    stim_raw = mne.io.RawArray(np.zeros((1, len(raw.times))), stinfo)
    raw.add_channels([stim_raw], force_update_info=True)
    raw.add_events(events, stim_channel='STI 014')
    
    return raw


def load_file(file):
    loaded_runs = []
    data = loadmat(file, squeeze_me=True)['data']
    for run in data:
        x = run['X'].item()
        y = run['y'].item()
        events = run['trial'].item()
        artifacts = run['artifacts'].item()
        misc = {'fs': run['fs'], 'gender': run['gender'], 'age': run['age']}
        loaded_runs.append([x, y, events, artifacts, misc])
    
    return loaded_runs


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Downloads BCI IV 2a dataset and converts matlab files '
                                                 'into raw mne trials/dataset.')
    parser.add_argument('--dl-directory', '-d', default='.')
    parser.add_argument('--dest', help='Destination to put the resulting fif (mne formatted) files.',
                        default='mne_ready/')
    parser.add_argument('--no-dl', action='store_true',
                        help="Don't download the files, they are already in the dl-directory.")
    args = parser.parse_args()

    args.dl_directory = Path(args.dl_directory)
    args.dl_directory.mkdir(parents=True, exist_ok=True)
    args.dest = Path(args.dest)
    args.dest.mkdir(parents=True, exist_ok=True)

    for subject in tqdm.trange(1, NUM_SUBJECTS+1, desc='Subject:', unit='subjects'):
        for subset in tqdm.tqdm('TE', desc='Train/Evaluate Subset'):
            f = 'A0{}{}.mat'.format(subject, subset)
            destination = (args.dest / Path(f).name).with_suffix('.raw.fif')
            if destination.exists():
                tqdm.tqdm.write('Skipping {}'.format(destination.name))
                continue
            if not args.no_dl and not (args.dl_directory / f).exists():
                wget.download(URL+f, out=str(args.dl_directory))

            tqdm.tqdm.write('Loading: ' + f)
            raws = []
            for X, y, trials, arts, misc in tqdm.tqdm(load_file(str(args.dl_directory / f)),
                                                      desc='Trials', unit='trials'):
                raws.append(create_raw(X, y, trials, arts, misc))

            raw = mne.concatenate_raws(raws)

            tqdm.tqdm.write('Raw created for: ' + f)
            tqdm.tqdm.write('Saving to: ' + str(destination))
            raw.save(str(destination), fmt='single', verbose=2, overwrite=True)
