# SCNN

A Convolutional Neural Network model paradigm for learning to classify
M/EEG implemented with **Keras** and **MNE**. Including techniques to 
produce spatial and temporal visualizations of model insights.

First introduced in [Machine learning for MEG during speech tasks](www.nature.com/articles/s41598-019-38612-9),
if you find any of the models, or tools herein useful, we kindly request
that you cite this work.

This repository contains the layer and models proposed in the paper. As 
well as an example workflow using the popular and easy-to-access BCI
competition IV dataset 2a motor imagery dataset.

### Requirements

- python 3
- Keras
- mne
- tqdm
- numpy
- scipy
- wget

## Getting Started

### Installation

This repository can be installed through pip using the following:
````
pip install git+https://github.com/SPOClab-ca/SCNN.git
````

### Example Workflow

Once installed, start by downloading the BCI IV 2a dataset using the 
*get_bciiv2a* module.
````
python -m SCNN.get_bciiv2a
````

Next we can train the base SCNN model for a subject of choice (there are
9). Starting with subject 1:

````
python -m SCNN.train_example 1
````

The best model will be saved by default to the file `best_model.h5`. 
We can then use this model to train some activation visualizations 
using:

````
python -m SCNN.activations best_model.h5
````

## Final Notes
All the scripts have arguments that can configure their behaviour, try 
running them with `--help` for more details