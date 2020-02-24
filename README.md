Unsupervised Audio Spectrogram Compression using Vector Quantized Autoencoders
=================================================

master thesis project

The work was published [here](http://kth.diva-portal.org/smash/record.jsf?pid=diva2%3A1376201&dswid=4801)


tested on Python 3.6.5

Requirements:
- `tensorflow==1.13.0rc2`
- `dm-sonnet==1.27`
- `tensorflow-probability==0.5.0`

Datasets:
- cifar10

OR

- dataset of 4-second sound files

YAML-files describing each experiment type can be found in `experiments/`. The `minimal` experiments used only a minimal subset of the data and are intended for testing. 

### Run cifar experiment (subset of data; for testing)
`python train.py -f experiments/cifar10-minimal.yaml`

### Run cifar experiment (whole dataset)
`python train.py -f experiments/cifar10-full.yaml`

### Run cifar experiment (whole dataset)
`python train.py -f experiments/cifar10-full.yaml`