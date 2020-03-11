## Unsupervised Audio Spectrogram Compression using Vector Quantized Autoencoders

| [report](http://kth.diva-portal.org/smash/record.jsf?pid=diva2%3A1376201&dswid=4801) |


### Overview 
Tensorflow implementation of Unsupervised Audio Spectrogram Compression using Vector Quantized Autoencoders. Using this framework allows for compressing a short `.wav` sound file into a compact, discrete representation, and reconstruct it again. The method relies on an input pipeline which preprocesses the sound into an intermediate "spectrogram" representation, as well as an estimated inverse operation for post-processing.

![error-freq-reponse](images/error-freq-response.jpg)

## Getting started
### Install dependencies
#### Requirements
- Python 3.6.5
- tensorflow==1.13.0rc2
- dm-sonnet==1.27
- tensorflow-probability==0.5.0
```
pip install -r requirements.txt
```

### Training
An experiment setup YAML is required for `train.py`. The setup used in the report can be found in `experiments/`. The `minimal` experiments are intended for debugging. 

`python train.py -f experiments/nsynth-full.yaml`

### Evaluation
`evaluate.py` runs the testset to evalute the predictive performance of a trained model.

### Prediction
`predict.py` compresses/reconstructs a new sound file.

### Dataset pipelines:
- `.wav` soundfiles, 4-seconds ([Nsynth](https://magenta.tensorflow.org/datasets/nsynth#files))
- CIFAR10
- MNIST

### Results (please refer to [report](http://kth.diva-portal.org/smash/record.jsf?pid=diva2%3A1376201&dswid=4801)):
Nsynth
![error-val-nsynth](images/error-val-nsynth.png)

CIFAR10
![error-val-cifar10](images/error-val-cifar10.png)

### Citation
```
@mastersthesis{HansenVedal1376201,
	title = {Unsupervised Audio Spectrogram Compression using Vector Quantized Autoencoders},	
	author = {Hansen Vedal, Amund},
	institution = {KTH, School of Electrical Engineering and Computer Science (EECS)},
	pages = {75},
	year = {2019}
}
```

