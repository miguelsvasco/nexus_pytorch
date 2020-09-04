# nexus_pytorch
Pytorch implementation of the Nexus model.

## Setup/Instalation
1. Run ``` ./install_pyenv.sh ``` to install the pyenv environment;
2. Activate the pyenv environment ``` pyenv activate nexus ``` (or create a ``` .python-version ``` file);
3. Install the required dependencies ``` poetry install ```.

## Download MHD dataset
To download the "Multimodal Handwritten Digits" Dataset run:
```
cd nexus_pytorch/scenarios/
./get_mhd_dataset.sh    
```

## Download Pretrained models for Evaluation
To download the pretrained models (autoencoders, classifiers, Nexus and baselines) run:
```
cd nexus_pytorch/evaluation/
./get_pretrained_models.sh    
```

## Experiments

### Training
To train a model with CUDA:
```
python train.py
```

To train a model without CUDA:
```
python train.py with gpu.cuda=False
```
After training, place the model ``` *_checkpoint.pth.rar``` in the ``` /trained_models``` folder.

### Generation
To generate modality information from symbolic information (labels):

#### Preliminary evaluation
```
python generate.py
```
#### Multimodal evaluation
```
python generate_image.py
python generate_sound.py
python generate_trajectory.py
```

### Evaluation
To evaluate _Coherence_ and _Dissimilarity_:

#### Preliminary evaluation
```
python evaluate_coherence.py
python evaluate_dissimilarity.py
```
#### Multimodal evaluation
```
python evaluate_coherence_image.py
python evaluate_coherence_sound.py
python evaluate_coherence_trajectory.py
python evaluate_coherence_symbol.py

python evaluate_dissimilarity_image.py
python evaluate_dissimilarity_sound.py
python evaluate_dissimilarity_trajectory.py
```
