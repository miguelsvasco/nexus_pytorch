# Nexus model
Pytorch implementation of the Nexus model.

## Citation
```
@article{vasco2022leveraging,
  title={Leveraging hierarchy in multimodal generative models for effective cross-modality inference},
  author={Vasco, Miguel and Yin, Hang and Melo, Francisco S and Paiva, Ana},
  journal={Neural Networks},
  volume={146},
  pages={238--255},
  year={2022},
  publisher={Elsevier}
}
```

## Setup/Instalation
Tested on Ubuntu 16.04 LTS, CUDA 10.2:

1. Run ``` ./install_pyenv.sh ``` to install the pyenv environment (requires administrative privilige to install pyenv dependencies)
2. Add the following to your  ``` .bashrc ``` file:
 ``` 
export PATH="$HOME/.poetry/bin:$PATH"
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
  ```  
2. Activate the pyenv environment ``` pyenv activate nexus ``` (or create a ``` .python-version ``` file);
3. Upgrade pip: ``` pip install --upgrade pip```
4. Install the required dependencies ``` poetry install ```.

### Troubleshooting:
- If ``` ./install_pyenv.sh ``` fails on Ubuntu 18.04 LTS, replace in the file all entries refering to Python ```3.6.4``` with ```3.6.9```.

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
After training, place the model ``` *_checkpoint.pth.rar``` in the ``` /trained_models``` folder. To change the training hyperparameters or the network architecture, please modify the file ``` ingredients.py```.

### Generation
To generate modality information from symbolic information (labels):

#### Preliminary evaluation and Standard Evaluation
CUDA: ```python generate.py```
 
Without CUDA: ```python generate.py with gpu.cuda=False```

#### Multimodal evaluation
With CUDA:
```
python generate_image.py
python generate_sound.py
python generate_trajectory.py
```

Without CUDA:
```
python generate_image.py with gpu.cuda=False
python generate_sound.py with gpu.cuda=False
python generate_trajectory.py with gpu.cuda=False
```

### Evaluation
To evaluate _Accuracy_ and _Rank_:

#### Preliminary evaluation and Standard Evaluation

With CUDA:
```
python evaluate_accuracy.py
python evaluate_rank.py
```

Without CUDA:
```
python evaluate_accuracy.py with gpu.cuda=False
python evaluate_rank.py with gpu.cuda=False
```

#### Multimodal evaluation
With CUDA:

```
python evaluate_accuracy_image.py
python evaluate_accuracy_sound.py
python evaluate_accuracy_trajectory.py
python evaluate_accuracy_symbol.py

python evaluate_rank_image.py
python evaluate_rank_sound.py
python evaluate_rank_trajectory.py
```
Without CUDA:
```
python evaluate_accuracy_image.py with gpu.cuda=False
python evaluate_accuracy_sound.py with gpu.cuda=False
python evaluate_accuracy_trajectory.py with gpu.cuda=False
python evaluate_accuracy_symbol.py with gpu.cuda=False

python evaluate_rank_image.py with gpu.cuda=False
python evaluate_rank_sound.py with gpu.cuda=False
python evaluate_rank_trajectory.py with gpu.cuda=False
```
