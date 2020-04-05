# Lightweight U-Nets

Code for paper: [Low-Memory CNNs Enabling Real-Time Ultrasound Segmentation Towards Mobile Deployment](https://ieeexplore.ieee.org/document/8999615)

Train lightweight U-Net models. Code facillitates training of the following models: 

  - [Original U-net](https://arxiv.org/abs/1505.04597) as proposed by Ronneberger et al.
  - Thin U-Nets with few feature channels per layer (with either regular or separable convolutions)
  - Thin, separable convolution U-Nets adapted for knowledge distillation

## Data

  - Data is expected as two NumPy arrays of grayscale images and corresponding binary labels with dimensions (Batch, Height, Width, Channels = 1)
  - No augmentation currently implemented, listed as TODO in ```lightweight_unet/dataloader.py```
  - ```kaggle_to_npz.py``` script is included to convert Kaggle data to numpy arrays, while removing all empty images (no salient structure) as detailed in paper

## Installation

Recommended installation wih ```conda```:  


```
$ conda env create -f environments/environment.yml 
```

Add locations of directories in which Experiment Data and Input Data are to be stored to `trainer_cfg` in ```runner.py```

**Note:** Code uses graphviz and pydot to visualize model architectures when saving experiment data and may need separate installation

## Notes on running

Main code is run from ```runner.py```, with experiments customizable through ```model_cfg``` and ```trainer_cfg``` dictionaries. Hard coded modes of operation exist: each experiment must be one of ```distillation```, ```thin_unet``` or ```original_unet```. Furthermore, experiments expect a parameter to vary (```trainer_cfg['parameter_varied']```), which must be selected as a key to one of the configuration dictionaries.

## Notes on how experiment results are saved

For each experiment, a directory is created (in the path specified in ```trainer_cfg```) with the following contents:
    
  - Sub-directory for each evaluation fold
    * Model architecture as JSON file
    * Model weights as .hdf5 file
    * Visualisation (.png) of model, containing layer names, shapes etc.
    * Training and evaluation information as .csv file. Contains various loss histories and Dice performance of model on evaluation set
        	
  - Log file as .txt detailing configuration of experiments

Finally, a .csv file is also maintained summarising the results of all experiments

              
