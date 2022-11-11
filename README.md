# LSTM-Based Spatial Encoding: Explainable Path Planning for Time-Variant Multi-Agent Systems 
This repository contains the source code used for the LSTM-based multi-agent path planning method described in *LSTM-Based Spatial Encoding: Explainable Path Planning for Time-Variant Multi-Agent Systems* (Schlichting et al., 2021).
The code can be used as starting point for other multi-agent environments (exchanging the gym environment), following ideas outlined in the paper regarding saftey guarantees, 
or using the code for own experiments. The implementation is based on Open AI's Gym and PyTorch. The files along descriptions how to use them will be uploaded in time for the AIAA SciTech 2021 Conference.


The paper can be found using this [link](https://arc.aiaa.org/doi/10.2514/6.2021-1860). In case you use this work for your own research, please cite as:

```
@inproceedings{Schlichting2021,
author = {Marc R. Schlichting and Stefan Notter and Walter Fichter},
title = {LSTM-Based Spatial Encoding: Explainable Path Planning for Time-Variant Multi-Agent Systems},
booktitle = {AIAA Scitech 2021 Forum},
month = {Jan},
year = {2021},
doi = {10.2514/6.2021-1860},
URL = {https://arc.aiaa.org/doi/abs/10.2514/6.2021-1860},
}
```

### Installation
The setup described here has mainly been tested on Linux-based systems. A Python installation (as of November 2022, we recommend Python 3.9.13) is required. The following Python packages are required (pip is recommended for installation): **numpy** (==1.23.4), **scipy** (==1.9.3), **gym** (only basic installation, ==0.23.1), **PyTorch** (see [here](https://pytorch.org/) for installation notes, CUDA is not used in the implementation so far, ==1.13.0). Depending on the platform, other packages such as Visual Studio Build Tools (maybe required for gym) need to be installed. Test that all packages are are properly installed before proceding with the next step. To simplify the installation, the following command will install all necessary packages.

'''
pip3 install -r requirements.txt
'''

Before beginning with the training of the policy, the custom environment (located in the *drone-sim2d* folder) must be installed. For this purpose navigate into the previously mentioned folder:
```
cd drone-sim2d
```
Now use pip to install the custom environment:
```
pip3 install -e .
```
If no error occurs, the installation is successfully completed.

### Usage
All parameters can be changed within the *main* function of the *training.py* file. The overall structure of a typical PPO implementation has been adapted to work with mutli-agent environments. The training can be started using:
```
python3 training.py
```
The *logs* folders will contain the log files for each run. For each run, a unique timestamp is created which will be used for all log files and model names. Within the *logs* folder, two files are created: One parameter file that contains relevant parameters for each run and a second file that contains the average episode length as well as the average reward per log interval. The models are saved to the *models* folder after a specified number of training episodes (as defined in the beginning of the main function). 
